"""
End-to-End Pipeline Runner for LinkedIn Job Market Prediction & Feature Prioritization.
Executes EDA, preprocessing, model training, evaluation, and the proposed research extension.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, preprocessing, visualization

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def run_eda(df: pd.DataFrame):
    """Generates all EDA visualizations."""
    logger.info("--- STAGE 1: Generating EDA Visualizations ---")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating target distribution plot...")
    visualization.plot_target_distribution(df, config.FIGURES_DIR / "target_distribution.png")

    logger.info("Generating missing values plot...")
    visualization.plot_missing_values(df, config.FIGURES_DIR / "missing_values.png")

    logger.info("Generating salary and work type plot...")
    visualization.plot_salary_and_work_type(df, config.FIGURES_DIR / "salary_and_work_type.png")
    logger.info("EDA visualizations saved to %s", config.FIGURES_DIR)


def run_extension(df: pd.DataFrame):
    """Executes the proposed temporal skills dynamics research extension."""
    logger.info("--- STAGE 2: Executing Proposed Research Extension ---")
    from src import extension

    logger.info("Computing monthly job posting trends...")
    df_temporal = extension.compute_temporal_posting_trends(df)

    logger.info("Generating temporal trends plot...")
    visualization.plot_temporal_trends(df_temporal, config.FIGURES_DIR / "temporal_trends.png")

    logger.info("Analyzing emerging vs saturated skill trajectories...")
    # Use a representative sample of 25,000 descriptions for rapid, robust extraction
    sample_size = min(25000, len(df))
    df_sample = df.sample(n=sample_size, random_state=config.RANDOM_SEED)
    skills_df = extension.analyze_emerging_skills(df_sample)

    logger.info("Generating emerging skills visualization...")
    visualization.plot_emerging_skills(skills_df, config.FIGURES_DIR / "emerging_skills.png")
    logger.info("Research extension completed successfully.")


def run_preprocessing_and_training(df: pd.DataFrame, max_samples: int = 15000):
    """Preprocesses text, splits data, builds vocabulary, trains CNN model, and evaluates."""
    logger.info("--- STAGE 3: Preprocessing, Model Training & Evaluation ---")
    import torch
    from torch.utils.data import DataLoader
    from src.model import MarketSignalCNN
    from src import train, evaluation

    # Step 1: Target construction and filtering
    df_clean = preprocessing.prepare_target(df)

    # Subsample for training if specified to balance GPU/CPU throughput
    if max_samples and len(df_clean) > max_samples:
        logger.info("Subsampling %d records for model training/evaluation...", max_samples)
        df_clean = df_clean.sample(n=max_samples, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # Clean description text
    logger.info("Cleaning job descriptions (%d records)...", len(df_clean))
    df_clean["cleaned_desc"] = df_clean["description"].apply(preprocessing.clean_text)

    # Split dataset
    train_df, val_df, test_df = preprocessing.split_dataset(df_clean)

    # Build vocabulary strictly on training split
    logger.info("Building token vocabulary on training split only...")
    vocab = preprocessing.TextVocabulary(max_vocab_size=config.MAX_VOCAB_SIZE, min_freq=config.MIN_WORD_FREQ)
    vocab.fit(train_df["cleaned_desc"].tolist())
    vocab.save(config.RESULTS_DIR / "vocab.json")

    # Encode token sequences
    logger.info("Encoding token sequences...")
    x_train = [vocab.encode(t, max_length=config.MAX_SEQ_LENGTH) for t in train_df["cleaned_desc"]]
    y_train = train_df[config.TARGET_BINARY_COL].tolist()

    x_val = [vocab.encode(t, max_length=config.MAX_SEQ_LENGTH) for t in val_df["cleaned_desc"]]
    y_val = val_df[config.TARGET_BINARY_COL].tolist()

    x_test = [vocab.encode(t, max_length=config.MAX_SEQ_LENGTH) for t in test_df["cleaned_desc"]]
    y_test = test_df[config.TARGET_BINARY_COL].tolist()

    # Create PyTorch datasets & dataloaders
    train_dataset = train.JobPostingsDataset(x_train, y_train)
    val_dataset = train.JobPostingsDataset(x_val, y_val)
    test_dataset = train.JobPostingsDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Device detection
    device = train.get_device()

    # Model instantiation
    logger.info("Instantiating MarketSignalCNN model...")
    model = MarketSignalCNN(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        filter_sizes=config.CNN_FILTER_SIZES,
        num_filters=config.NUM_FILTERS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    # Training with early stopping
    history = train.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.LEARNING_RATE,
        max_epochs=15,  # Efficient convergence with patience 7
        patience=config.EARLY_STOPPING_PATIENCE,
        model_save_path=config.MODELS_DIR / "best_market_signal_cnn.pt"
    )

    # Plot training curves
    visualization.plot_training_curves(history, config.FIGURES_DIR / "training_curves.png")

    # Load best checkpoint for test evaluation
    best_checkpoint = torch.load(config.MODELS_DIR / "best_market_signal_cnn.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    logger.info("Loaded best model checkpoint from epoch %d (val_loss=%.4f)",
                best_checkpoint["epoch"], best_checkpoint["val_loss"])

    # Test set evaluation
    criterion = torch.nn.BCEWithLogitsLoss()
    _, test_acc, test_auc, test_probs, test_targets = train.evaluate(model, test_loader, criterion, device)

    metrics = evaluation.compute_classification_metrics(test_targets, test_probs, threshold=0.5)
    logger.info("Test Evaluation Results:\nAccuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
                metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"])

    # Save metrics
    evaluation.save_metrics_to_csv(metrics, config.EVALUATION_METRICS_PATH)

    # Confusion matrix and ROC/PR plots
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_targets, (test_probs >= 0.5).astype(int))
    visualization.plot_confusion_matrix(cm, config.FIGURES_DIR / "confusion_matrix.png")
    visualization.plot_roc_and_pr_curves(test_targets, test_probs, config.FIGURES_DIR / "roc_pr_curves.png")

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    t_start = time.time()
    logger.info("Starting End-to-End Research Implementation Pipeline...")
    df_raw = preprocessing.load_raw_dataset(config.RAW_DATA_PATH)
    run_eda(df_raw)
    run_extension(df_raw)
    try:
        run_preprocessing_and_training(df_raw)
    except ImportError as e:
        logger.warning("Deep learning packages (torch/scikit-learn) pending install: %s", e)
    logger.info("Total execution elapsed time: %.2f seconds", time.time() - t_start)
