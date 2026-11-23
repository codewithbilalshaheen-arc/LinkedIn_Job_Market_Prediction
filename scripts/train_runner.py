"""
End-to-end model training, validation checkpointing, and evaluation script.
Operates on preprocessed npy arrays from Dataset/processed/ and outputs all model checkpoints,
evaluation metrics, and figures.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainRunner")


def run():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src import config, evaluation, visualization
    from src.model import MarketSignalCNN
    from src import train

    t0 = time.time()
    logger.info("Initializing Model Training Suite...")

    # Load preprocessed arrays
    proc_dir = config.PROCESSED_DATA_DIR
    logger.info("Loading preprocessed arrays from %s...", proc_dir)

    x_train = np.load(proc_dir / "x_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    x_val = np.load(proc_dir / "x_val.npy")
    y_val = np.load(proc_dir / "y_val.npy")
    x_test = np.load(proc_dir / "x_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")

    with open(config.RESULTS_DIR / "vocab.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    vocab_size = len(vocab_data["word2idx"])

    logger.info("Loaded partitions: Train=%d, Val=%d, Test=%d, VocabSize=%d",
                len(x_train), len(x_val), len(x_test), vocab_size)

    # For fast convergence while preserving statistical significance,
    # we can train on a strong subset (e.g., 20,000 train, 4,000 val, 5,000 test)
    # or the entire dataset if compute permits.
    max_train_samples = 25000
    if len(x_train) > max_train_samples:
        logger.info("Subsampling %d training observations for rapid, robust CPU/GPU convergence...", max_train_samples)
        indices = np.random.RandomState(config.RANDOM_SEED).permutation(len(x_train))[:max_train_samples]
        x_train = x_train[indices]
        y_train = y_train[indices]

    # Convert to PyTorch Tensors
    train_dataset = TensorDataset(torch.from_numpy(x_train).long(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(x_val).long(), torch.from_numpy(y_val).float())
    test_dataset = TensorDataset(torch.from_numpy(x_test).long(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    device = train.get_device()

    # Instantiate MarketSignalCNN model
    model = MarketSignalCNN(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        filter_sizes=config.CNN_FILTER_SIZES,
        num_filters=config.NUM_FILTERS,
        dense_units=config.DENSE_UNITS,
        dropout_rate=config.DROPOUT_RATE
    ).to(device)

    logger.info("Model Architecture:\n%s", model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Trainable Parameters: %d", total_params)

    # Train model
    history = train.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.LEARNING_RATE,
        max_epochs=12,
        patience=config.EARLY_STOPPING_PATIENCE,
        model_save_path=config.MODELS_DIR / "best_market_signal_cnn.pt"
    )

    # Plot training curves
    visualization.plot_training_curves(history, config.FIGURES_DIR / "training_curves.png")
    logger.info("Saved training curves to %s", config.FIGURES_DIR / "training_curves.png")

    # Load best checkpoint
    best_checkpoint = torch.load(config.MODELS_DIR / "best_market_signal_cnn.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    logger.info("Loaded best model checkpoint from Epoch %d (val_loss=%.4f)",
                best_checkpoint["epoch"], best_checkpoint["val_loss"])

    # Test set evaluation
    criterion = nn.BCEWithLogitsLoss()
    _, test_acc, test_auc, test_probs, test_targets = train.evaluate(model, test_loader, criterion, device)

    metrics = evaluation.compute_classification_metrics(test_targets, test_probs, threshold=0.5)
    logger.info("\n--- TEST EVALUATION RESULTS ---")
    logger.info("Accuracy:             %.4f (Paper reported: ~0.824)", metrics["accuracy"])
    logger.info("Precision (Class 1):  %.4f (Paper reported: ~0.810)", metrics["precision"])
    logger.info("Recall (Class 1):     %.4f (Paper reported: ~0.830)", metrics["recall"])
    logger.info("F1-Score (Class 1):   %.4f (Paper reported: ~0.820)", metrics["f1_score"])
    logger.info("ROC-AUC:              %.4f (Paper reported: ~0.880)", metrics["roc_auc"])
    logger.info("Total Test Samples:   %d (Paper claimed: 24,823, Matrix Sum: 9,930)", metrics["test_samples"])

    # Save metrics to CSV and JSON
    evaluation.save_metrics_to_csv(metrics, config.EVALUATION_METRICS_PATH)

    # Confusion matrix and ROC/PR curves
    cm = np.array([
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]]
    ])
    visualization.plot_confusion_matrix(cm, config.FIGURES_DIR / "confusion_matrix.png")
    visualization.plot_roc_and_pr_curves(test_targets, test_probs, config.FIGURES_DIR / "roc_pr_curves.png")

    logger.info("Completed Training & Evaluation in %.2f seconds!", time.time() - t0)


if __name__ == "__main__":
    run()
