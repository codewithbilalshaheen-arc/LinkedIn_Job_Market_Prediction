"""
Standalone script to execute full data preprocessing and serialization.
Guarantees zero data leakage and saves preprocessed numpy arrays and vocabulary.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import config, preprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PreprocessingPipeline")

if __name__ == "__main__":
    t0 = time.time()
    logger.info("Executing Preprocessing Pipeline...")

    # Load raw dataset
    df = preprocessing.load_raw_dataset()

    # Target formulation
    df_clean = preprocessing.prepare_target(df)

    # Clean job descriptions
    logger.info("Cleaning job descriptions (%d records)...", len(df_clean))
    df_clean["cleaned_desc"] = df_clean["description"].apply(preprocessing.clean_text)

    # Perform stratified split
    train_df, val_df, test_df = preprocessing.split_dataset(df_clean)

    # Fit vocabulary strictly on training split
    logger.info("Fitting vocabulary on training split...")
    vocab = preprocessing.TextVocabulary(max_vocab_size=config.MAX_VOCAB_SIZE, min_freq=config.MIN_WORD_FREQ)
    vocab.fit(train_df["cleaned_desc"].tolist())

    # Save vocabulary
    vocab_path = config.RESULTS_DIR / "vocab.json"
    vocab.save(vocab_path)

    # Encode token sequences
    logger.info("Encoding sequences to token IDs (max_length=%d)...", config.MAX_SEQ_LENGTH)
    x_train = np.array([vocab.encode(t, max_length=config.MAX_SEQ_LENGTH) for t in train_df["cleaned_desc"]], dtype=np.int32)
    y_train = train_df[config.TARGET_BINARY_COL].values.astype(np.float32)

    x_val = np.array([vocab.encode(t, max_length=config.MAX_SEQ_LENGTH) for t in val_df["cleaned_desc"]], dtype=np.int32)
    y_val = val_df[config.TARGET_BINARY_COL].values.astype(np.float32)

    x_test = np.array([vocab.encode(t, max_length=config.MAX_SEQ_LENGTH) for t in test_df["cleaned_desc"]], dtype=np.int32)
    y_test = test_df[config.TARGET_BINARY_COL].values.astype(np.float32)

    # Save processed arrays to disk
    out_dir = config.PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "x_train.npy", x_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "x_val.npy", x_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "x_test.npy", x_test)
    np.save(out_dir / "y_test.npy", y_test)

    summary = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "vocab_size": len(vocab),
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "embedding_dim": config.EMBEDDING_DIM,
        "train_class_distribution": train_df[config.TARGET_BINARY_COL].value_counts(normalize=True).to_dict(),
        "elapsed_seconds": round(time.time() - t0, 2)
    }

    with open(config.RESULTS_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Preprocessing complete! Saved arrays to %s. Total time: %.2fs", out_dir, time.time() - t0)
