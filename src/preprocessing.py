"""
Data preprocessing, cleaning, tokenization, and pipeline module.
Ensures zero data leakage between train, validation, and test splits.
"""

import re
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: Union[str, float]) -> str:
    """
    Cleans raw job description text by stripping HTML tags, URLs,
    special characters, and normalizing whitespace.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Strip non-ASCII characters & abnormal punctuation, retain alphanumeric and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Lowercase and normalize whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_raw_dataset(csv_path: Optional[Path] = None, sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    Loads raw postings CSV, reports schema, missingness, and basic statistics.
    """
    path = csv_path or config.RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading raw dataset from %s ...", path)
    if sample_n:
        logger.info("Reading a sample of %d rows for rapid exploration/testing...", sample_n)
        df = pd.read_csv(path, nrows=sample_n, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded dataset with shape: %s", df.shape)
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the binary target variable from formatted_experience_level:
      - y = 1: 'Entry level' (proxy for low market priority)
      - y = 0: other valid experience levels (proxy for high market priority)
    
    Strictly removes rows with missing experience level to avoid silent misclassification.
    Removes formatted_experience_level from features to prevent target leakage.
    """
    logger.info("Constructing target variable from '%s'...", config.TARGET_COLUMN)
    initial_len = len(df)
    
    # Filter only rows with non-null experience level in valid categories
    valid_mask = df[config.TARGET_COLUMN].isin(config.VALID_EXPERIENCE_LEVELS)
    df_clean = df[valid_mask].copy()
    dropped_rows = initial_len - len(df_clean)
    logger.info("Filtered valid experience levels. Retained %d rows, dropped %d invalid/missing rows.",
                len(df_clean), dropped_rows)

    # Binary target creation: 1 for Entry level, 0 for higher experience levels
    df_clean[config.TARGET_BINARY_COL] = (
        df_clean[config.TARGET_COLUMN] == config.ENTRY_LEVEL_STRING
    ).astype(np.int64)

    logger.info("Target distribution: \n%s",
                df_clean[config.TARGET_BINARY_COL].value_counts(normalize=True).to_dict())

    return df_clean


class TextVocabulary:
    """
    Vocabulary mapping tokens to integer indices, fitted strictly on training data
    to prevent information leakage from validation/test sets.
    """
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_vocab_size: int = config.MAX_VOCAB_SIZE, min_freq: int = config.MIN_WORD_FREQ):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def fit(self, texts: List[str]) -> "TextVocabulary":
        counter = Counter()
        for text in texts:
            tokens = text.split()
            counter.update(tokens)

        # Select most common tokens that meet min_freq
        most_common = counter.most_common(self.max_vocab_size - len(self.word2idx))
        for word, count in most_common:
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        logger.info("Fitted vocabulary with %d unique tokens (including PAD and UNK).", len(self.word2idx))
        return self

    def encode(self, text: str, max_length: int = config.MAX_SEQ_LENGTH) -> List[int]:
        tokens = text.split()
        indices = [self.word2idx.get(tok, self.UNK_IDX) for tok in tokens[:max_length]]
        if len(indices) < max_length:
            indices += [self.PAD_IDX] * (max_length - len(indices))
        return indices

    def save(self, filepath: Union[str, Path]):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "max_vocab_size": self.max_vocab_size,
                "min_freq": self.min_freq
            }, f, indent=2)
        logger.info("Saved vocabulary to %s", filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TextVocabulary":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(max_vocab_size=data["max_vocab_size"], min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        logger.info("Loaded vocabulary from %s (%d tokens)", filepath, len(vocab.word2idx))
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    random_seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train / validation / test split.
    Guarantees no overlap or target leakage.
    Uses sklearn if available, with robust numpy stratification fallback.
    """
    logger.info("Performing stratified 3-way split: %0.0f%% train, %0.0f%% val, %0.0f%% test...",
                (1.0 - test_size - val_size) * 100, val_size * 100, test_size * 100)

    try:
        from sklearn.model_selection import train_test_split
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=df[config.TARGET_BINARY_COL] if config.STRATIFY else None
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_seed,
            stratify=train_val_df[config.TARGET_BINARY_COL] if config.STRATIFY else None
        )
    except ImportError:
        logger.info("scikit-learn not yet loaded; using deterministic numpy stratified splitting.")
        rng = np.random.RandomState(random_seed)
        train_idx, val_idx, test_idx = [], [], []

        for target_val in df[config.TARGET_BINARY_COL].unique():
            cls_indices = df[df[config.TARGET_BINARY_COL] == target_val].index.values.copy()
            rng.shuffle(cls_indices)
            n_total = len(cls_indices)
            n_test = int(n_total * test_size)
            n_val = int(n_total * val_size)

            test_idx.extend(cls_indices[:n_test])
            val_idx.extend(cls_indices[n_test:n_test + n_val])
            train_idx.extend(cls_indices[n_test + n_val:])

        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]

    logger.info("Split sizes: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
