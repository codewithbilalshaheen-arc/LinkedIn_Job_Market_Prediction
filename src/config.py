"""
Configuration module for LinkedIn Job Market Prediction & Feature Prioritization.
Encapsulates all project paths, hyperparameters, target mapping rules,
and reproducibility seeds.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset"
RAW_DATA_PATH = DATA_DIR / "postings.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "Results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
EVALUATION_METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"
RESEARCH_PAPER_DIR = PROJECT_ROOT / "Research_Paper"

# Ensure runtime directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target Variable Configuration (Research Proxy)
# In the research framework, market signals serve as proxies for startup feature prioritization.
# Class 1: "Entry level" -> Low market-priority proxy (lower baseline market seniority demand)
# Class 0: Other valid experience levels -> High market-priority proxy (higher market seniority demand)
TARGET_COLUMN = "formatted_experience_level"
TARGET_BINARY_COL = "target"
ENTRY_LEVEL_STRING = "Entry level"

VALID_EXPERIENCE_LEVELS = [
    "Entry level",
    "Mid-Senior level",
    "Associate",
    "Director",
    "Executive",
    "Internship"
]

# Random Seed for Complete Reproducibility
RANDOM_SEED = 42

# Train / Validation / Test Splitting Configuration
TEST_SIZE = 0.20
VAL_SIZE = 0.15  # 15% of total dataset, resulting in 65% Train, 15% Val, 20% Test
STRATIFY = True

# Text Processing Hyperparameters
MAX_VOCAB_SIZE = 25000
MAX_SEQ_LENGTH = 250
EMBEDDING_DIM = 100
MIN_WORD_FREQ = 2

# Market Signal 1D-CNN Architecture Hyperparameters (Paper Specification)
CNN_FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100  # 100 filters per size -> 300 total pooled features
DENSE_UNITS = 128
DROPOUT_RATE = 0.5

# Training Hyperparameters (Paper Specification)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
WEIGHT_DECAY = 1e-4

# Published Inconsistency Note
# Paper claims N_test = 24,823, but reported confusion matrix values sum to 9,930:
# TP + FP + FN + TN = 9,930 (a factor of 2.5x mismatch). Documented for scientific integrity.
PUBLISHED_CONFUSION_MATRIX_SUM = 9930
PUBLISHED_REPORTED_TEST_N = 24823
