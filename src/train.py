"""
Model training engine with early stopping, checkpointing, and evaluation tracking.
Adheres strictly to the paper's learning rate, optimizer, batch size, and patience specs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

from src import config
from src.model import MarketSignalCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class JobPostingsDataset(Dataset):
    """
    PyTorch Dataset yielding tokenized job description input sequences
    and their binary market-signal target labels.
    """

    def __init__(self, encoded_texts: List[List[int]], targets: List[int]):
        self.encoded_texts = torch.tensor(encoded_texts, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_texts[idx], self.targets[idx]


def get_device() -> torch.device:
    """Detects and returns torch device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Running on CPU.")
    return device


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(dataloader.dataset)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluates model on validation or test dataloader.
    Returns (loss, accuracy, roc_auc, all_predictions, all_targets).
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    binary_preds = (all_preds_arr >= 0.5).astype(int)

    acc = accuracy_score(all_targets_arr, binary_preds)
    try:
        auc = roc_auc_score(all_targets_arr, all_preds_arr)
    except Exception:
        auc = 0.5

    return avg_loss, acc, auc, all_preds_arr, all_targets_arr


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float = config.LEARNING_RATE,
    max_epochs: int = config.MAX_EPOCHS,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    model_save_path: Optional[Path] = None
) -> Dict[str, list]:
    """
    Full training loop with Adam optimizer, BCE loss, and early stopping.
    Saves best checkpoint based on validation loss.
    """
    save_path = model_save_path or (config.MODELS_DIR / "best_market_signal_cnn.pt")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Commencing training: max_epochs=%d, lr=%s, patience=%d", max_epochs, learning_rate, patience)

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_auc"].append(val_auc)

        logger.info(
            "Epoch [%02d/%02d] - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val AUC: %.4f",
            epoch, max_epochs, train_loss, val_loss, val_acc, val_auc
        )

        # Early Stopping and Model Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "config": {
                    "vocab_size": model.vocab_size,
                    "embedding_dim": model.embedding_dim,
                    "filter_sizes": model.filter_sizes,
                    "num_filters": model.num_filters,
                    "dense_units": model.dense_units
                }
            }, save_path)
            logger.info("--> Checkpoint saved at %s (val_loss improved to %.4f)", save_path, best_val_loss)
        else:
            patience_counter += 1
            logger.info("--> No improvement for %d / %d epochs", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # Save training history
    history_path = config.RESULTS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved training history to %s", history_path)

    return history
