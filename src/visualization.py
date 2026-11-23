"""
Visualization module for publication-quality figures:
EDA distributions, missing value maps, training curves, ROC/PR curves,
confusion matrices, and temporal trend analyses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

# Publication style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.dpi": 300
})


def plot_target_distribution(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Visualizes the class distribution of formatted_experience_level and binary target."""
    save_path = save_path or (config.FIGURES_DIR / "target_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Experience level counts
    exp_counts = df[config.TARGET_COLUMN].value_counts().dropna()
    sns.barplot(x=exp_counts.values, y=exp_counts.index, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Distribution of Raw Experience Levels")
    axes[0].set_xlabel("Number of Job Postings")
    axes[0].set_ylabel("Formatted Experience Level")
    for i, v in enumerate(exp_counts.values):
        axes[0].text(v + (max(exp_counts.values) * 0.01), i, f"{v:,} ({v/len(df)*100:.1f}%)", va="center")

    # Binary target distribution
    if config.TARGET_BINARY_COL in df.columns:
        target_counts = df[config.TARGET_BINARY_COL].value_counts()
        labels = ["High Priority / Other (0)", "Low Priority / Entry (1)"]
        colors = ["#2b5c8f", "#d95f02"]
        axes[1].pie(
            [target_counts.get(0, 0), target_counts.get(1, 0)],
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            explode=(0, 0.05)
        )
        axes[1].set_title("Binary Market-Signal Target Ratio")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_missing_values(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots percentage of missing values across all columns."""
    save_path = save_path or (config.FIGURES_DIR / "missing_values.png")

    missing = (df.isnull().sum() / len(df)) * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=missing.values, y=missing.index, palette="mako")
    plt.title("Missing Value Analysis Across Job Postings")
    plt.xlabel("Percentage Missing (%)")
    plt.ylabel("Dataset Columns")
    plt.xlim(0, 105)

    for i, v in enumerate(missing.values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_salary_and_work_type(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots salary distribution and work-type breakdowns."""
    save_path = save_path or (config.FIGURES_DIR / "salary_and_work_type.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Work type distribution
    if "formatted_work_type" in df.columns:
        work_counts = df["formatted_work_type"].value_counts().head(6)
        sns.barplot(x=work_counts.values, y=work_counts.index, ax=axes[0], palette="viridis")
        axes[0].set_title("Job Postings by Work Type")
        axes[0].set_xlabel("Count")
        axes[0].set_ylabel("Work Type")

    # Salary distribution (normalized or min/max)
    salary_col = "normalized_salary" if "normalized_salary" in df.columns else "min_salary"
    if salary_col in df.columns:
        valid_salaries = df[salary_col].dropna()
        valid_salaries = valid_salaries[(valid_salaries > 10000) & (valid_salaries < 350000)]
        if len(valid_salaries) > 0:
            sns.histplot(valid_salaries, kde=True, ax=axes[1], color="#2b5c8f", bins=30)
            axes[1].set_title("Annualized Salary Distribution ($ USD)")
            axes[1].set_xlabel("Salary ($)")
            axes[1].set_ylabel("Density / Count")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: Dict[str, list], save_path: Optional[Path] = None):
    """Plots training and validation loss, accuracy, and ROC-AUC over epochs."""
    save_path = save_path or (config.FIGURES_DIR / "training_curves.png")

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["val_accuracy"], "^-", label="Val Accuracy", color="#2ca02c")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # ROC-AUC
    axes[2].plot(epochs, history["val_auc"], "d-", label="Val ROC-AUC", color="#9467bd")
    axes[2].set_title("Validation ROC-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[Path] = None):
    """Plots formatted confusion matrix with raw counts and percentages."""
    save_path = save_path or (config.FIGURES_DIR / "confusion_matrix.png")

    plt.figure(figsize=(7, 6))
    labels = ["High Priority (0)", "Low / Entry (1)"]
    group_counts = [f"{v:,}" for v in cm.flatten()]
    group_percentages = [f"{v:.2%}" for v in cm.flatten() / np.sum(cm)]
    annot_labels = [f"{c}\n({p})" for c, p in zip(group_counts, group_percentages)]
    annot_labels = np.asarray(annot_labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title(f"Test Confusion Matrix (Total N = {np.sum(cm):,})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_and_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, save_path: Optional[Path] = None):
    """Plots ROC Curve and Precision-Recall Curve side-by-side."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    save_path = save_path or (config.FIGURES_DIR / "roc_pr_curves.png")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label="1D-CNN Market Signal")
    axes[0].plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random Baseline")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # PR Curve
    baseline_pr = np.mean(y_true)
    axes[1].plot(recall, precision, color="#d95f02", lw=2, label="1D-CNN Market Signal")
    axes[1].axhline(y=baseline_pr, color="grey", lw=1, linestyle="--", label=f"Baseline ({baseline_pr:.2f})")
    axes[1].set_title("Precision-Recall (PR) Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_temporal_trends(df_temporal: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots temporal job postings and experience level evolution (Research Extension)."""
    save_path = save_path or (config.FIGURES_DIR / "temporal_trends.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Monthly total job postings
    sns.lineplot(data=df_temporal, x="year_month", y="total_postings", marker="o", ax=axes[0], color="#2b5c8f", lw=2)
    axes[0].set_title("Monthly LinkedIn Job Postings Volume (2023–2024)")
    axes[0].set_ylabel("Total Postings")

    # Ratio of Entry level vs Senior postings over time
    if "entry_level_ratio" in df_temporal.columns:
        sns.lineplot(data=df_temporal, x="year_month", y="entry_level_ratio", marker="s", ax=axes[1], color="#e7298a", lw=2)
        axes[1].set_title("Entry-Level Demand Share Over Time (Market Proxy Shift)")
        axes[1].set_ylabel("Entry Level Ratio")
        axes[1].set_xlabel("Year-Month")
        axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_emerging_skills(skills_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots top emerging vs declining technical skills across 2023–2024."""
    save_path = save_path or (config.FIGURES_DIR / "emerging_skills.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_growing = skills_df.sort_values(by="growth_rate", ascending=False).head(10)
    top_declining = skills_df.sort_values(by="growth_rate", ascending=True).head(10)

    # Growing skills
    sns.barplot(data=top_growing, x="growth_rate", y="skill", ax=axes[0], palette="Greens_r")
    axes[0].set_title("Top Emerging / High-Growth Skills (%)")
    axes[0].set_xlabel("Demand Growth Rate (%)")
    axes[0].set_ylabel("Skill / Competency")

    # Declining skills
    sns.barplot(data=top_declining, x="growth_rate", y="skill", ax=axes[1], palette="Reds_r")
    axes[1].set_title("Top Saturated / Declining Skills (%)")
    axes[1].set_xlabel("Demand Growth Rate (%)")
    axes[1].set_ylabel("Skill / Competency")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
