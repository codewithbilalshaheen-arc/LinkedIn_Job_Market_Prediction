"""
Proposed Research Extension:
Temporal Job Market Demand & Emerging Skills Dynamics (2023–2024).

Analyzes longitudinal job-posting volume, technical skill growth/decline velocity,
and shifts in entry-level vs senior experience ratios across time windows.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standardized Technical Competency Lexicon
SKILL_TAXONOMY = {
    # AI / Machine Learning
    "Python": r"\bpython\b",
    "PyTorch": r"\bpytorch\b",
    "TensorFlow": r"\btensorflow\b",
    "Large Language Models (LLM)": r"\b(llm|llms|large language models?)\b",
    "Generative AI": r"\b(generative ai|genai)\b",
    "NLP": r"\b(nlp|natural language processing)\b",
    "Computer Vision": r"\b(computer vision|opencv)\b",
    # Cloud & DevOps
    "AWS": r"\b(aws|amazon web services)\b",
    "Azure": r"\bazure\b",
    "GCP": r"\b(gcp|google cloud)\b",
    "Docker": r"\bdocker\b",
    "Kubernetes": r"\b(kubernetes|k8s)\b",
    "Terraform": r"\bterraform\b",
    "CI/CD": r"\b(ci/cd|continuous integration)\b",
    "Linux": r"\blinux\b",
    # Modern Languages & Web
    "TypeScript": r"\btypescript\b",
    "JavaScript": r"\bjavascript\b",
    "React": r"\breact(\.js)?\b",
    "Node.js": r"\bnode(\.js)?\b",
    "Go / Golang": r"\b(golang|go programming)\b",
    "Rust": r"\brust\b",
    "Java": r"\bjava\b",
    "C++": r"\bc\+\+\b",
    # Data Engineering & Databases
    "SQL": r"\bsql\b",
    "PostgreSQL": r"\b(postgres|postgresql)\b",
    "MongoDB": r"\bmongodb\b",
    "Kafka": r"\bkafka\b",
    "Spark": r"\b(spark|pyspark)\b",
    "Redis": r"\bredis\b"
}


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses epoch listed_time to UTC DateTime and extracts
    year_month, week, and day representations.
    """
    df_out = df.copy()
    if "listed_time" not in df_out.columns:
        logger.warning("'listed_time' column missing from dataframe.")
        return df_out

    # Millisecond epoch conversion
    valid_ts = pd.to_numeric(df_out["listed_time"], errors="coerce")
    valid_mask = valid_ts > 1e11
    valid_ts.loc[valid_mask] = valid_ts.loc[valid_mask] / 1000.0

    df_out["datetime_utc"] = pd.to_datetime(valid_ts, unit="s", errors="coerce", utc=True)
    df_out = df_out.dropna(subset=["datetime_utc"]).copy()

    df_out["date"] = df_out["datetime_utc"].dt.date
    df_out["year_month"] = df_out["datetime_utc"].dt.to_period("M").astype(str)
    df_out["year_week"] = df_out["datetime_utc"].dt.strftime("%Y-W%U")

    logger.info("Successfully converted %d valid timestamps (Range: %s to %s).",
                len(df_out), df_out["date"].min(), df_out["date"].max())
    return df_out


def compute_temporal_posting_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes monthly aggregate postings and entry-level ratio dynamics.
    """
    if "datetime_utc" not in df.columns:
        df = parse_timestamps(df)

    grouped = df.groupby("year_month").agg(
        total_postings=("job_id", "count"),
        entry_level_count=("formatted_experience_level", lambda s: (s == "Entry level").sum()),
        senior_level_count=("formatted_experience_level", lambda s: (s == "Mid-Senior level").sum())
    ).reset_index()

    grouped["entry_level_ratio"] = grouped["entry_level_count"] / grouped["total_postings"]
    grouped["senior_level_ratio"] = grouped["senior_level_count"] / grouped["total_postings"]

    logger.info("Monthly posting trends computed:\n%s", grouped.to_string(index=False))
    return grouped


def analyze_emerging_skills(
    df: pd.DataFrame,
    min_postings_threshold: int = 15
) -> pd.DataFrame:
    """
    Measures skill demand trajectory by comparing early observation window (Period 1)
    with later observation window (Period 2) across the job description corpus.
    
    Computes:
      - Period 1 Frequency & Share
      - Period 2 Frequency & Share
      - Absolute Difference
      - Growth Velocity (% Change)
    """
    if "datetime_utc" not in df.columns:
        df = parse_timestamps(df)

    # Sort temporally
    df_sorted = df.sort_values(by="datetime_utc").reset_index(drop=True)
    midpoint_idx = len(df_sorted) // 2

    period1_df = df_sorted.iloc[:midpoint_idx]
    period2_df = df_sorted.iloc[midpoint_idx:]

    p1_total = len(period1_df)
    p2_total = len(period2_df)
    logger.info("Comparing Period 1 (N=%d) vs Period 2 (N=%d) for skill growth analysis...", p1_total, p2_total)

    skill_rows = []

    # Compile regexes for fast vector matching
    for skill_name, pattern in SKILL_TAXONOMY.items():
        rx = re.compile(pattern, re.IGNORECASE)

        # Count occurrences in period 1
        p1_matches = period1_df["description"].dropna().astype(str).str.contains(rx).sum()
        # Count occurrences in period 2
        p2_matches = period2_df["description"].dropna().astype(str).str.contains(rx).sum()

        p1_rate = (p1_matches / p1_total) * 100
        p2_rate = (p2_matches / p2_total) * 100

        # Percentage growth rate in relative share
        if p1_rate > 0:
            growth_rate = ((p2_rate - p1_rate) / p1_rate) * 100
        else:
            growth_rate = 100.0 if p2_rate > 0 else 0.0

        if (p1_matches + p2_matches) >= min_postings_threshold:
            skill_rows.append({
                "skill": skill_name,
                "p1_count": int(p1_matches),
                "p1_rate_pct": round(p1_rate, 2),
                "p2_count": int(p2_matches),
                "p2_rate_pct": round(p2_rate, 2),
                "growth_rate": round(growth_rate, 2),
                "status": "Emerging / High Growth" if growth_rate > 10.0 else ("Declining / Saturated" if growth_rate < -10.0 else "Stable")
            })

    results_df = pd.DataFrame(skill_rows).sort_values(by="growth_rate", ascending=False).reset_index(drop=True)

    # Save to CSV
    csv_out = config.RESULTS_DIR / "emerging_skills_analysis.csv"
    results_df.to_csv(csv_out, index=False)
    logger.info("Saved emerging skills analysis to %s", csv_out)

    return results_df
