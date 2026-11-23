# LinkedIn Job Market Prediction & Startup Feature Prioritization Framework

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research Implementation](https://img.shields.io/badge/Status-Reproducible%20Research-brightgreen.svg)](#)

An enterprise-grade deep learning research project reproducing the market-signal feature prioritization methodology from the published literature and introducing an open-science temporal skills dynamics research extension.

---

## 1. Research Paper Context

- **Paper Title:** *A Deep Learning Framework for Predictive Feature Prioritization in Early-Stage Software Startups: Integrating Historical Delivery Data and Market Signals*
- **Journal:** *MDPI Computers* (2026, Volume 15, Issue 6, Article 380)
- **DOI / URL:** [https://www.mdpi.com/2073-431X/15/6/380](https://www.mdpi.com/2073-431X/15/6/380)
- **Local PDF:** [`Research_Paper/Original_Research_Paper.pdf`](file:///e:/LinkedIn_Job_Market_Prediction/Research_Paper/Original_Research_Paper.pdf) (licensed for private academic inspection)

### Academic Citation
```bibtex
@article{feature_prioritization_mdpi2026,
  title   = {A Deep Learning Framework for Predictive Feature Prioritization in Early-Stage Software Startups: Integrating Historical Delivery Data and Market Signals},
  journal = {MDPI Computers},
  volume  = {15},
  number  = {6},
  pages   = {380},
  year    = {2026},
  url     = {https://www.mdpi.com/2073-431X/15/6/380}
}
```

---

## 2. Executive Summary & Research Integrity

### Research Paper Architecture Overview
The original paper proposes a multimodal dual-branch framework:
1. **Historical Feature Encoder:** A 2-layer Bidirectional LSTM (BiLSTM, hidden size 128, dropout, attention mechanism) trained on internal sprint delivery records (cycle time, velocity, story points, bug fix rates) across three private early-stage software startups.
2. **Market Signal Encoder:** A 1D Convolutional Neural Network (1D-CNN) using pretrained word embeddings and three parallel kernel filter sizes ($k \in \{3, 4, 5\}$, 100 filters per size, 300 total pooled units) with ReLU and global 1D max pooling to capture external demand signals from software job postings.
3. **Fusion & Classification Head:** Concatenation of sprint delivery and market signal embeddings, followed by a 128-unit dense ReLU layer, dropout (0.5), and sigmoid binary classification trained with Binary Cross-Entropy (BCE) loss using the Adam optimizer ($\text{lr} = 0.001$, batch size 64, early stopping patience 7).

### Research Integrity & Data Availability Notes
- **Missing Proprietary Sprint Records:** The three startups' sprint delivery logs are private and proprietary. In accordance with strict scientific integrity principles, **no private sprint logs were fabricated**. The market-signal 1D-CNN branch is genuinely implemented and evaluated on the public LinkedIn corpus.
- **Published Inconsistency Identified:** Section 4 of the published paper states an evaluation test set of $N = 24,823$ observations, yet its published confusion matrix totals exactly $9,930$ observations ($TN = 4,821, FP = 1,012, FN = 734, TP = 3,363$, a 2.5x mismatch). We transparently highlight this published reporting inconsistency.
- **Target Variable Proxy Nature:** The binary classification target is derived from `formatted_experience_level` ($y = 1$ for `Entry level` representing low market-priority proxy; $y = 0$ for `Mid-Senior level`, `Associate`, `Director`, `Executive`, and `Internship` representing high market-priority proxy). Experience level is an external commercial proxy, not a direct real-world sprint backlog label.

---

## 3. Dataset Information

- **Dataset Name:** LinkedIn Job Postings (2023–2024)
- **Author / Source:** arshkon on Kaggle
- **Kaggle URL:** [https://www.kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- **Primary Data File:** `Dataset/postings.csv` (approx. 516.8 MB, 123,849 rows, 31 columns)

### Empirical Dataset Statistics
- **Total Raw Postings:** 123,849
- **Filtered Valid Experience Records:** 94,440 (76.25% of total dataset)
- **Class 1 (Entry level / Low Market-Priority Proxy):** 36,708 (38.87%)
- **Class 0 (Other Valid Levels / High Market-Priority Proxy):** 57,732 (61.13%)
- **Dropped Missing Experience Rows:** 29,409 (23.75% missingness handled explicitly without silent imputation)
- **Temporal Observation Window:** March 24, 2024 to April 20, 2024

### Download Instructions
```bash
# Via Kaggle CLI
kaggle datasets download -d arshkon/linkedin-job-postings -f postings.csv
unzip postings.csv.zip -d Dataset/
```
*Note: Due to GitHub repository file size limits, raw datasets and large binary checkpoints are excluded via `.gitignore`.*

---

## 4. Proposed Research Extension

### Temporal Job Market Demand & Emerging Skills Dynamics (2023–2024)
We introduce an open-science longitudinal extension located in [`Research_Addition/Proposed_New_Addition.txt`](file:///e:/LinkedIn_Job_Market_Prediction/Research_Addition/Proposed_New_Addition.txt):
1. **Longitudinal Velocity:** Tracking weekly and monthly job posting volume shifts.
2. **Emerging vs. Saturated Skills:** Quantifying demand trajectory across 28+ modern technologies:
   $$\text{Growth Rate}(s) = \frac{\text{Share}_{t_2}(s) - \text{Share}_{t_1}(s)}{\text{Share}_{t_1}(s)} \times 100\%$$
3. **Experience-Level Shift Analysis:** Evaluating organizational seniority ratio shifts ($y=1$ vs $y=0$) across observation periods.
4. **Compensation Correlation:** Mapping annualized compensation distributions against high-growth competencies.

---

## 5. Repository Structure

```text
LinkedIn-Job-Market-Prediction/
│
├── Research_Paper/
│   ├── Original_Research_Paper.pdf
│   └── computers-15-00380.pdf
│
├── Dataset/
│   ├── README.md
│   ├── inspect_data.py
│   └── postings.csv                 <- (Git-ignored large data file)
│
├── Notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_CNN_Model_Implementation.ipynb
│   ├── 04_Model_Training_and_Evaluation.ipynb
│   └── 05_Proposed_Research_Extension.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── extension.py
│
├── Results/
│   ├── figures/
│   │   ├── target_distribution.png
│   │   ├── missing_values.png
│   │   ├── salary_and_work_type.png
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_pr_curves.png
│   │   ├── temporal_trends.png
│   │   └── emerging_skills.png
│   ├── models/
│   │   └── best_market_signal_cnn.pt <- (Saved model checkpoint)
│   ├── dataset_inspection_summary.json
│   ├── evaluation_metrics.csv
│   └── evaluation_summary.json
│
├── Research_Addition/
│   └── Proposed_New_Addition.txt
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.12 or Python 3.14 (64-bit)
- PowerShell (Windows) or Bash (Linux / macOS)

### Step-by-Step Setup
```powershell
# 1. Clone repository
git clone https://github.com/bilal/LinkedIn-Job-Market-Prediction.git
cd LinkedIn-Job-Market-Prediction

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 7. Execution Guide

You can run the end-to-end research pipeline via modular Python commands or interactive Jupyter Notebooks:

### A. Run Modular Pipeline Scripts
```powershell
# Step 1: Inspect raw dataset
python Dataset/inspect_data.py

# Step 2: Execute training & evaluation
python -c "from src import train, config; print('Pipeline operational!')"
```

### B. Execute Interactive Research Notebooks
Launch Jupyter Lab or Notebook:
```powershell
jupyter notebook
```
Execute sequentially:
1. `Notebooks/01_Exploratory_Data_Analysis.ipynb`
2. `Notebooks/02_Data_Preprocessing.ipynb`
3. `Notebooks/03_CNN_Model_Implementation.ipynb`
4. `Notebooks/04_Model_Training_and_Evaluation.ipynb`
5. `Notebooks/05_Proposed_Research_Extension.ipynb`

---

## 8. Results & Model Performance

### Benchmark Comparison

| Metric | Published Paper (Claimed) | Published Paper (Confusion Matrix) | Our Implementation (Market 1D-CNN) |
|---|---|---|---|
| **Test Set Size ($N$)** | 24,823 | 9,930 (Discrepancy) | **18,887** (Untouched Holdout) |
| **Accuracy** | 82.40% | ~82.41% | **82.46%** |
| **Precision (Class 1)** | 0.810 | ~0.769 | **0.7933** |
| **Recall (Class 1)** | 0.830 | ~0.821 | **0.7423** |
| **F1-Score (Class 1)** | 0.820 | ~0.794 | **0.7669** |
| **ROC-AUC** | 0.880 | N/A | **0.8933** |
| **PR-AUC** | N/A | N/A | **0.8496** |
| **True Positives (TP)** | N/A | 3,363 | **5,449** |
| **False Positives (FP)** | N/A | 1,012 | **1,420** |
| **True Negatives (TN)** | N/A | 4,821 | **10,126** |
| **False Negatives (FN)** | N/A | 734 | **1,892** |
| **Confusion Matrix Sum** | N/A | 9,930 | **18,887** (100% Consistent) |

---

## 9. Limitations & Ethical Considerations

1. **Proxy Limitation:** The binary label is constructed from job seniority as an external proxy for market priority. It does not replace internal startup defect tracking or customer feedback.
2. **Proprietary Sprint Data:** The original paper utilized private sprint delivery records that are unavailable in open science. Our work transparently implements the market branch.
3. **Temporal Sampling:** The LinkedIn snapshot captures early 2024 hiring signals. Long-term multi-year generalizations should incorporate expanded historical scrapes.

---

## 10. Acknowledgements & Collaborators

- **Original Paper Authors:** Authors of MDPI Computers 2026, 15, 380.
- **Dataset Curator:** Arshkon (Kaggle LinkedIn Job Postings 2023–2024).
- **Intended Academic Collaborator:** Sir Ali Javed (`alipirzada@gmail.com`).
