# LinkedIn Job Postings Dataset (2023–2024)

## Overview
This directory contains dataset documentation and metadata for the LinkedIn Job Postings dataset used in this research project.

- **Primary File:** `postings.csv` (approx. 516 MB raw CSV)
- **Source:** Kaggle — *LinkedIn Job Postings (2023 - 2024)* by arshkon
- **Kaggle Dataset URL:** [https://www.kaggle.com/datasets/arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

## Download Instructions
1. Download `postings.csv` from Kaggle:
   ```bash
   kaggle datasets download -d arshkon/linkedin-job-postings -f postings.csv
   ```
   Or manually download and unzip `postings.csv` from the Kaggle web interface.
2. Place `postings.csv` directly inside this `Dataset/` directory:
   ```text
   LinkedIn-Job-Market-Prediction/
   └── Dataset/
       ├── README.md
       └── postings.csv  (ignored by git)
   ```

## Schema & Column Information
The dataset captures detailed information across job postings:
- `job_id`: Unique identifier for the job posting.
- `company_name`: Employer name.
- `title`: Job position title.
- `description`: Full job description text (used for NLP & 1D-CNN market signal extraction).
- `max_salary`, `med_salary`, `min_salary`: Numerical compensation metrics.
- `pay_period`: Frequency of compensation (HOURLY, MONTHLY, YEARLY).
- `formatted_work_type`: Full-time, Part-time, Contract, etc.
- `formatted_experience_level`: Target proxy column (`Entry level` mapped to `1`, other valid levels mapped to `0`).
- `location`, `zip_code`, `fips`: Geographic location metadata.
- `applies`, `views`: Candidate engagement metrics.
- `listed_time`, `original_listed_time`, `expiry`, `closed_time`: Millisecond epoch timestamps.
- `skills_desc`: Textual description of required skills.
- `posting_domain`: Domain hosting the application URL.
- `sponsored`: Binary indicator of sponsored job posting.
- `work_type`, `currency`, `compensation_type`, `normalized_salary`: Standardized salary and contract fields.

## Data Governance & Git Policy
As specified by standard research engineering practices and the `.gitignore` policy, `postings.csv` is excluded from git version control due to its file size (~516 MB).
