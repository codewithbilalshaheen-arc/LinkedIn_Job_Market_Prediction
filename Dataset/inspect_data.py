"""
Standalone exploratory data inspection using Python standard library.
Computes exact dimensions, column null counts, target category frequencies,
and timestamp ranges directly from Dataset/postings.csv.
"""

import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

csv_path = "Dataset/postings.csv"
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} does not exist!")
    sys.exit(1)

print(f"Inspecting {csv_path} (file size: {os.path.getsize(csv_path) / 1e6:.2f} MB)...")

col_null_counts = Counter()
total_rows = 0
header = []
experience_levels = Counter()
work_types = Counter()
pay_periods = Counter()
listed_times = []
closed_times = []

with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        print("Empty file!")
        sys.exit(1)

    exp_idx = header.index("formatted_experience_level") if "formatted_experience_level" in header else -1
    work_idx = header.index("formatted_work_type") if "formatted_work_type" in header else -1
    pay_idx = header.index("pay_period") if "pay_period" in header else -1
    desc_idx = header.index("description") if "description" in header else -1
    time_idx = header.index("listed_time") if "listed_time" in header else -1
    closed_idx = header.index("closed_time") if "closed_time" in header else -1

    for row in reader:
        total_rows += 1
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))

        for i, val in enumerate(row[:len(header)]):
            if val is None or val.strip() == "":
                col_null_counts[header[i]] += 1

        if exp_idx >= 0:
            exp_val = row[exp_idx].strip()
            experience_levels[exp_val if exp_val else "(Missing/Null)"] += 1

        if work_idx >= 0:
            w_val = row[work_idx].strip()
            work_types[w_val if w_val else "(Missing/Null)"] += 1

        if pay_idx >= 0:
            p_val = row[pay_idx].strip()
            pay_periods[p_val if p_val else "(Missing/Null)"] += 1

        if time_idx >= 0 and row[time_idx].strip():
            try:
                t_val = float(row[time_idx].strip())
                # Timestamps are often millisecond epoch
                if t_val > 1e11:
                    t_val /= 1000.0
                listed_times.append(t_val)
            except ValueError:
                pass

print(f"\n--- DATASET SUMMARY ---")
print(f"Total Rows: {total_rows:,}")
print(f"Total Columns: {len(header)}")
print(f"Columns: {header}")

print(f"\n--- FORMATTED EXPERIENCE LEVEL DISTRIBUTION ---")
valid_levels = ["Entry level", "Mid-Senior level", "Associate", "Director", "Executive", "Internship"]
valid_total = sum(count for k, count in experience_levels.items() if k in valid_levels)
entry_count = experience_levels.get("Entry level", 0)
other_count = valid_total - entry_count

for k, count in experience_levels.most_common():
    pct = (count / total_rows) * 100
    print(f"  {k:30s}: {count:8,d} ({pct:5.2f}%)")

print(f"\nTarget Variable Statistics (Filtered on Valid Levels):")
print(f"  Total Valid Rows: {valid_total:,} ({(valid_total / total_rows)*100:.2f}% of all postings)")
print(f"  Class 1 (Entry level / Low Priority Proxy): {entry_count:,} ({(entry_count / valid_total)*100:.2f}%)")
print(f"  Class 0 (Other Levels / High Priority Proxy): {other_count:,} ({(other_count / valid_total)*100:.2f}%)")

print(f"\n--- TOP WORK TYPES ---")
for k, count in work_types.most_common(10):
    print(f"  {k:30s}: {count:8,d}")

print(f"\n--- TOP MISSING COLUMNS ---")
for col, missing_count in col_null_counts.most_common(15):
    pct = (missing_count / total_rows) * 100
    print(f"  {col:30s}: {missing_count:8,d} ({pct:5.2f}% missing)")

if listed_times:
    min_date = datetime.fromtimestamp(min(listed_times), tz=timezone.utc)
    max_date = datetime.fromtimestamp(max(listed_times), tz=timezone.utc)
    print(f"\n--- TEMPORAL RANGE ---")
    print(f"  Min Listed Time: {min_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Max Listed Time: {max_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Save inspection summary to JSON
summary_data = {
    "total_rows": total_rows,
    "total_columns": len(header),
    "columns": header,
    "experience_levels": dict(experience_levels),
    "valid_total": valid_total,
    "entry_count": entry_count,
    "other_count": other_count,
    "missing_counts": dict(col_null_counts),
    "min_date": min_date.strftime('%Y-%m-%d') if listed_times else None,
    "max_date": max_date.strftime('%Y-%m-%d') if listed_times else None
}

with open("Results/dataset_inspection_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2)

print("\nSaved summary to Results/dataset_inspection_summary.json")
