"""
One-time script: assign train/val/test split to scan CSV.

Adds a 'split' column (values: train / val / test) using patient-level
stratified split (70 / 10 / 20), seed=42.

Usage:
    python make_splits.py --scan_csv <path_to_scan_csv> [--seed 42]

The CSV is updated in-place.
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(scan_csv_path: str, train_ratio: float = 0.7,
                val_ratio: float = 0.1, test_ratio: float = 0.2,
                seed: int = 42) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df = pd.read_csv(scan_csv_path)

    if "split" in df.columns:
        print("WARNING: 'split' column already exists. Overwriting.")

    # --- Patient-level grouping ---
    # Majority label per patient
    patient_ids = df["patient_id"].unique().tolist()
    patient_label = {}
    for pid in patient_ids:
        sub = df[df["patient_id"] == pid]["label"]
        patient_label[pid] = sub.value_counts().idxmax()

    pid_list = list(patient_label.keys())
    lbl_list = [patient_label[p] for p in pid_list]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_pids, valtest_pids, _, valtest_lbls = train_test_split(
        pid_list, lbl_list,
        test_size=val_test_ratio, stratify=lbl_list, random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    val_pids, test_pids, _, _ = train_test_split(
        valtest_pids, valtest_lbls,
        test_size=relative_test, stratify=valtest_lbls, random_state=seed,
    )

    pid_to_split = {}
    for p in train_pids:
        pid_to_split[p] = "train"
    for p in val_pids:
        pid_to_split[p] = "val"
    for p in test_pids:
        pid_to_split[p] = "test"

    df["split"] = df["patient_id"].map(pid_to_split)

    # Sanity check
    missing = df["split"].isna().sum()
    if missing > 0:
        print(f"WARNING: {missing} rows have no split assigned.")

    counts = df.groupby("split")["patient_id"].nunique()
    scan_counts = df["split"].value_counts()
    print(f"Split assigned (patients): {dict(counts)}")
    print(f"Split assigned (scans):    {dict(scan_counts)}")
    print(f"Ratios (scans): "
          f"train={scan_counts.get('train',0)/len(df):.3f}, "
          f"val={scan_counts.get('val',0)/len(df):.3f}, "
          f"test={scan_counts.get('test',0)/len(df):.3f}")

    df.to_csv(scan_csv_path, index=False)
    print(f"\nSaved: {scan_csv_path}")


if __name__ == "__main__":
    import yaml, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_csv", default=None, help="Path to scan CSV (default: read from divnet_config.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scan_csv = args.scan_csv
    if scan_csv is None:
        config_path = os.path.join(os.path.dirname(__file__), "divnet_config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        scan_csv = cfg["data"]["scan_csv"]
        print(f"scan_csv from config: {scan_csv}")

    make_splits(scan_csv, seed=args.seed)
