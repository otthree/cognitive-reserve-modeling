"""
Dataset and data loading utilities for DivNetFusion training.

Combines:
  - 3D MRI .pt tensors from {data_root}/3D_tensors/{CN,MCI,AD}/*.pt
  - 6 tabular features from ADNI_master_merged CSV, joined via scan CSV

Tabular features (logical name -> actual CSV column):
  FDG_BL   -> FDG_bl
  VSBPDIA  -> VSBPDIA
  VSTEM    -> VSTEMP
  VSRESP   -> VSRESP

Missing value handling:
  - VS columns (VSBPDIA, VSTEMP, VSRESP): -1 treated as NaN (sentinel value in ADNI)
  - All missing values imputed with per-diagnosis-group (CN/MCI/AD) median
  - Tabular features z-score normalized using training set statistics
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ── Constants ────────────────────────────────────────────────────────────────

CLASS_MAP = {"CN": 0, "MCI": 1, "AD": 2}

# Maps logical feature name -> actual column name in master CSV
FEATURE_COL_MAP = {
    "FDG_BL":  "FDG_bl",
    "VSBPDIA": "VSBPDIA",
    "VSTEM":   "VSTEMP",
    "VSRESP":  "VSRESP",
}

# Columns that use -1 as a sentinel "missing" value in ADNI vitals tables
VS_SENTINEL_COLS = {"VSBPDIA", "VSTEMP", "VSRESP"}

FEATURE_NAMES = list(FEATURE_COL_MAP.keys())   # canonical order
ACTUAL_COLS   = list(FEATURE_COL_MAP.values())


# ── Tabular lookup builder ────────────────────────────────────────────────────

def build_tabular_lookup(scan_csv_path: str, master_csv_path: str, report: bool = True):
    """
    Build {pt_index_str: np.float32 array of 6 features} with group-median imputation.

    Steps:
      1. Load master CSV; replace -1 sentinels with NaN in VS columns.
      2. Compute per-group (CN/MCI/AD) medians from master CSV.
      3. Join scan CSV with master CSV on image_id == 'Image Data ID'.
      4. Impute missing values using the scan label (CN/MCI/AD) group median.

    Returns:
        lookup        : dict {str(pt_index): np.ndarray shape (6,)}
        group_medians : dict {dx: {col: median_value}}
        raw_missing   : dict {feature_name: total_missing_count in master CSV}
    """
    scan_df = pd.read_csv(scan_csv_path)
    master_df = pd.read_csv(master_csv_path, low_memory=False)

    # --- Replace -1 sentinels with NaN in VS vitals columns ---
    for col in VS_SENTINEL_COLS:
        if col in master_df.columns:
            master_df[col] = master_df[col].replace(-1.0, np.nan)

    # --- Map master DX: Dementia -> AD ---
    dx_remap = {"CN": "CN", "MCI": "MCI", "Dementia": "AD"}
    master_df["DX_group"] = master_df["DX"].map(dx_remap)

    # --- Report raw missing counts (after sentinel replacement) ---
    raw_missing = {}
    if report:
        print("\n=== Missing Value Report (master CSV, after -1 sentinel removal) ===")
        print(f"  {'Variable':<10}  {'Actual Col':<10}  {'Missing':>8}  {'Total':>8}  {'%':>6}")
        print("  " + "-" * 50)
    for feat, col in FEATURE_COL_MAP.items():
        n_miss = int(master_df[col].isna().sum())
        n_total = len(master_df)
        raw_missing[feat] = n_miss
        if report:
            print(f"  {feat:<10}  {col:<10}  {n_miss:>8,}  {n_total:>8,}  {100*n_miss/n_total:>5.1f}%")
    if report:
        print()

    # --- Compute group medians from master CSV ---
    group_medians: dict[str, dict[str, float]] = {}
    for dx in ["CN", "MCI", "AD"]:
        sub = master_df[master_df["DX_group"] == dx]
        group_medians[dx] = {col: float(sub[col].median()) for col in ACTUAL_COLS}

    if report:
        print("=== Group Medians used for Imputation ===")
        print(f"  {'Variable':<10}  {'Col':<10}  {'CN':>10}  {'MCI':>10}  {'AD':>10}")
        print("  " + "-" * 58)
        for feat, col in FEATURE_COL_MAP.items():
            cn_m  = group_medians["CN"][col]
            mci_m = group_medians["MCI"][col]
            ad_m  = group_medians["AD"][col]
            print(f"  {feat:<10}  {col:<10}  {cn_m:>10.4f}  {mci_m:>10.4f}  {ad_m:>10.4f}")
        print()

    # --- Join scan CSV with master CSV on image_id ---
    master_sub = master_df[["Image Data ID", "DX_group"] + ACTUAL_COLS].copy()
    merged = scan_df.merge(
        master_sub,
        left_on="image_id",
        right_on="Image Data ID",
        how="left",
    )

    # --- Build lookup dict with imputed values ---
    lookup: dict[str, np.ndarray] = {}
    n_imputed = Counter()

    for _, row in merged.iterrows():
        pt_idx = str(row["pt_index"])
        label = row["label"]  # CN / MCI / AD  (from scan CSV)

        feats = []
        for col in ACTUAL_COLS:
            val = row[col]
            if pd.isna(val):
                val = group_medians[label][col]
                n_imputed[col] += 1
            feats.append(float(val))
        lookup[pt_idx] = np.array(feats, dtype=np.float32)

    if report:
        print("=== Imputation Applied (scan-level) ===")
        for feat, col in FEATURE_COL_MAP.items():
            print(f"  {feat} ({col}): {n_imputed[col]} scans imputed")
        print()

    return lookup, group_medians, raw_missing


def compute_tab_normalization(tab_lookup: dict, train_paths: list[str]):
    """
    Compute z-score normalization stats (mean, std) from training scans only.

    Returns:
        mean : np.ndarray shape (6,)
        std  : np.ndarray shape (6,)
    """
    feats = []
    for p in train_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in tab_lookup:
            feats.append(tab_lookup[stem])
    feats = np.stack(feats, axis=0)  # [N_train, 6]
    mean = feats.mean(axis=0)
    std  = feats.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    return mean, std


# ── Dataset ──────────────────────────────────────────────────────────────────

class ADNIFusionDataset(Dataset):
    """
    Dataset returning (volume [1,192,192,192], tab_feats [6], label) tuples.

    Args:
        paths      : list of .pt file paths
        labels     : list of integer labels (CN=0, MCI=1, AD=2)
        tab_lookup : {pt_index_str: np.float32 array (6,)}
        tab_mean   : np.ndarray (6,) for z-score normalization
        tab_std    : np.ndarray (6,) for z-score normalization
        augment    : whether to apply MRI augmentation
    """

    def __init__(
        self,
        paths: list[str],
        labels: list[int],
        tab_lookup: dict[str, np.ndarray],
        tab_mean: np.ndarray,
        tab_std: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.01,
        intensity_shift: float = 0.1,
        crop_size: int = 176,
        input_size: int = 192,
    ):
        self.paths = paths
        self.labels = labels
        self.tab_lookup = tab_lookup
        self.tab_mean = tab_mean
        self.tab_std = tab_std
        self.augment = augment
        self.noise_std = noise_std
        self.intensity_shift = intensity_shift
        self.crop_size = crop_size
        self.input_size = input_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        # ---- MRI volume ----
        volume = torch.load(path, map_location="cpu", weights_only=True)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)

        # Per-volume min-max normalization to [0, 1]
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 1e-8:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = torch.zeros_like(volume)

        if self.augment:
            max_offset = self.input_size - self.crop_size
            ox = random.randint(0, max_offset)
            oy = random.randint(0, max_offset)
            oz = random.randint(0, max_offset)
            cropped = volume[
                :, ox:ox+self.crop_size, oy:oy+self.crop_size, oz:oz+self.crop_size
            ]
            volume = torch.zeros(1, self.input_size, self.input_size, self.input_size)
            volume[
                :, ox:ox+self.crop_size, oy:oy+self.crop_size, oz:oz+self.crop_size
            ] = cropped

            if random.random() > 0.5:
                scale = random.uniform(1.05, 1.2)
                scaled_size = int(self.input_size * scale)
                volume = torch.nn.functional.interpolate(
                    volume.unsqueeze(0), size=scaled_size,
                    mode="trilinear", align_corners=False,
                ).squeeze(0)
                start = (scaled_size - self.input_size) // 2
                volume = volume[
                    :,
                    start:start+self.input_size,
                    start:start+self.input_size,
                    start:start+self.input_size,
                ]

            if random.random() > 0.5:
                volume = torch.flip(volume, dims=[1])

            volume = volume + torch.randn_like(volume) * self.noise_std
            volume = volume + random.uniform(-self.intensity_shift, self.intensity_shift)
            volume = volume.clamp(0.0, 1.0)

        # ---- Tabular features (z-score normalized) ----
        stem = os.path.splitext(os.path.basename(path))[0]
        raw_tab = self.tab_lookup[stem]  # np.float32 (6,)
        tab = (raw_tab - self.tab_mean) / self.tab_std
        tab = torch.from_numpy(tab)

        return volume, tab, label


# ── Helpers shared with original divnet_dataset.py ───────────────────────────

def collect_file_paths(data_root: str, exclude_indices=None):
    tensor_dir = os.path.join(data_root, "3D_tensors")
    paths, labels = [], []
    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(tensor_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: directory not found: {class_dir}")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".pt"):
                if exclude_indices is not None:
                    stem = os.path.splitext(fname)[0]
                    if stem in exclude_indices:
                        continue
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    print(f"Found {len(paths)} total samples: {dict(Counter(labels))}")
    return paths, labels


def build_exclude_set(scan_csv_path: str, exclude_csv_path: str):
    scan_df = pd.read_csv(scan_csv_path)
    exclude_df = pd.read_csv(exclude_csv_path)
    exclude_iids = set(exclude_df["image_id"].astype(str).unique())
    mask = scan_df["image_id"].astype(str).isin(exclude_iids)
    exclude_indices = set(scan_df.loc[mask, "pt_index"].astype(str).values)
    print(f"CR exclusion: {len(exclude_indices)} scans excluded")
    return exclude_indices


def build_pid_map(scan_csv_path: str):
    df = pd.read_csv(scan_csv_path)
    pid_map = {str(row["pt_index"]): row["patient_id"] for _, row in df.iterrows()}
    print(f"PID map: {len(pid_map)} entries, {len(set(pid_map.values()))} unique patients")
    return pid_map


def extract_patient_id(filepath: str) -> str:
    fname = os.path.basename(filepath)
    if "_ses-" in fname:
        return fname.split("_ses-")[0]
    return os.path.splitext(fname)[0]


def patient_stratified_split(paths, labels, train_ratio, val_ratio, test_ratio, seed, pid_map=None):
    patient_to_indices: dict[str, list[int]] = {}
    for idx, path in enumerate(paths):
        if pid_map is not None:
            stem = os.path.splitext(os.path.basename(path))[0]
            pid = pid_map.get(stem, stem)
        else:
            pid = extract_patient_id(path)
        patient_to_indices.setdefault(pid, []).append(idx)

    patient_ids = list(patient_to_indices.keys())
    patient_labels = [
        max(set(lbl := [labels[i] for i in patient_to_indices[pid]]), key=lbl.count)
        for pid in patient_ids
    ]

    val_test_ratio = val_ratio + test_ratio
    train_pids, valtest_pids, train_pl, valtest_pl = train_test_split(
        patient_ids, patient_labels,
        test_size=val_test_ratio, stratify=patient_labels, random_state=seed,
    )
    relative_test = test_ratio / val_test_ratio
    val_pids, test_pids, _, _ = train_test_split(
        valtest_pids, valtest_pl,
        test_size=relative_test, stratify=valtest_pl, random_state=seed,
    )

    def pids_to_data(pids):
        indices = [i for pid in pids for i in patient_to_indices[pid]]
        return [paths[i] for i in indices], [labels[i] for i in indices]

    train_d, val_d, test_d = pids_to_data(train_pids), pids_to_data(val_pids), pids_to_data(test_pids)
    print(f"Split (patients): train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")
    print(f"Split (scans):    train={len(train_d[0])}, val={len(val_d[0])}, test={len(test_d[0])}")
    return train_d, val_d, test_d


def patient_stratified_kfold(paths, labels, n_folds=5, seed=42, pid_map=None):
    patient_to_indices: dict[str, list[int]] = {}
    for idx, path in enumerate(paths):
        if pid_map is not None:
            stem = os.path.splitext(os.path.basename(path))[0]
            pid = pid_map.get(stem, stem)
        else:
            pid = extract_patient_id(path)
        patient_to_indices.setdefault(pid, []).append(idx)

    patient_ids = np.array(list(patient_to_indices.keys()))
    patient_labels = np.array([
        max(set(lbl := [labels[i] for i in patient_to_indices[pid]]), key=lbl.count)
        for pid in patient_ids
    ])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        tr_pids = patient_ids[tr_idx]
        va_pids = patient_ids[va_idx]
        tr_indices = [i for pid in tr_pids for i in patient_to_indices[pid]]
        va_indices = [i for pid in va_pids for i in patient_to_indices[pid]]
        folds.append((
            [paths[i] for i in tr_indices], [labels[i] for i in tr_indices],
            [paths[i] for i in va_indices], [labels[i] for i in va_indices],
        ))
        print(f"Fold {fold_idx}: train_patients={len(tr_pids)}, val_patients={len(va_pids)}, "
              f"train_scans={len(tr_indices)}, val_scans={len(va_indices)}")
    return folds


def compute_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    n_cls = len(counts)
    weights = torch.zeros(n_cls)
    for cls, cnt in counts.items():
        weights[cls] = total / (n_cls * cnt)
    return weights


def make_weighted_sampler(labels):
    counts = Counter(labels)
    class_weight = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = [class_weight[lbl] for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ── DataLoader builders ───────────────────────────────────────────────────────

def load_splits_from_csv(scan_csv_path: str, data_root: str, exclude_indices=None):
    """
    Load pre-assigned train/val/test splits from scan CSV 'split' column.
    The CSV must have been prepared with make_splits.py beforehand.

    Returns:
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
    """
    df = pd.read_csv(scan_csv_path)
    if "split" not in df.columns:
        raise RuntimeError(
            "'split' column not found in scan CSV. "
            "Run make_splits.py first to assign fixed splits."
        )
    split_map = {str(row["pt_index"]): row["split"] for _, row in df.iterrows()}

    paths, labels = collect_file_paths(data_root, exclude_indices=exclude_indices)

    train_paths, train_labels = [], []
    val_paths,   val_labels   = [], []
    test_paths,  test_labels  = [], []

    for path, label in zip(paths, labels):
        stem = os.path.splitext(os.path.basename(path))[0]
        split = split_map.get(stem)
        if split == "train":
            train_paths.append(path); train_labels.append(label)
        elif split == "val":
            val_paths.append(path);   val_labels.append(label)
        elif split == "test":
            test_paths.append(path);  test_labels.append(label)

    print(f"Loaded splits from CSV: train={len(train_paths)}, "
          f"val={len(val_paths)}, test={len(test_paths)}")
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def build_fusion_dataloaders(cfg: dict):
    """
    Build train/val/test DataLoaders for DivNetFusion.

    Returns:
        train_loader, val_loader, test_loader, class_weights, tab_mean, tab_std
    """
    data_cfg = cfg["data"]

    # Exclude set
    exclude_indices = None
    scan_csv  = data_cfg.get("scan_csv")
    excl_csv  = data_cfg.get("exclude_csv")
    if scan_csv and excl_csv:
        exclude_indices = build_exclude_set(scan_csv, excl_csv)

    if not scan_csv:
        raise RuntimeError("scan_csv must be set in config.")

    # Load fixed splits from scan CSV
    (tr_p, tr_l), (va_p, va_l), (te_p, te_l) = load_splits_from_csv(
        scan_csv, data_cfg["data_root"], exclude_indices=exclude_indices
    )

    if not tr_p:
        raise RuntimeError(
            f"No training samples found. Check data_root and split column in scan CSV."
        )

    # Build tabular lookup (with imputation report)
    master_csv = cfg["tabular"]["master_csv"]
    tab_lookup, _, _ = build_tabular_lookup(scan_csv, master_csv, report=True)

    # Normalization stats from training set only
    tab_mean, tab_std = compute_tab_normalization(tab_lookup, tr_p)
    print(f"Tabular normalization (train set):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name}: mean={tab_mean[i]:.4f}, std={tab_std[i]:.4f}")
    print()

    # Datasets
    train_ds = ADNIFusionDataset(tr_p, tr_l, tab_lookup, tab_mean, tab_std,
                                 augment=data_cfg.get("augment", True))
    val_ds   = ADNIFusionDataset(va_p, va_l, tab_lookup, tab_mean, tab_std, augment=False)
    test_ds  = ADNIFusionDataset(te_p, te_l, tab_lookup, tab_mean, tab_std, augment=False)

    train_sampler = make_weighted_sampler(tr_l)
    class_weights = compute_class_weights(tr_l)

    nw = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"],
        sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg["val_batch_size"],
        shuffle=False, num_workers=nw, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=data_cfg["val_batch_size"],
        shuffle=False, num_workers=nw, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights, tab_mean, tab_std


def build_fusion_dataloaders_kfold(cfg: dict, fold_idx: int, folds_data: list,
                                    tab_lookup: dict):
    """
    Build train/val DataLoaders for a single k-fold split.

    Returns:
        train_loader, val_loader, class_weights, tab_mean, tab_std
    """
    data_cfg = cfg["data"]
    tr_p, tr_l, va_p, va_l = folds_data[fold_idx]

    tab_mean, tab_std = compute_tab_normalization(tab_lookup, tr_p)

    train_ds = ADNIFusionDataset(tr_p, tr_l, tab_lookup, tab_mean, tab_std, augment=True)
    val_ds   = ADNIFusionDataset(va_p, va_l, tab_lookup, tab_mean, tab_std, augment=False)

    train_sampler = make_weighted_sampler(tr_l)
    class_weights = compute_class_weights(tr_l)

    nw = data_cfg["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"],
        sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg["val_batch_size"],
        shuffle=False, num_workers=nw, pin_memory=True,
    )
    return train_loader, val_loader, class_weights, tab_mean, tab_std
