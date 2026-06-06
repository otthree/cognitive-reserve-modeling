"""
Dataset for 3D Decoupling AD Network (Wei et al., 2025).

Data format (from divnet preprocessing):
  {data_root}/3D_tensors/{CN,MCI,AD}/{pt_index}.pt
  Each .pt file: float32 tensor shape [1, 192, 192, 192] (or [192, 192, 192])

Split metadata:
  csv_splits_all_mri_scan_list.csv
  Columns: pt_index, image_path, patient_id, image_id, label, split
  - split column already assigns train/val/test at patient level → no leakage.

Supported tasks:
  'AD_vs_NC'  → keep CN(=0) and AD(=1)
  '3class'    → keep CN(=0), MCI(=1), AD(=2)

Augmentation is applied ONLY in train mode.
"""

import os
import random
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


TASK_LABEL_MAPS = {
    "AD_vs_NC": {"CN": 0, "AD": 1},
    "3class":   {"CN": 0, "MCI": 1, "AD": 2},
}


# ---------------------------------------------------------------------------
# Split loading from CSV
# ---------------------------------------------------------------------------

def load_splits_from_csv(scan_csv: str, data_root: str, task: str):
    """
    Load pre-assigned train/val/test splits from the scan CSV.

    The CSV split column is already patient-level (no leakage guaranteed by
    make_splits.py). We only filter rows whose label is relevant to `task`.

    Returns:
        train_data, val_data, test_data: each is (paths, labels) tuple
    """
    label_map = TASK_LABEL_MAPS[task]

    df = pd.read_csv(scan_csv)

    # Keep only task-relevant labels
    df = df[df["label"].isin(label_map.keys())].copy()

    # Build file paths:  {data_root}/3D_tensors/{label}/{pt_index}.pt
    df["file_path"] = df.apply(
        lambda r: os.path.join(data_root, "3D_tensors", r["label"], f"{r['pt_index']}.pt"),
        axis=1,
    )
    df["int_label"] = df["label"].map(label_map)

    splits = {}
    for split_name in ("train", "val", "test"):
        sub = df[df["split"] == split_name].reset_index(drop=True)
        splits[split_name] = (
            sub["file_path"].tolist(),
            sub["int_label"].tolist(),
        )

    for name, (paths, labels) in splits.items():
        counts = Counter(labels)
        print(f"{name:5s}: {len(paths)} scans  {dict(counts)}")

    # Verify patient-level non-overlap (leakage guard)
    if "patient_id" in df.columns:
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            pids_a = set(df[df["split"] == a]["patient_id"])
            pids_b = set(df[df["split"] == b]["patient_id"])
            overlap = pids_a & pids_b
            assert not overlap, f"Data leakage: {len(overlap)} patients in both {a} and {b}"

    return splits["train"], splits["val"], splits["test"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WeiDataset(Dataset):
    """
    Loads preprocessed .pt MRI volumes for Wei et al. network.

    Parameters
    ----------
    paths : list[str]
        Full paths to .pt files.
    labels : list[int]
        Integer class labels.
    mode : str
        'train' applies augmentation; 'val' / 'test' does not.
    input_size : int
        Target cubic spatial size. Volumes are trilinearly resized if needed.
    """

    def __init__(self, paths: list, labels: list, mode: str = "train",
                 input_size: int = 96):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.mode = mode
        self.input_size = input_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        volume = torch.load(self.paths[idx], map_location="cpu", weights_only=True)

        # Ensure shape [1, D, H, W]
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)

        # Resize to target input_size if needed (trilinear)
        if volume.shape[-1] != self.input_size:
            volume = F.interpolate(
                volume.unsqueeze(0),
                size=self.input_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        # Per-volume min-max normalisation
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 1e-8:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = torch.zeros_like(volume)

        if self.mode == "train":
            volume = self._augment(volume)

        return volume.float(), self.labels[idx]

    # ------------------------------------------------------------------
    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        S = self.input_size

        # Random left-right flip
        if random.random() > 0.5:
            volume = torch.flip(volume, dims=[1])

        # Random Gaussian noise
        volume = volume + torch.randn_like(volume) * 0.01

        # Random intensity shift
        volume = volume + random.uniform(-0.1, 0.1)

        # Random patch masking
        if random.random() > 0.5:
            ps = S // 8
            d0 = random.randint(0, S - ps)
            h0 = random.randint(0, S - ps)
            w0 = random.randint(0, S - ps)
            volume = volume.clone()
            volume[:, d0:d0+ps, h0:h0+ps, w0:w0+ps] = 0.0

        return volume.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    counts = Counter(labels)
    class_weight = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = [class_weight[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    return weights


def build_dataloaders(cfg: dict):
    """
    Build train / val / test DataLoaders from config.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    data_cfg = cfg["data"]
    task = data_cfg["task"]

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        load_splits_from_csv(
            scan_csv  = data_cfg["scan_csv"],
            data_root = data_cfg["root"],
            task      = task,
        )

    if not train_paths:
        raise RuntimeError("No training samples found. Check scan_csv and data_root.")

    input_size = data_cfg.get("input_size", 96)
    nw = data_cfg.get("num_workers", 4)

    train_ds = WeiDataset(train_paths, train_labels, mode="train",  input_size=input_size)
    val_ds   = WeiDataset(val_paths,   val_labels,   mode="val",    input_size=input_size)
    test_ds  = WeiDataset(test_paths,  test_labels,  mode="test",   input_size=input_size)

    train_loader = DataLoader(
        train_ds,
        batch_size  = data_cfg["batch_size"],
        sampler     = make_weighted_sampler(train_labels),
        num_workers = nw,
        pin_memory  = True,
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        shuffle     = False,
        num_workers = nw,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        shuffle     = False,
        num_workers = nw,
        pin_memory  = True,
    )

    num_classes   = len(TASK_LABEL_MAPS[task])
    class_weights = compute_class_weights(train_labels, num_classes)

    return train_loader, val_loader, test_loader, class_weights
