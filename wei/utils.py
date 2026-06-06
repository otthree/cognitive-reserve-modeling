"""
Utilities: subject-level splitting, metrics, checkpoint helpers.
"""

import os
import logging
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)

TASK_VALID_DIAGS = {
    "AD_vs_NC":    ["CN", "NC", "AD"],
    "pMCI_vs_sMCI": ["sMCI", "pMCI"],
}


# ---------------------------------------------------------------------------
# Subject-level split
# ---------------------------------------------------------------------------

def split_by_subject(tsv_path: str, task: str, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, seed: int = 42):
    """
    Split by participant_id so no subject appears in more than one split.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    sep = "\t" if tsv_path.endswith(".tsv") else ","
    df = pd.read_csv(tsv_path, sep=sep)

    # Keep only task-relevant labels
    valid = TASK_VALID_DIAGS[task]
    df = df[df["diagnosis"].isin(valid)].copy()

    # One representative row per subject (for stratified split)
    subj_df = df.groupby("participant_id")["diagnosis"].first().reset_index()
    subjects = subj_df["participant_id"].values
    labels = subj_df["diagnosis"].values

    # Split subjects: all → train+val vs test
    train_val_subj, test_subj, train_val_lbl, _ = train_test_split(
        subjects, labels, test_size=test_ratio, random_state=seed, stratify=labels
    )

    # Split train+val → train vs val
    val_frac = val_ratio / (1.0 - test_ratio)
    train_subj, val_subj = train_test_split(
        train_val_subj, test_size=val_frac, random_state=seed, stratify=train_val_lbl
    )

    train_set = set(train_subj)
    val_set = set(val_subj)
    test_set = set(test_subj)

    # Verify no overlap (data leakage guard)
    assert train_set.isdisjoint(val_set), "Leakage: train and val share subjects"
    assert train_set.isdisjoint(test_set), "Leakage: train and test share subjects"
    assert val_set.isdisjoint(test_set), "Leakage: val and test share subjects"

    train_df = df[df["participant_id"].isin(train_set)].reset_index(drop=True)
    val_df   = df[df["participant_id"].isin(val_set)].reset_index(drop=True)
    test_df  = df[df["participant_id"].isin(test_set)].reset_index(drop=True)

    logger.info(
        "Split sizes  train=%d  val=%d  test=%d  (sessions)",
        len(train_df), len(val_df), len(test_df)
    )
    logger.info(
        "Subject counts  train=%d  val=%d  test=%d",
        len(train_set), len(val_set), len(test_set)
    )

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, split_dir: str):
    os.makedirs(split_dir, exist_ok=True)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        df.to_csv(os.path.join(split_dir, f"{name}.tsv"), sep="\t", index=False)
    logger.info("Saved splits to %s", split_dir)


def load_splits(split_dir: str):
    dfs = {}
    for name in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{name}.tsv")
        dfs[name] = pd.read_csv(path, sep="\t")
    return dfs["train"], dfs["val"], dfs["test"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels_true: np.ndarray, labels_pred: np.ndarray,
                    scores: np.ndarray) -> dict:
    """
    Compute ACC, SEN, SPE, AUC.

    For binary tasks: standard TP/TN/FP/FN from confusion matrix.
    For 3-class tasks: macro-averaged SEN/SPE; AUC via one-vs-rest (scores
      should be the probability of the last class, used only for binary).

    labels_true / labels_pred: 1-D integer arrays.
    scores: predicted probability for the positive class (last class index).
    """
    num_classes = len(set(labels_true.tolist()))

    acc = (labels_true == labels_pred).mean()

    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(labels_true, labels_pred,
                                          labels=[0, 1]).ravel()
        sen = tp / (tp + fn + 1e-9)
        spe = tn / (tn + fp + 1e-9)
        try:
            auc = roc_auc_score(labels_true, scores)
        except ValueError:
            auc = float("nan")
    else:
        # Macro-average across classes
        cm = confusion_matrix(labels_true, labels_pred)
        sen_list, spe_list = [], []
        for c in range(num_classes):
            tp = cm[c, c]
            fn = cm[c, :].sum() - tp
            fp = cm[:, c].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sen_list.append(tp / (tp + fn + 1e-9))
            spe_list.append(tn / (tn + fp + 1e-9))
        sen = float(np.mean(sen_list))
        spe = float(np.mean(spe_list))
        auc = float("nan")   # multi-class AUC requires per-class probs

    return {"ACC": float(acc), "SEN": float(sen), "SPE": float(spe),
            "AUC": float(auc)}


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info("Saved checkpoint → %s", path)


def load_checkpoint(path: str, model, optimizer=None, loss_fn=None, device="cpu"):
    import torch
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if loss_fn and "loss_fn" in ckpt:
        loss_fn.load_state_dict(ckpt["loss_fn"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1))
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


def log_metrics(metrics: dict, prefix: str = ""):
    parts = [f"{prefix}{k}={v:.4f}" for k, v in metrics.items()]
    logger.info("  ".join(parts))


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
