"""
Training script for DivNetFusion: late fusion of 3D CNN (DivNet) + tabular MLP.

Usage:
    python divnet_fusion_train.py --config divnet_fusion_config.yaml
    python divnet_fusion_train.py --config divnet_fusion_config.yaml --test
    python divnet_fusion_train.py --config divnet_fusion_config.yaml --kfold
"""

import argparse
import datetime
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import wandb

from divnet_fusion_model import DivNetFusion
from divnet_fusion_dataset import (
    CLASS_MAP,
    FEATURE_NAMES,
    build_exclude_set,
    build_fusion_dataloaders,
    build_fusion_dataloaders_kfold,
    build_pid_map,
    build_tabular_lookup,
    collect_file_paths,
    compute_class_weights,
    compute_tab_normalization,
    patient_stratified_kfold,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DivNetFusion for AD classification")
    p.add_argument("--config", type=str, default="divnet_fusion_config.yaml")
    p.add_argument("--test",   action="store_true", help="Test-only mode (requires checkpoint)")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    p.add_argument("--kfold",  action="store_true", help="5-fold cross-validation mode")
    p.add_argument("--gpu",    type=int, default=0)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_batch(volumes, tabs, labels, alpha, device):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(volumes.size(0), device=device)
    return (
        lam * volumes + (1 - lam) * volumes[idx],
        lam * tabs    + (1 - lam) * tabs[idx],
        labels, labels[idx], lam,
    )


def mixup_criterion(criterion, outputs, la, lb, lam):
    return lam * criterion(outputs, la) + (1 - lam) * criterion(outputs, lb)


# ── Train / Validate ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip,
                    mixup_alpha=0.0):
    model.train()
    running_loss = correct = total = 0

    for volumes, tabs, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        tabs    = tabs.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            volumes, tabs, la, lb, lam = mixup_batch(volumes, tabs, labels,
                                                      mixup_alpha, device)
            outputs = model(volumes, tabs)
            loss = mixup_criterion(criterion, outputs, la, lb, lam)
        else:
            outputs = model(volumes, tabs)
            loss = criterion(outputs, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item() * volumes.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for volumes, tabs, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        tabs    = tabs.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        outputs = model(volumes, tabs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * volumes.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    total = len(all_labels)

    metrics = {
        "loss":              running_loss / total,
        "accuracy":          100.0 * np.sum(all_preds == all_labels) / total,
        "balanced_accuracy": 100.0 * balanced_accuracy_score(all_labels, all_preds),
        "f1_macro":          f1_score(all_labels, all_preds, average="macro",    zero_division=0),
        "f1_weighted":       f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "confusion_matrix":  confusion_matrix(all_labels, all_preds,
                                              labels=list(range(num_classes))),
    }
    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for c in range(num_classes):
        metrics[f"f1_class_{c}"] = float(f1_per[c]) if c < len(f1_per) else 0.0
    metrics.update(_compute_auc(all_labels, all_probs, num_classes))
    return metrics


def _compute_auc(labels, probs, num_classes):
    results = {}
    unique = np.unique(labels)
    if len(unique) < 2:
        for c in range(num_classes):
            results[f"auc_class_{c}"] = 0.0
        results["micro_auc"] = results["macro_auc"] = 0.0
        return results

    onehot = np.zeros((len(labels), num_classes))
    for i, lbl in enumerate(labels):
        onehot[i, lbl] = 1

    per_class = []
    for c in range(num_classes):
        if c in unique and onehot[:, c].sum() > 0:
            try:
                auc = roc_auc_score(onehot[:, c], probs[:, c])
            except ValueError:
                auc = 0.0
        else:
            auc = 0.0
        results[f"auc_class_{c}"] = auc
        per_class.append(auc)

    valid = [a for a in per_class if a > 0]
    results["macro_auc"] = float(np.mean(valid)) if valid else 0.0
    try:
        results["micro_auc"] = roc_auc_score(onehot, probs, average="micro")
    except ValueError:
        results["micro_auc"] = 0.0
    return results


def _per_class_metrics(cm, class_names):
    out = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[name] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        }
    return out


# ── Printing / Plotting ───────────────────────────────────────────────────────

def print_metrics(metrics, class_names, phase="Validation"):
    print(f"\n{'='*60}")
    print(f"  {phase} Results")
    print(f"{'='*60}")
    print(f"  Loss:              {metrics['loss']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  Micro AUC:         {metrics['micro_auc']:.4f}")
    print(f"  Macro AUC:         {metrics['macro_auc']:.4f}")
    for i, name in enumerate(class_names):
        print(f"  AUC ({name}):         {metrics[f'auc_class_{i}']:.4f}")
        print(f"  F1  ({name}):         {metrics[f'f1_class_{i}']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print("          " + "  ".join(f"{n:>5}" for n in class_names))
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i,j]:5d}" for j in range(len(class_names)))
        print(f"  {name:>8}  {row}")

    per = _per_class_metrics(cm, class_names)
    print(f"\n  Per-class metrics:")
    for name in class_names:
        m = per[name]
        print(f"    {name}: Sensitivity={m['sensitivity']:.4f}  Specificity={m['specificity']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(cm, class_names, phase="Validation", save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)
    cm_flipped = cm[::-1, :]
    names_y = list(reversed(class_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_flipped, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=class_names, yticklabels=names_y,
        xlabel="Predicted label", ylabel="True label",
        title=f"{phase} Confusion Matrix",
    )
    thresh = cm_flipped.max() / 2.0
    for i in range(cm_flipped.shape[0]):
        for j in range(cm_flipped.shape[1]):
            ax.text(j, i, str(cm_flipped[i, j]), ha="center", va="center",
                    color="white" if cm_flipped[i, j] > thresh else "black", fontsize=14)
    fig.tight_layout()
    fname = f"{phase.lower().replace(' ', '_')}_confusion_matrix.png"
    save_path = os.path.join(save_dir, fname)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {save_path}")


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg, device):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    save_cfg  = cfg["save"]
    class_names = list(CLASS_MAP.keys())

    print("Loading data...")
    train_loader, val_loader, test_loader, class_weights, tab_mean, tab_std = \
        build_fusion_dataloaders(cfg)
    print(f"Class weights: {class_weights.tolist()}")

    model = DivNetFusion(
        num_filters=model_cfg["num_filters"],
        tab_input_dim=len(FEATURE_NAMES),
        mlp_hidden=model_cfg["mlp_hidden"],
        num_classes=model_cfg["num_classes"],
        dropout1=model_cfg["dropout1"],
        dropout2=model_cfg["dropout2"],
        tab_dropout=model_cfg.get("tab_dropout", 0.3),
    ).to(device)

    # Optionally load pretrained CNN backbone
    pretrained = model_cfg.get("pretrained_backbone")
    if pretrained and os.path.isfile(pretrained):
        print(f"Loading pretrained CNN backbone from: {pretrained}")
        model.load_cnn_backbone(pretrained, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Use only class-weighted loss (WeightedRandomSampler already balances batches)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    if train_cfg.get("optimizer", "SGD").upper() == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=train_cfg["lr"],
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg["weight_decay"],
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg["lr_milestones"],
        gamma=train_cfg["lr_gamma"],
    )

    best = {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_auc": 0.0, "loss": float("inf")}
    patience_counter = 0
    ckpt_dir = save_cfg["checkpoint_dir"]

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        wandb.login()
        wandb.init(
            project=wandb_cfg.get("project", "DivNetFusion AD Classification"),
            name=f"DivNetFusion {datetime.datetime.now():%Y%m%d-%H%M}",
            config=cfg,
        )

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"Device: {device} | Batch size: {cfg['data']['batch_size']} | "
          f"LR: {train_cfg['lr']} | Milestones: {train_cfg['lr_milestones']}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=train_cfg["grad_clip"],
            mixup_alpha=train_cfg.get("mixup_alpha", 0.0),
        )
        val_m = validate(model, val_loader, criterion, device,
                         num_classes=model_cfg["num_classes"])
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:3d}/{train_cfg['epochs']}]  "
              f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.2f}%  |  "
              f"Val Loss: {val_m['loss']:.4f}  "
              f"Val Acc: {val_m['accuracy']:.2f}%  "
              f"Val BAcc: {val_m['balanced_accuracy']:.2f}%  "
              f"Val mAUC: {val_m['macro_auc']:.4f}  "
              f"LR: {lr:.6f}  Time: {elapsed:.1f}s")

        if wandb.run is not None:
            wandb.log({
                "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_m["loss"], "val_acc": val_m["accuracy"],
                "val_balanced_acc": val_m["balanced_accuracy"],
                "val_f1_macro": val_m["f1_macro"], "val_macro_auc": val_m["macro_auc"],
                "lr": lr,
            })

        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_m,
            "tab_mean": tab_mean,
            "tab_std": tab_std,
        }

        if val_m["accuracy"] > best["accuracy"]:
            best["accuracy"] = val_m["accuracy"]
            save_checkpoint(ckpt_state, os.path.join(ckpt_dir, "best_accuracy.pth"))
            print(f"  -> New best accuracy: {best['accuracy']:.2f}%")

        if val_m["balanced_accuracy"] > best["balanced_accuracy"]:
            best["balanced_accuracy"] = val_m["balanced_accuracy"]
            save_checkpoint(ckpt_state, os.path.join(ckpt_dir, "best_balanced_acc.pth"))
            print(f"  -> New best balanced accuracy: {best['balanced_accuracy']:.2f}%")

        if val_m["macro_auc"] > best["macro_auc"]:
            best["macro_auc"] = val_m["macro_auc"]
            save_checkpoint(ckpt_state, os.path.join(ckpt_dir, "best_macro_auc.pth"))
            print(f"  -> New best macro AUC: {best['macro_auc']:.4f}")

        if val_m["loss"] < best["loss"]:
            best["loss"] = val_m["loss"]
            patience_counter = 0
            save_checkpoint(ckpt_state, os.path.join(ckpt_dir, "lowest_loss.pth"))
            print(f"  -> New lowest loss: {best['loss']:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(patience={train_cfg['early_stopping_patience']})")
            break

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best Accuracy:          {best['accuracy']:.2f}%")
    print(f"Best Balanced Accuracy: {best['balanced_accuracy']:.2f}%")
    print(f"Best Macro AUC:         {best['macro_auc']:.4f}")
    print(f"Lowest Val Loss:        {best['loss']:.4f}")

    # Test set evaluation
    best_ckpt = os.path.join(ckpt_dir, "best_balanced_acc.pth")
    if os.path.exists(best_ckpt):
        print("\nEvaluating best model on test set...")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_m = validate(model, test_loader, criterion, device,
                          num_classes=model_cfg["num_classes"])
        print_metrics(test_m, class_names, phase="Test")
        plot_confusion_matrix(test_m["confusion_matrix"], class_names,
                              phase="Test",
                              save_dir=os.path.join(ckpt_dir, "figures"))
        if wandb.run is not None:
            wandb.log({
                "test_loss": test_m["loss"], "test_acc": test_m["accuracy"],
                "test_balanced_acc": test_m["balanced_accuracy"],
                "test_f1_macro": test_m["f1_macro"],
                "test_macro_auc": test_m["macro_auc"],
            })

    if wandb.run is not None:
        wandb.finish()

    return model


# ── K-fold ────────────────────────────────────────────────────────────────────

def train_kfold(cfg, device):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    save_cfg  = cfg["save"]
    cv_cfg    = cfg.get("cross_validation", {})
    k_folds   = cv_cfg.get("k_folds", 5)
    class_names = list(CLASS_MAP.keys())

    data_cfg = cfg["data"]
    scan_csv = data_cfg.get("scan_csv")
    excl_csv = data_cfg.get("exclude_csv")

    exclude_indices = build_exclude_set(scan_csv, excl_csv) if (scan_csv and excl_csv) else None
    pid_map = build_pid_map(scan_csv) if scan_csv else None
    paths, labels = collect_file_paths(data_cfg["data_root"], exclude_indices)

    # Build tabular lookup once
    master_csv = cfg["tabular"]["master_csv"]
    tab_lookup, _, _ = build_tabular_lookup(scan_csv, master_csv, report=True)

    folds_data = patient_stratified_kfold(
        paths, labels, n_folds=k_folds, seed=data_cfg["seed"], pid_map=pid_map
    )

    all_fold_metrics = []
    wandb_cfg = cfg.get("wandb", {})

    for fold_idx in range(k_folds):
        print(f"\n{'#'*60}\n  FOLD {fold_idx + 1} / {k_folds}\n{'#'*60}\n")

        if wandb_cfg.get("enabled", True):
            wandb.login()
            wandb.init(
                project=wandb_cfg.get("project", "DivNetFusion AD Classification"),
                name=f"DivNetFusion fold-{fold_idx}",
                group="kfold_cv",
                config={**cfg, "fold": fold_idx},
                reinit=True,
            )

        train_loader, val_loader, class_weights, tab_mean, tab_std = \
            build_fusion_dataloaders_kfold(cfg, fold_idx, folds_data, tab_lookup)

        model = DivNetFusion(
            num_filters=model_cfg["num_filters"],
            tab_input_dim=len(FEATURE_NAMES),
            mlp_hidden=model_cfg["mlp_hidden"],
            num_classes=model_cfg["num_classes"],
            dropout1=model_cfg["dropout1"],
            dropout2=model_cfg["dropout2"],
            tab_dropout=model_cfg.get("tab_dropout", 0.3),
        ).to(device)

        pretrained = model_cfg.get("pretrained_backbone")
        if pretrained and os.path.isfile(pretrained):
            model.load_cnn_backbone(pretrained, device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        if train_cfg.get("optimizer", "SGD").upper() == "ADAM":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=train_cfg["lr"],
                momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg["weight_decay"],
            )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=train_cfg["lr_milestones"], gamma=train_cfg["lr_gamma"]
        )

        best = {"balanced_accuracy": 0.0, "loss": float("inf")}
        best_metrics = None
        patience_counter = 0
        fold_ckpt_dir = os.path.join(save_cfg["checkpoint_dir"], f"fold_{fold_idx}")

        for epoch in range(1, train_cfg["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                grad_clip=train_cfg["grad_clip"],
            )
            val_m = validate(model, val_loader, criterion, device,
                             num_classes=model_cfg["num_classes"])
            scheduler.step()

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"Fold {fold_idx} Epoch [{epoch:3d}/{train_cfg['epochs']}]  "
                  f"Train Loss: {tr_loss:.4f}  |  "
                  f"Val Loss: {val_m['loss']:.4f}  "
                  f"Val BAcc: {val_m['balanced_accuracy']:.2f}%  "
                  f"Val mAUC: {val_m['macro_auc']:.4f}  "
                  f"LR: {lr:.6f}  Time: {elapsed:.1f}s")

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": val_m["loss"], "val_balanced_acc": val_m["balanced_accuracy"],
                    "val_f1_macro": val_m["f1_macro"], "val_macro_auc": val_m["macro_auc"],
                    "lr": lr,
                })

            if val_m["balanced_accuracy"] > best["balanced_accuracy"]:
                best["balanced_accuracy"] = val_m["balanced_accuracy"]
                best_metrics = val_m.copy()
                save_checkpoint(
                    {"epoch": epoch, "fold": fold_idx,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(),
                     "val_metrics": val_m,
                     "tab_mean": tab_mean, "tab_std": tab_std},
                    os.path.join(fold_ckpt_dir, "best_balanced_acc.pth"),
                )
                print(f"  -> New best balanced accuracy: {best['balanced_accuracy']:.2f}%")

            if val_m["loss"] < best["loss"]:
                best["loss"] = val_m["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        best_ckpt = os.path.join(fold_ckpt_dir, "best_balanced_acc.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            best_metrics = validate(model, val_loader, criterion, device,
                                    num_classes=model_cfg["num_classes"])

        print_metrics(best_metrics, class_names, phase=f"Fold {fold_idx} Best Val")
        plot_confusion_matrix(best_metrics["confusion_matrix"], class_names,
                              phase=f"Fold {fold_idx}",
                              save_dir=os.path.join(save_cfg["checkpoint_dir"], "figures"))

        cm = best_metrics["confusion_matrix"]
        per = _per_class_metrics(cm, class_names)
        fold_rec = {
            "accuracy": best_metrics["accuracy"],
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "f1_macro": best_metrics["f1_macro"],
            "macro_auc": best_metrics["macro_auc"],
        }
        for i, name in enumerate(class_names):
            fold_rec[f"auc_{name}"]         = best_metrics[f"auc_class_{i}"]
            fold_rec[f"sensitivity_{name}"] = per[name]["sensitivity"]
            fold_rec[f"specificity_{name}"] = per[name]["specificity"]
        all_fold_metrics.append(fold_rec)

        if wandb.run is not None:
            wandb.finish()

    # Summary
    print(f"\n{'='*60}\n  {k_folds}-Fold Cross-Validation Summary\n{'='*60}")
    for key in all_fold_metrics[0]:
        vals = [m[key] for m in all_fold_metrics]
        print(f"  {key:35s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(f"{'='*60}\n")


# ── Test-only mode ────────────────────────────────────────────────────────────

def test_only(cfg, checkpoint_path, device):
    model_cfg   = cfg["model"]
    class_names = list(CLASS_MAP.keys())

    print("Loading data...")
    _, val_loader, test_loader, class_weights, tab_mean, tab_std = \
        build_fusion_dataloaders(cfg)

    model = DivNetFusion(
        num_filters=model_cfg["num_filters"],
        tab_input_dim=len(FEATURE_NAMES),
        mlp_hidden=model_cfg["mlp_hidden"],
        num_classes=model_cfg["num_classes"],
        dropout1=model_cfg["dropout1"],
        dropout2=model_cfg["dropout2"],
        tab_dropout=model_cfg.get("tab_dropout", 0.3),
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint from epoch {ckpt.get('epoch', '?')}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    save_dir  = os.path.join(cfg["save"]["checkpoint_dir"], "figures")

    val_m = validate(model, val_loader, criterion, device,
                     num_classes=model_cfg["num_classes"])
    print_metrics(val_m, class_names, phase="Validation")
    plot_confusion_matrix(val_m["confusion_matrix"], class_names,
                          phase="Validation", save_dir=save_dir)

    test_m = validate(model, test_loader, criterion, device,
                      num_classes=model_cfg["num_classes"])
    print_metrics(test_m, class_names, phase="Test")
    plot_confusion_matrix(test_m["confusion_matrix"], class_names,
                          phase="Test", save_dir=save_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.test:
        ckpt_path = args.resume or os.path.join(
            cfg["save"]["checkpoint_dir"], "best_balanced_acc.pth"
        )
        test_only(cfg, ckpt_path, device)
    elif args.kfold:
        train_kfold(cfg, device)
    else:
        train(cfg, device)


if __name__ == "__main__":
    main()
