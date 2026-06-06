"""
Training script for 3D Decoupling AD Network (Wei et al., 2025).

Usage:
    python train.py --config config.yaml

Data splits:
    Pre-assigned in scan CSV (patient-level). No split logic here.
    Augmentation only on train set. Clustering-loss centers updated only on train set.
"""

import argparse
import logging
import os
import json

import numpy as np
import torch
import yaml

from dataset import build_dataloaders, TASK_LABEL_MAPS
from loss import JointLoss
from model import DecouplingADNet
from utils import compute_metrics, load_checkpoint, save_checkpoint, log_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, mode="train",
              log_interval=10):
    is_train = mode == "train"
    model.train(is_train)
    criterion.train(is_train)   # controls clustering-center updates

    total_loss = total_ce = total_sc = 0.0
    all_labels, all_preds, all_scores = [], [], []

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits, feat = model(imgs)
            loss, l_ce, l_sc = criterion(logits, feat, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_ce   += l_ce
        total_sc   += l_sc

        probs = torch.softmax(logits, dim=1)[:, -1].detach().cpu().numpy()
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)
        all_scores.extend(probs)

        if is_train and (step + 1) % log_interval == 0:
            logger.info(
                "  step %d/%d  loss=%.4f  ce=%.4f  sc=%.4f",
                step + 1, len(loader),
                total_loss / (step + 1),
                total_ce   / (step + 1),
                total_sc   / (step + 1),
            )

    n = max(len(loader), 1)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )
    metrics["loss"] = total_loss / n
    metrics["ce"]   = total_ce   / n
    metrics["sc"]   = total_sc   / n
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict):
    out_dir   = cfg["output"]["dir"]
    ckpt_path = os.path.join(out_dir, "best.pt")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- data --------------------------------------------------------------
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)

    # ---- model -------------------------------------------------------------
    task        = cfg["data"]["task"]
    num_classes = len(TASK_LABEL_MAPS[task])
    base_ch     = cfg["model"]["base_ch"]
    feat_dim    = base_ch * 16

    model = DecouplingADNet(
        num_classes = num_classes,
        base_ch     = base_ch,
        num_heads   = cfg["model"]["num_heads"],
        dropout     = cfg["model"]["dropout"],
    ).to(device)

    logger.info(
        "Model: %d classes  base_ch=%d  feat_dim=%d",
        num_classes, base_ch, feat_dim,
    )

    # ---- loss & optimiser --------------------------------------------------
    criterion = JointLoss(
        num_classes = num_classes,
        feat_dim    = feat_dim,
        alpha       = cfg["training"]["alpha"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    # ---- resume ------------------------------------------------------------
    start_epoch    = 0
    best_val_loss  = float("inf")
    if os.path.exists(ckpt_path):
        start_epoch, best_val_loss = load_checkpoint(
            ckpt_path, model, optimizer, criterion, device
        )
        start_epoch += 1

    # ---- training loop -----------------------------------------------------
    patience     = cfg["training"]["patience"]
    min_delta    = cfg["training"]["min_delta"]
    log_interval = cfg["output"]["log_interval"]
    epochs       = cfg["training"]["epochs"]
    no_improve   = 0
    history      = []

    for epoch in range(start_epoch, epochs):
        logger.info("=== Epoch %d/%d ===", epoch + 1, epochs)

        train_m = run_epoch(model, train_loader, criterion, optimizer, device,
                            mode="train", log_interval=log_interval)
        log_metrics(train_m, prefix="TRAIN  ")

        val_m = run_epoch(model, val_loader, criterion, optimizer, device,
                          mode="val")
        log_metrics(val_m, prefix="VAL    ")

        history.append({"epoch": epoch + 1, "train": train_m, "val": val_m})
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ---- checkpoint & early stopping -----------------------------------
        val_loss = val_m["loss"]
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint(
                {"epoch": epoch,
                 "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "loss_fn": criterion.state_dict(),
                 "best_val_loss": best_val_loss},
                ckpt_path,
            )
            logger.info("  --> New best val_loss=%.4f  (saved)", best_val_loss)
        else:
            no_improve += 1
            logger.info(
                "  No improvement %d/%d (best=%.4f)",
                no_improve, patience, best_val_loss,
            )
            if no_improve >= patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training finished. Best val_loss=%.4f", best_val_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
