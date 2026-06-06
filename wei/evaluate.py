"""
Evaluation script for 3D Decoupling AD Network (Wei et al., 2025).

Usage:
    python evaluate.py --config config.yaml [--split test]

Loads the best checkpoint and reports ACC, SEN, SPE, AUC.
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
from utils import compute_metrics, load_checkpoint, log_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(cfg: dict, split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- data --------------------------------------------------------------
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]

    # ---- model & checkpoint ------------------------------------------------
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

    criterion = JointLoss(num_classes, feat_dim, cfg["training"]["alpha"]).to(device)

    ckpt_path = os.path.join(cfg["output"]["dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")

    load_checkpoint(ckpt_path, model, loss_fn=criterion, device=device)

    # ---- inference ---------------------------------------------------------
    model.eval()
    criterion.eval()

    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, -1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_scores.extend(probs)

    # ---- metrics -----------------------------------------------------------
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )

    logger.info("=== Results on %s set ===", split.upper())
    log_metrics(metrics)

    out_dir  = cfg["output"]["dir"]
    out_path = os.path.join(out_dir, f"metrics_{split}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics → %s", out_path)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split",  default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, split=args.split)


if __name__ == "__main__":
    main()
