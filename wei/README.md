# Wei et al. (2025) — 3D Decoupling AD Prediction Network

Implementation of: *A 3D decoupling Alzheimer's disease prediction network based on structural MRI*
Shicheng Wei et al., Health Information Science and Systems (2025)

---

## Architecture

```
Input (1×96×96×96)
  └─ Stem (Conv×2, total stride 4)
       └─ Stage 1: MSD×4  64→128ch   (single 3×3×3 + Group Decouple)
            └─ Stage 2: MSD×4  128→256ch  (kernels (1,3,5) then (1,3))
                 └─ Stage 3: MSD×6  256→512ch  (kernels (1,3,5) then (1,3))
                      └─ Stage 4: MSD×4  512→1024ch (single 3×3×3 + Group Decouple)
                           └─ SA Block (Multi-head Self-Attention)
                                └─ Global AvgPool → FC → output
```

**MSD (Multi-Scale Decoupling) Block**
- Applies different kernel sizes to three directional views (axial / coronal / sagittal), then concatenates
- Parallel group convolution decoupling (G=1, 2, 4) followed by 1×1×1 compression conv
- Residual connection

**SA (Self-Attention) Block**
- Flattens 3D feature map into spatial tokens
- Multi-head self-attention + FFN (standard Transformer encoder block)

**Joint Loss**
`L_total = L_CE + α × L_SC`
- `L_CE`: Cross-entropy loss
- `L_SC`: Clustering loss — pulls each sample toward its class center (centers updated per batch, training only)

---

## Data Structure

```
{data_root}/
  3D_tensors/
    CN/   0.pt  5.pt  9.pt  ...
    MCI/  3.pt  7.pt  ...
    AD/   1.pt  4.pt  ...
```

- Each `.pt` file: `float32` tensor, shape `[1, 192, 192, 192]`
- Trilinearly resized to `input_size=96` at load time (configurable)

**Split CSV** (`csv_splits_all_mri_scan_list.csv`):

| pt_index | image_path | patient_id | image_id | label | split |
|----------|-----------|------------|----------|-------|-------|
| 0 | .../xxx.nii.gz | 082_S_1256 | 63155 | CN | test |
| 5 | .../yyy.nii.gz | 082_S_4090 | 241401 | CN | train |

- `split` column is pre-assigned at **patient level** (via `make_splits.py`)
- All scans from the same patient belong to exactly one split — **no data leakage**

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration (`config.yaml`)

Key settings:

```yaml
data:
  root: /workspace/pumpkinlab-storage-dhl
  scan_csv: /workspace/cognitive-reserve-modeling/divnet/csv_splits_all_mri_scan_list.csv
  task: AD_vs_NC      # 'AD_vs_NC' or '3class'
  input_size: 96      # volumes resized from 192 to this size via trilinear interpolation

model:
  base_ch: 64         # reduce to 32 if GPU memory is limited
  num_heads: 8
  dropout: 0.1

training:
  epochs: 100
  lr: 1.0e-4
  alpha: 0.1          # clustering loss weight
  patience: 5         # early stopping patience
```

**Task options:**
- `AD_vs_NC` — CN (0) vs AD (1); MCI excluded
- `3class` — CN (0) vs MCI (1) vs AD (2)

---

## Usage

### Training

```bash
cd /path/to/wei
python3 train.py --config config.yaml
```

Checkpoints and training history are saved to `output.dir` (default: `./runs`):

```
runs/
  best.pt          # checkpoint at best validation loss
  history.json     # per-epoch train/val metrics
```

### Evaluation

```bash
# evaluate on test set (default)
python evaluate.py --config config.yaml

# evaluate on validation set
python evaluate.py --config config.yaml --split val
```

Reported metrics: **ACC, SEN, SPE, AUC**
Results are saved to `runs/metrics_test.json`.

---

## GPU Memory Guide

| base_ch | feat_dim | input_size | batch_size | Est. VRAM |
|---------|----------|------------|------------|-----------|
| 64 | 1024 | 96 | 4 | ~14 GB |
| 64 | 1024 | 96 | 2 | ~8 GB |
| 32 | 512  | 96 | 4 | ~6 GB |

If you run out of memory, reduce `base_ch: 32` or `batch_size: 2` in `config.yaml`.

---

## File Structure

```
wei/
  config.yaml      main configuration
  model.py         DecouplingADNet (MSD Block, SA Block)
  loss.py          JointLoss (CrossEntropy + ClusteringLoss)
  dataset.py       WeiDataset — .pt loading, augmentation, split loading
  utils.py         metrics (ACC/SEN/SPE/AUC), checkpoint save/load
  train.py         training loop
  evaluate.py      evaluation script
  requirements.txt
  README.md
```
