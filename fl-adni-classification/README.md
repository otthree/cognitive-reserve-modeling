# ADNI: Centralized 3-Way Alzheimer's Disease Classification on 3D MRI Data

<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue.svg?style=flat-square)](https://pytorch.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=flat-square)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat-square)](https://github.com/astral-sh/ruff)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-purple.svg?style=flat-square)](https://monai.io/)

</div>

Centralized deep learning pipeline for **CN vs MCI vs AD** 3-way classification using 3D MRI scans from the ADNI dataset.

## Repository Structure

```
fl-adni-classification/
├── adni_classification/       # Core classification components
│   ├── models/                # ResNet3D, DenseNet3D, SimpleCNN, RosannaCNN, SecureFedCNN
│   ├── datasets/              # NIfTI / .pt tensor dataset implementations
│   ├── utils/                 # Training, losses, visualization utilities
│   └── config/                # Configuration dataclasses
├── scripts/
│   ├── train.py               # Main training script
│   ├── test.py                # Model evaluation script
│   ├── split_by_patient.py    # Patient-wise train/val split
│   └── preprocess_mri.py      # MRI preprocessing pipeline
├── configs/                   # YAML config files
└── docs/
```

---

## Quick Start (Remote Server)

### 1. Clone (sparse checkout — fl-adni-classification only)

```bash
git clone --no-checkout https://github.com/otthree/cognitive-reserve-modeling.git
cd cognitive-reserve-modeling
git sparse-checkout init --cone
git sparse-checkout set fl-adni-classification
git checkout main
cd fl-adni-classification
```

### 2. Install UV and dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 3. Mount data storage

데이터는 `/workspace/pumpkinlab-storage-dhl`에 마운트되어 있어야 합니다:

```
/workspace/pumpkinlab-storage-dhl/
├── 3D_tensors/
│   ├── CN/   *.pt
│   ├── MCI/  *.pt
│   └── AD/   *.pt
└── csv_splits/
    └── all_mri_scan_list.csv
```

### 4. Generate train/val split

```bash
python scripts/split_by_patient.py \
  --csv /workspace/pumpkinlab-storage-dhl/csv_splits/all_mri_scan_list.csv \
  --output_dir /workspace/pumpkinlab-storage-dhl/csv_splits
```

`train.csv`, `val.csv`가 `/workspace/pumpkinlab-storage-dhl/csv_splits/`에 생성됩니다.

### 5. Train

```bash
python scripts/train.py --config configs/tensor_resnet18.yaml
```

### 6. Test

```bash
python scripts/test.py --config configs/tensor_resnet18.yaml \
  --checkpoint outputs/<run_name>/checkpoints/best_model.pth
```

---

## Configuration

`configs/tensor_resnet18.yaml` 주요 항목:

```yaml
data:
  train_csv_path: "/workspace/pumpkinlab-storage-dhl/csv_splits/train.csv"
  val_csv_path: "/workspace/pumpkinlab-storage-dhl/csv_splits/val.csv"
  tensor_dir: "/workspace/pumpkinlab-storage-dhl/3D_tensors"
  dataset_type: "tensor_folder"
  resize_size: [128, 128, 128]
  classification_mode: "CN_MCI_AD"   # CN=0, MCI=1, AD=2

model:
  name: "resnet3d"
  num_classes: 3
  model_depth: 18

training:
  batch_size: 4
  num_epochs: 300
  learning_rate: 0.0001
  mixed_precision: true
```

경로가 다를 경우 `tensor_dir`, `train_csv_path`, `val_csv_path`만 수정하면 됩니다.

### Available models

| `model.name` | Description |
|---|---|
| `resnet3d` | ResNet3D (depth: 10/18/34/50/101/152/200) |
| `densenet3d` | DenseNet3D |
| `simple_cnn` | Lightweight CNN |
| `rosanna_cnn` | RosannaCNN |
| `securefed_cnn` | SecureFedCNN |

---

## Data Format

### Option A: Pre-processed PyTorch Tensors (`.pt`) — recommended

각 `.pt` 파일은 `torch.save()`로 저장된 `(1, 192, 192, 192)` float32 텐서.

마스터 CSV 필수 컬럼:

| 컬럼 | 설명 |
|---|---|
| `pt_index` | `.pt` 파일명 (확장자 제외) |
| `patient_id` | 환자 고유 ID (split 기준) |
| `label` | `CN` / `MCI` / `AD` |

### Option B: Raw NIfTI (`.nii` / `.nii.gz`)

```
<img_dir>/
└── <subject_id>/
    └── .../
        └── ADNI_<subject_id>_..._I<image_id>.nii.gz
```

CSV 필수 컬럼: `image_id`, `subject_id`, `DX`

전처리 파이프라인 (ANTs + FSL 필요):

```bash
python scripts/preprocess_mri.py --input input_folder
```

자세한 내용: [docs/MRI_PREPROCESSING.md](docs/MRI_PREPROCESSING.md)

---

## License

See [LICENSE](LICENSE) for details.
