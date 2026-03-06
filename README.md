# Cognitive Reserve Modeling

Research code for modeling cognitive reserve (CR) using ADNI MRI data.

## Repository Structure

### Original Work

| Directory | Description |
|-----------|-------------|
| `DHL/` | Main CR modeling experiments — EDA, preprocessing, clustering, and model development using ADNI data |
| `divnet/` | Custom implementation of DivNet from scratch based on the paper |
| `3D Tensor Creation_Custom.py` | Custom script for converting 3D MRI images to tensors |

### Baseline Model Testing

The following are existing public models adapted and tested on our dataset. Each directory retains its original README and attribution.

| Directory | Original Repository | Description |
|-----------|---------------------|-------------|
| `CNN_design_for_AD-master/` | [Marcela Paz Contreras Osorio et al.](https://github.com/marcottt/CNN_design_for_AD) | CNN-based AD classification |
| `Alzheimer-Detection-with-3D-HCCT-main/` | [Alzheimer Detection with 3D HCCT](https://github.com/bhanML/Alzheimer-Detection-with-3D-HCCT) | Hybrid CNN + Compact Transformer for 3D MRI |
| `DiaMond/` | [DiaMond](https://github.com/ai-med/DiaMond) | Diagnosis of Alzheimer's disease with multi-modal data |
| `AD-DL/` | [AD-DL](https://github.com/aramis-lab/AD-DL) | Deep learning for AD classification (ClinicaDL) |
| `M3AD/` | [M3AD](https://github.com/shu-hai/M3AD) | Multi-modal multi-task AD diagnosis |
| `fl-adni-classification/` | — | Federated learning for AD classification, adapted for custom dataset |

## Data

All experiments use data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/).

## Acknowledgements

This project builds upon the work of the original authors of each baseline model. Please refer to the individual `README` and `LICENSE` files within each subdirectory for full attribution and licensing terms.