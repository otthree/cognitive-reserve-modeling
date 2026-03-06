# Custom Training with 50 Selected Subjects

This guide explains how to train the CNN model with our custom 50-subject dataset.

## Setup Completed

1. **Data Split**:
   - Train: 40 subjects
   - Validation: 10 subjects (CN: 3, MCI: 4, Dementia: 3)

2. **Files Created**:
   - `datasets/adni_3d_custom.py`: Custom data loader for our data structure
   - `datasets/files/Train_diagnosis_ADNI_custom.tsv`: Training metadata
   - `datasets/files/Val_diagnosis_ADNI_custom.tsv`: Validation metadata
   - `config_custom.yaml`: Configuration file for custom dataset
   - `main_custom.py`: Modified main script using custom data loader

## Before Training

### 1. Check Image Sizes
Run the notebook cell in `ADNI_matching.ipynb` to verify all images are >= 96x96x96.
If images are too small, they need to be resized or padded.

### 2. Install Dependencies
```bash
cd "/Users/othree/Cognitive Reserve Modeling/CNN_design_for_AD-master"
pip install torch torchvision nibabel scipy pandas pyyaml progress matplotlib numpy
```

## Training

### Basic Training
```bash
cd "/Users/othree/Cognitive Reserve Modeling/CNN_design_for_AD-master"
python main_custom.py
```

### Advanced Options
```bash
# With specific expansion factor
python main_custom.py --expansion 8

# With partial data usage
python main_custom.py --percentage_usage 0.8

# With custom config
python main_custom.py --config config_custom
```

## Model Architecture

- **Type**: 3D CNN
- **Classes**: 3 (CN, MCI, AD)
- **Input**: 96x96x96 MRI volumes
- **Expansion factor**: 8 (default)
- **Normalization**: Instance Normalization
- **Optimizer**: SGD (lr=0.01)
- **Epochs**: 50

## Output

Trained models will be saved to:
```
./saved_model/custom_50subjects/
```

Best models saved:
- `_model_best.pth.tar`: Best validation accuracy
- `_model_low_loss.pth.tar`: Lowest validation loss
- `_model_best_micro.pth.tar`: Best micro AUC
- `_model_best_macro.pth.tar`: Best macro AUC

## Troubleshooting

### Issue: Image size too small
**Solution**: Images must be >= 96x96x96. If smaller, modify `centerCrop` and `randomCrop` sizes in `adni_3d_custom.py`

### Issue: CUDA out of memory
**Solution**: Reduce batch_size in `config_custom.yaml`:
```yaml
data:
  batch_size: 2  # reduce from 4
  val_batch_size: 1  # reduce from 2
```

### Issue: File not found
**Solution**: Verify TSV files contain correct filepaths. Run the TSV generation cell in `ADNI_matching.ipynb` again.

## Data Structure

Our custom data loader expects:
- TSV files with `filepath` column pointing directly to .nii.gz files
- Images in cropped format (already preprocessed)
- No specific directory structure required (uses direct file paths)

## Next Steps

After training:
1. Evaluate model performance on validation set
2. Analyze confusion matrix and AUC curves
3. Visualize learned features
4. Compare with baseline models
