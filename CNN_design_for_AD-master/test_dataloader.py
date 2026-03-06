import sys
import os
import yaml
import torch
print("Imports successful")

# Load config
with open('config_custom.yaml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

print("Config loaded")
print(f"Config: {cfg}")

# Import dataset
from datasets.adni_3d_custom import ADNI_3D_Custom

print("Dataset class imported")

# Create dataset
dir_to_scans = cfg['data']['dir_to_scans']
dir_to_tsv = cfg['data']['dir_to_tsv']

print(f"Creating dataset...")
print(f"dir_to_scans: {dir_to_scans}")
print(f"dir_to_tsv: {dir_to_tsv}")

try:
    train_dataset = ADNI_3D_Custom(dir_to_scans, dir_to_tsv, mode='Train',
        n_label=cfg['model']['n_label'], percentage_usage=cfg['data']['percentage_usage'], use_custom=True)
    print(f"Train dataset created successfully. Length: {len(train_dataset)}")
except Exception as e:
    print(f"Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to load one sample
print("\nTrying to load first sample...")
try:
    sample = train_dataset[0]
    print(f"Sample loaded successfully!")
    print(f"Sample types: {[type(s) for s in sample]}")
    if sample[0] is not None:
        print(f"Image shape: {sample[0].shape}")
        print(f"Label: {sample[1]}")
    else:
        print("Sample is None - data loading failed")
except Exception as e:
    print(f"Error loading sample: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDataloader test successful!")
