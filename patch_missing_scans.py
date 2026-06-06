"""
Patch 21 missing scans that were skipped during original .pt conversion.

Root cause: 2005 ADNI-1 scan filenames lack the S##### session ID,
so regex (S[digits]+)_(I[digits]+) failed to match -> 21 scans never converted.

This script:
  1. Identifies the 21 missing scans (master CSV vs scan CSV)
  2. Finds each .nii.gz in the mounted GCS folder by I{image_id} pattern
  3. Processes: pad 256³ → zoom 192³ → [1,192,192,192] float32 tensor
  4. Saves .pt directly into /workspace/.../3D_tensors/{CN|MCI|AD}/{pt_index}.pt
  5. Appends 21 new rows to csv_splits_all_mri_scan_list.csv

Usage (on the server):
    python patch_missing_scans.py
"""

import os
import subprocess
import sys

# Auto-install missing dependencies
def _ensure(pkg, import_name=None):
    import importlib
    try:
        importlib.import_module(import_name or pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

_ensure('nibabel')
_ensure('scipy')
_ensure('tqdm')

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MASTER_CSV = '/workspace/pumpkinlab-storage-dhl/tabular/ADNI_master_merged_12-17-2025.csv'
SCAN_CSV   = '/workspace/cognitive-reserve-modeling/divnet/csv_splits_all_mri_scan_list.csv'

# Where the source .nii.gz files live (original 21 were never deleted because they were never matched)
GCS_NII_DIR = 'gs://pumpkinlab-storage-dhl/all_brain_mni152_1mm_02_04_2026'

# Where existing .pt files live — new ones go into {PT_DIR}/{CN|MCI|AD}/{pt_index}.pt
PT_DIR     = '/workspace/pumpkinlab-storage-dhl/3D_tensors'

WORK_DIR   = '/tmp/patch_missing_scans'

IMG_SIZE   = 192
DEFAULT_SPLIT_FOR_NEW_PATIENT = 'train'   # only used for 011_S_0002 (no existing sessions)

# ---------------------------------------------------------------------------
# Step 1: Find missing scans
# ---------------------------------------------------------------------------

def find_missing():
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    scan   = pd.read_csv(SCAN_CSV)

    scan_ids     = set(scan['image_id'].astype(str))
    missing_mask = ~master['Image Data ID'].astype(str).isin(scan_ids)
    missing_df   = master[missing_mask & master['DX'].notna()].copy()

    pid_split = scan.groupby('patient_id')['split'].first().to_dict()
    max_pt    = scan['pt_index'].max()

    rows = []
    for i, (_, row) in enumerate(missing_df.iterrows()):
        label = 'AD' if row['DX'] == 'Dementia' else row['DX']
        pid   = row['Subject']
        rows.append({
            'pt_index':   max_pt + 1 + i,
            'patient_id': pid,
            'image_id':   str(int(row['Image Data ID'])),
            'label':      label,
            'split':      pid_split.get(pid, DEFAULT_SPLIT_FOR_NEW_PATIENT),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: Find .nii.gz in GCS by I{image_id} wildcard, then download
# ---------------------------------------------------------------------------

def find_and_download_nii(image_id):
    """Search GCS for file matching *I{image_id}*, download to /tmp, return local path."""
    result = subprocess.run(
        ['gsutil', 'ls', f'{GCS_NII_DIR}/*I{image_id}*'],
        capture_output=True, text=True
    )
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return None
    if len(lines) > 1:
        print(f"  [WARN] Multiple matches for I{image_id}: {lines}")
    gcs_uri = lines[0]

    os.makedirs(WORK_DIR, exist_ok=True)
    local_path = os.path.join(WORK_DIR, os.path.basename(gcs_uri))
    subprocess.run(['gsutil', 'cp', gcs_uri, local_path], check=True,
                   capture_output=True)
    return local_path, gcs_uri


# ---------------------------------------------------------------------------
# Step 3: Process .nii.gz → tensor  (same logic as original script)
# ---------------------------------------------------------------------------

def process_nii(nii_path):
    img  = nib.as_closest_canonical(nib.load(nii_path))
    data = img.get_fdata().astype(np.float32)
    xd, yd, zd = data.shape

    # Pad to 256³
    data = np.pad(
        data,
        [((256 - xd) // 2, (256 - xd) // 2),
         ((256 - yd) // 2, (256 - yd) // 2),
         ((256 - zd) // 2, (256 - zd) // 2)],
        mode='constant', constant_values=0,
    )

    # Zoom 256³ → 192³
    f    = IMG_SIZE / 256.0
    data = zoom(data, (f, f, f), order=1)

    return torch.from_numpy(data).unsqueeze(0)   # [1, 192, 192, 192]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 1: Identifying missing scans")
    print("=" * 60)
    missing_df = find_missing()
    print(missing_df[['pt_index', 'patient_id', 'image_id', 'label', 'split']].to_string())
    print(f"\nTotal: {len(missing_df)}")

    # Label distribution of missing scans
    print("\nLabel breakdown:")
    print(missing_df['label'].value_counts().to_string())

    success   = []
    not_found = []
    new_csv_rows = []

    print("\n" + "=" * 60)
    print("Steps 2–4: Find → Process → Save")
    print("=" * 60)

    for _, row in tqdm(missing_df.iterrows(), total=len(missing_df)):
        iid      = row['image_id']
        label    = row['label']
        pt_idx   = int(row['pt_index'])

        # Find in GCS and download
        result = find_and_download_nii(iid)
        if result is None:
            print(f"\n  [SKIP] I{iid} not found in GCS")
            not_found.append(iid)
            continue
        nii_path, gcs_uri = result

        # Output path: {PT_DIR}/{label}/{pt_index}.pt
        out_dir = os.path.join(PT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)
        out_pt  = os.path.join(out_dir, f'{pt_idx}.pt')

        # Process & save
        tensor = process_nii(nii_path)
        torch.save(tensor, out_pt)

        # Clean up downloaded nii
        os.remove(nii_path)

        new_csv_rows.append({
            'pt_index':   pt_idx,
            'image_path': gcs_uri,
            'patient_id': row['patient_id'],
            'image_id':   iid,
            'label':      label,
            'split':      row['split'],
        })
        success.append(iid)

    # Step 5: Update scan CSV
    print("\n" + "=" * 60)
    print("Step 5: Updating scan CSV")
    print("=" * 60)
    if new_csv_rows:
        scan    = pd.read_csv(SCAN_CSV)
        updated = pd.concat([scan, pd.DataFrame(new_csv_rows)], ignore_index=True)
        updated.to_csv(SCAN_CSV, index=False)
        print(f"CSV: {len(scan)} → {len(updated)} rows (+{len(new_csv_rows)})")

        print("\nNew entries by label & split:")
        new_df = pd.DataFrame(new_csv_rows)
        print(new_df.groupby(['label', 'split']).size().unstack(fill_value=0).to_string())

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Saved to PT_DIR:  {len(success)}")
    print(f"  Not found (nii):  {len(not_found)}")
    if not_found:
        print(f"  Missing IDs: {not_found}")
        print("  → These files may have been lost during original preprocessing.")


if __name__ == '__main__':
    main()
