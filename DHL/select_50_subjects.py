import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('/Users/othree/Cognitive Reserve Modeling/Data/ADNI_master_merged_12-17-2025.csv', low_memory=False)

np.random.seed(42)

# Select random sessions from unique subjects for each diagnosis group
selected_data = []

# Dementia (AD) - 17 subjects
ad_data = df[df['DX'] == 'Dementia']
ad_subjects = ad_data['Subject'].unique()
selected_ad_subjects = np.random.choice(ad_subjects, size=17, replace=False)

for subject in selected_ad_subjects:
    subject_ad_data = ad_data[ad_data['Subject'] == subject]
    random_session = subject_ad_data.sample(n=1)
    selected_data.append(random_session)

# MCI - 17 subjects
mci_data = df[df['DX'] == 'MCI']
mci_subjects = mci_data['Subject'].unique()
selected_mci_subjects = np.random.choice(mci_subjects, size=17, replace=False)

for subject in selected_mci_subjects:
    subject_mci_data = mci_data[mci_data['Subject'] == subject]
    random_session = subject_mci_data.sample(n=1)
    selected_data.append(random_session)

# CN - 16 subjects
cn_data = df[df['DX'] == 'CN']
cn_subjects = cn_data['Subject'].unique()
selected_cn_subjects = np.random.choice(cn_subjects, size=16, replace=False)

for subject in selected_cn_subjects:
    subject_cn_data = cn_data[cn_data['Subject'] == subject]
    random_session = subject_cn_data.sample(n=1)
    selected_data.append(random_session)

# Create final dataframe
final_df = pd.concat(selected_data, ignore_index=True)

print(f"Total selected rows: {len(final_df)}")
print(f"Unique subjects: {final_df['Subject'].nunique()}")
print("\nDiagnosis distribution:")
print(final_df['DX'].value_counts())

# Save to CSV
output_path = '/Users/othree/Cognitive Reserve Modeling/Data/ADNI_selected_50subjects.csv'
final_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
