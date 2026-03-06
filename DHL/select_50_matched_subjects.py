import pandas as pd
import numpy as np

# Load the matched data
df = pd.read_csv('/Users/othree/Cognitive Reserve Modeling/Data/ADNI_selected_50subjects_matched.csv')

print(f"Total matched sessions: {df['matched_filepath'].notna().sum()}")
print(f"Total sessions: {len(df)}")

# Filter only matched sessions
matched_df = df[df['matched_filepath'].notna()].copy()

print(f"\nMatched sessions by diagnosis:")
print(matched_df['DX'].value_counts())

# Select 50 unique subjects: AD 17, MCI 17, CN 16
np.random.seed(42)

selected_data = []
selected_subjects = set()

# Dementia (AD) - 17 subjects
ad_data = matched_df[matched_df['DX'] == 'Dementia']
ad_subjects = ad_data['Subject'].unique()
print(f"\nAvailable AD subjects: {len(ad_subjects)}")

if len(ad_subjects) >= 17:
    selected_ad_subjects = np.random.choice(ad_subjects, size=17, replace=False)
    for subject in selected_ad_subjects:
        subject_data = ad_data[ad_data['Subject'] == subject]
        random_session = subject_data.sample(n=1)
        selected_data.append(random_session)
        selected_subjects.add(subject)
else:
    print(f"Warning: Not enough AD subjects. Only {len(ad_subjects)} available.")
    for subject in ad_subjects:
        subject_data = ad_data[ad_data['Subject'] == subject]
        random_session = subject_data.sample(n=1)
        selected_data.append(random_session)
        selected_subjects.add(subject)

# MCI - 17 subjects (excluding already selected subjects)
mci_data = matched_df[matched_df['DX'] == 'MCI']
mci_subjects = [s for s in mci_data['Subject'].unique() if s not in selected_subjects]
print(f"Available MCI subjects (excluding already selected): {len(mci_subjects)}")

if len(mci_subjects) >= 17:
    selected_mci_subjects = np.random.choice(mci_subjects, size=17, replace=False)
    for subject in selected_mci_subjects:
        subject_data = mci_data[mci_data['Subject'] == subject]
        random_session = subject_data.sample(n=1)
        selected_data.append(random_session)
        selected_subjects.add(subject)
else:
    print(f"Warning: Not enough MCI subjects. Only {len(mci_subjects)} available.")
    for subject in mci_subjects:
        subject_data = mci_data[mci_data['Subject'] == subject]
        random_session = subject_data.sample(n=1)
        selected_data.append(random_session)
        selected_subjects.add(subject)

# CN - 16 subjects (excluding already selected subjects)
cn_data = matched_df[matched_df['DX'] == 'CN']
cn_subjects = [s for s in cn_data['Subject'].unique() if s not in selected_subjects]
print(f"Available CN subjects (excluding already selected): {len(cn_subjects)}")

if len(cn_subjects) >= 16:
    selected_cn_subjects = np.random.choice(cn_subjects, size=16, replace=False)
    for subject in selected_cn_subjects:
        subject_data = cn_data[cn_data['Subject'] == subject]
        random_session = subject_data.sample(n=1)
        selected_data.append(random_session)
        selected_subjects.add(subject)
else:
    print(f"Warning: Not enough CN subjects. Only {len(cn_subjects)} available.")
    for subject in cn_subjects:
        subject_data = cn_data[cn_data['Subject'] == subject]
        random_session = subject_data.sample(n=1)
        selected_data.append(random_session)
        selected_subjects.add(subject)

# Create final dataframe
final_df = pd.concat(selected_data, ignore_index=True)

print(f"\nFinal selection:")
print(f"Total rows: {len(final_df)}")
print(f"Unique subjects: {final_df['Subject'].nunique()}")
print(f"\nDiagnosis distribution:")
print(final_df['DX'].value_counts())

# Verify all have matched filepaths
print(f"\nAll matched: {final_df['matched_filepath'].notna().all()}")

# Save to CSV
output_path = '/Users/othree/Cognitive Reserve Modeling/Data/ADNI_final_50subjects_matched.csv'
final_df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")
