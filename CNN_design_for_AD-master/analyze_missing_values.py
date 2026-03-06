import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('/Users/othree/Cognitive Reserve Modeling/Data/ADNI_master_merged_12-17-2025.csv')

# Count missing values for each column
missing_counts = df.isnull().sum()
total_rows = len(df)

# Calculate missing percentage
missing_pct = (missing_counts / total_rows * 100).sort_values()

# Categorize columns by missing percentage thresholds
thresholds = [3, 5, 10, 30, 50]

print("="*80)
print("ADNI 데이터 결측치 분석 (퍼센트 기준)")
print("="*80)
print(f"전체 행 수: {total_rows:,}")
print(f"전체 열 수: {df.shape[1]:,}")
print()

for threshold in thresholds:
    cols_with_pct = [(col, missing_counts[col], missing_pct[col])
                     for col in missing_pct[missing_pct < threshold].index]

    print(f"\n{'='*80}")
    print(f"결측치가 {threshold}% 미만인 컬럼들 ({len(cols_with_pct)}개):")
    print(f"{'='*80}")

    if len(cols_with_pct) > 0:
        for col, count, pct in cols_with_pct:
            print(f"  - {col:30s}: {count:4d}개 ({pct:5.2f}%)")
    else:
        print("  (없음)")

# Additional summary
print(f"\n{'='*80}")
print("요약 통계:")
print(f"{'='*80}")
for threshold in thresholds:
    count = (missing_pct < threshold).sum()
    print(f"결측치 < {threshold:2d}%: {count:3d}개 컬럼")

# Show columns with specific features used in multimodal model
print(f"\n{'='*80}")
print("Multimodal 모델 Feature 결측치 현황:")
print(f"{'='*80}")
features_to_check = ['Age', 'Sex', 'PTEDUCAT', 'APOE4', 'Weight', 'Height',
                     'Pulse', 'Respiration', 'sys', 'did']

for feat in features_to_check:
    if feat in df.columns:
        count = missing_counts[feat]
        pct = missing_pct[feat]
        print(f"  - {feat:15s}: {count:4d}개 ({pct:5.2f}%)")
    else:
        # Try to find similar column names
        similar = [col for col in df.columns if feat.lower() in col.lower()]
        if similar:
            print(f"  - {feat:15s}: 컬럼 없음 (유사: {', '.join(similar[:3])})")
        else:
            print(f"  - {feat:15s}: 컬럼 없음")
