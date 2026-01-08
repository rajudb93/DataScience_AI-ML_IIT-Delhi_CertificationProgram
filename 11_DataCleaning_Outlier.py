from sklearn.datasets import load_diabetes
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

diab = load_diabetes()
df_diab = pd.DataFrame(diab.data, columns=diab.feature_names)
df_diab['target'] = diab.target

print("\nDiabetes DataFrame shape:", df_diab.shape)
print(df_diab.head())

# duplicates check
print("\nHas duplicate rows in diabetes data?", df_diab.duplicated().any())

# IQR-based outlier removal on 'target'
col = 'target'
Q1 = df_diab[col].quantile(0.25)
Q3 = df_diab[col].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
print(f"\nIQR bounds for '{col}': lower={lower:.2f}, upper={upper:.2f}")

plt.figure(figsize=(6, 2))
sns.boxplot(x=df_diab[col])
plt.title(f"{col} — before IQR filtering")
plt.show()

df_diab_iqr = df_diab[(df_diab[col] >= lower) & (df_diab[col] <= upper)].reset_index(drop=True)
print(f"Rows before: {len(df_diab)}, after IQR filtering: {len(df_diab_iqr)}")

plt.figure(figsize=(6, 2))
sns.boxplot(x=df_diab_iqr[col])
plt.title(f"{col} — after IQR filtering")
plt.show()

# IsolationForest for multivariate outlier detection (use a few features)
features = ['bmi', 'bp', 's1']
X = df_diab[features].to_numpy()
iso = IsolationForest(contamination=0.05, random_state=42)
labels = iso.fit_predict(X)  # -1 outlier, 1 inlier
df_diab_if = df_diab[labels == 1].reset_index(drop=True)

print(f"Rows before: {len(df_diab)}, after IsolationForest filtering: {len(df_diab_if)}")

plt.figure(figsize=(8, 3))
sns.boxplot(data=df_diab_if[features])
plt.title("Selected features after IsolationForest (boxplots)")
plt.show()
