import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# -----------------------------
# 1. Load Diabetes dataset
# -----------------------------
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

# Put into a DataFrame for convenience
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# We'll work with one feature: 'bmi'
feature = "bmi"
data = df[feature]

# -----------------------------
# 2. Compute IQR and bounds
# -----------------------------
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Feature: {feature}")
print(f"Q1: {Q1:.4f}")
print(f"Q3: {Q3:.4f}")
print(f"IQR: {IQR:.4f}")
print(f"Lower bound: {lower_bound:.4f}")
print(f"Upper bound: {upper_bound:.4f}")

# -----------------------------
# 3. Identify outliers
# -----------------------------
outlier_mask = (data < lower_bound) | (data > upper_bound)
df["is_outlier"] = outlier_mask

print(f"\nTotal samples: {len(df)}")
print(f"Outliers detected: {df['is_outlier'].sum()}")
print(f"Non-outliers: {len(df) - df['is_outlier'].sum()}")

# -----------------------------
# 4. Plot 1: Boxplot of BMI
# -----------------------------
plt.figure(figsize=(6, 5))
plt.boxplot(data, vert=True, showfliers=True)
plt.title(f"Boxplot of '{feature}' with IQR-based Outliers")
plt.ylabel(feature)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Plot 2: BMI vs Target (highlighting outliers)
# -----------------------------
plt.figure(figsize=(7, 5))

# Non-outliers
plt.scatter(df.loc[~df["is_outlier"], feature],
            df.loc[~df["is_outlier"], "target"],
            label="Normal points",
            alpha=0.7)

# Outliers
plt.scatter(df.loc[df["is_outlier"], feature],
            df.loc[df["is_outlier"], "target"],
            label="Outliers",
            edgecolor="red",
            facecolor="none",
            s=80,
            linewidth=1.5)

plt.title(f"'{feature}' vs Target (Outliers via IQR)")
plt.xlabel(feature)
plt.ylabel("Disease progression (target)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. (Optional) Create a cleaned dataset
# -----------------------------
df_clean = df[~df["is_outlier"]].copy()
print("\nShape before removing outliers:", df.shape)
print("Shape after removing outliers:", df_clean.shape)
