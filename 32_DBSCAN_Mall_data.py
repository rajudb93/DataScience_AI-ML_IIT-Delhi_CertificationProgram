import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# -------------------------------------------------
# 1. Load dataset from reliable raw URL
# -------------------------------------------------
url = "https://gist.githubusercontent.com/ryanorsinger/cc276eea59e8295204d1f581c8da509f/raw/mall_customers.csv"
df = pd.read_csv(url)

# -------------------------------------------------
# 2. Select features: Annual Income and Spending Score
# -------------------------------------------------
X = df[['annual_income', 'spending_score']].values

# -------------------------------------------------
# 3. Scale features for clustering
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 4. KMeans clustering
# -------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# 5. DBSCAN clustering
# -------------------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# -------------------------------------------------
# 6. Visualization
# -------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# KMeans plot
axs[0].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='Set1', s=50)
axs[0].set_title("KMeans Clustering")
axs[0].set_xlabel("Annual Income (k$)")
axs[0].set_ylabel("Spending Score")

# DBSCAN plot
axs[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='Set2', s=50)
axs[1].set_title("DBSCAN Clustering")
axs[1].set_xlabel("Annual Income (k$)")
axs[1].set_ylabel("Spending Score")

plt.tight_layout()
plt.show()

# -------------------------------------------------
# 7. Output cluster summaries
# -------------------------------------------------
print("KMeans cluster counts:\n")
print(pd.Series(kmeans_labels).value_counts().sort_index())

print("\nDBSCAN cluster counts (including noise -1):\n")
print(pd.Series(dbscan_labels).value_counts().sort_index())
