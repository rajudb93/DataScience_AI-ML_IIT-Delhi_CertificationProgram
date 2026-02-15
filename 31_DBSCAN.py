import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN, KMeans

# Generate non-linear data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=kmeans_labels)
plt.title("K-Means Clustering")

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=dbscan_labels)
plt.title("DBSCAN Clustering")

plt.show()
