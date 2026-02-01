import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reduce to 2D with PCA for visualization
# This is necessary because we can't plot 30 dimensions!
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# 5. Initialize and Train KNN classifier
# Moving this UP so the model exists before we predict
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 6. Prediction and accuracy
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (PCA + KNN on Breast Cancer): {accuracy:.2f}")

# 7. Plot decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
    knn,
    X_train,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.6,
    xlabel="PCA Component 1",
    ylabel="PCA Component 2"
)

# 8. Scatter plot of training data on top of boundary
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train,
    cmap=plt.cm.coolwarm,
    edgecolors='k'
)

plt.title("KNN (k=5) Decision Boundary on Breast Cancer PCA Data")
plt.show()