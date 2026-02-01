import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Create Synthetic Semiconductor Data
# Features: [Temperature, Voltage], Label: 0 (Pass), 1 (Fail)
data = {
    'Temp': [70, 72, 68, 85, 90, 71, 88, 67, 92, 75],
    'Voltage': [1.2, 1.1, 1.3, 0.8, 0.7, 1.2, 0.9, 1.4, 0.6, 1.1],
    'Status': [0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 2. Separate Features and Target
X = df[['Temp', 'Voltage']]
y = df['Status']

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (KNN relies on distance, so scale is vital!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Initialize and Train KNN (using K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 6. Make a Prediction for a New Batch
# New Batch Sensor Data: Temp=89, Voltage=0.75
new_batch = scaler.transform([[89, 0.75]])
prediction = knn.predict(new_batch)

print(f"Prediction for new batch (0=Pass, 1=Fail): {prediction[0]}")