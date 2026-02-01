from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load Data (Using the famous Iris flower dataset)
data = load_iris()
X, y = data.data, data.target

# 2. Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Create the Forest (with 100 trees)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 4. Train the model
rf_model.fit(X_train, y_train)

# 5. Check accuracy
accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")