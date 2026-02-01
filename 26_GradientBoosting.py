import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Setup: Create a synthetic dataset
# We're creating a non-linear relationship to show off GB's power
X, y = make_regression(n_samples=500, n_features=1, noise=15, random_state=42)
y = y**2  # Adding some complexity/non-linearity

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the Gradient Boosting Regressor
# n_estimators: Number of trees to build
# learning_rate: How much each tree contributes (shrinks the step)
# max_depth: Limits the complexity of each individual tree
model = GradientBoostingRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=3, 
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Evaluate
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score (Accuracy): {r2:.2f}")