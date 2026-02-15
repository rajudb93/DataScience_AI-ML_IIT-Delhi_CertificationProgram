import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def fit_gradient_boosting_manual(X, y, stumps, learning_rate=0.1):
    """
    Fit a manual gradient boosting regressor using provided stumps.
    stumps: list of tuples (feature, threshold, left_value, right_value)
    Returns predictions at each stage and residuals.
    """
    n = len(y)
    F = np.full(n, np.mean(y), dtype=np.float64)
    preds = [F.copy()]
    residuals = [y - F]
    for stump in stumps:
        feature, threshold, left_value, right_value = stump
        h = np.where(X[feature] <= threshold, left_value, right_value)
        F = F + learning_rate * h
        preds.append(F.copy())
        residuals.append(y - F)
    return preds, residuals

def predict_gradient_boosting_manual(X, stumps, learning_rate=0.1, F0=0):
    """
    Predict using manual gradient boosting with provided stumps.
    """
    F = np.full(X.shape[0], F0, dtype=np.float64)
    for stump in stumps:
        feature, threshold, left_value, right_value = stump
        h = np.where(X[feature] <= threshold, left_value, right_value)
        F = F + learning_rate * h
    return F

def main():
    # Example dataset
    data = pd.DataFrame({
        'Age': [25, 30, 35, 40, 45, 50],
        'Income': [40, 60, 50, 80, 100, 90],
        'Price': [200, 250, 240, 300, 360, 330]
    })
    X = data[['Age', 'Income']]
    y = data['Price'].values

    # Define stumps: (feature, threshold, left_value, right_value)
    # feature: column name
    stumps = [
        ('Income', 60, -60, 80),
        ('Age', 40, -6, 42)
    ]
    learning_rate = 0.1

    preds, residuals = fit_gradient_boosting_manual(X, y, stumps, learning_rate)
    data['F0'] = preds[0]
    data['r1'] = residuals[0]
    data['F1'] = preds[1]
    data['r2'] = residuals[1]
    data['F2'] = preds[2]
    data['FinalResidual'] = residuals[2]

    print("\nData with predictions and residuals:\n", data)

    # Predict for new data
    new_data = pd.DataFrame({'Age': [38], 'Income': [70]})
    F0 = np.mean(y)
    pred_new = predict_gradient_boosting_manual(new_data, stumps, learning_rate, F0)
    print(f"\nPredicted Price for Age=38, Income=70: {pred_new[0]:.2f}")

if __name__ == "__main__":
    main()
