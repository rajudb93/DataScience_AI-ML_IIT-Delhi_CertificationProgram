from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
def Mean_Squared_Error(Y_true, Y_pred):
    return np.square(np.subtract(Y_true, Y_pred)).mean()
# Given value
# Y_true = Y (Original values)
Y_true = [1,1,2,2,4]
# Calculated values
Y_pred = [0.6, 1.29, 1.99, 2.69, 3.4]
# Calculation of MSE using sklearn library method
print(f"MSE Using sklearn library: {mean_squared_error(Y_true, Y_pred)}")
# Calculating MSE Using custom function
print(f"MSE Using custom function: {Mean_Squared_Error(Y_true, Y_pred)}")
x = np.arange(len(Y_true))
# Plot
plt.figure()
plt.scatter(x, Y_true, label="Actual values")
plt.plot(x, Y_pred, label="Fitted line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Actual vs Fitted Values")
plt.legend()
plt.show()