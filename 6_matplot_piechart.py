import matplotlib.pyplot as plt

cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
data = [23, 17, 35, 29, 12, 41]

plt.figure(figsize=(10,7))
plt.pie(data, labels=cars, autopct='%1.1f%%')   # <- added autopct
plt.show()