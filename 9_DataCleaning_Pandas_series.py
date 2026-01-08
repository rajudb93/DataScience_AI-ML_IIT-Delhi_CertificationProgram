# ...existing code...
import pandas as pd

data = ['apple', 'banana', 'apple', 'orange', 'banana', 'grape']
s = pd.Series(data)

print("Has duplicates?", s.duplicated().any())
print("Unique (order preserved):", s.drop_duplicates().tolist())
# alternative (also preserves order): print("Unique (Series.unique):", s.unique().tolist())
# ...existing code...