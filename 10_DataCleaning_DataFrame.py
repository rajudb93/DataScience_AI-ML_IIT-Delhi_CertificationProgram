# ...existing code...
import pandas as pd

data = ['apple', 'banana', 'apple', 'orange', 'banana', 'grape']
s = pd.Series(data)

print("Has duplicates?", s.duplicated().any())
print("Unique (order preserved):", s.drop_duplicates().tolist())
# alternative (also preserves order): print("Unique (Series.unique):", s.unique().tolist())

# --- DataFrame example using dictionary data ---
data_dict = {
    'Name':  ['Alice', 'Bob', 'Alice', 'David', 'Bob'],
    'Age':   [25,     30,    25,     40,      30],
    'City':  ['NY',   'Paris','NY',  'Tokyo', 'Paris']
}
df = pd.DataFrame(data_dict)

print("\nDataFrame:\n", df)

# check if there are any duplicate rows
print("\nHas duplicate rows?", df.duplicated().any())

# show duplicate rows (all except first occurrences)
print("\nDuplicate rows (except first occurrences):\n", df[df.duplicated()])

# indices of duplicate rows
print("\nDuplicate row indices:", df[df.duplicated()].index.tolist())

# unique rows keeping first occurrence
print("\nUnique rows (drop_duplicates):\n", df.drop_duplicates().reset_index(drop=True))

# example: consider duplicates only based on 'Name' and 'Age' (ignore City)
print("\nHas duplicates based on ['Name','Age']?", df.duplicated(subset=['Name','Age']).any())
print("Unique rows by ['Name','Age']:\n", df.drop_duplicates(subset=['Name','Age']).reset_index(drop=True))
# ...existing code...