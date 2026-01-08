import pandas as pd
 # create a list named data
data = [2,4,5,8]
 # create Pandas array using data
array1 = pd.array(data)
print(array1)
''' Output - <IntegerArray>
 [2, 4, 5, 8]
 Length: 4, dtype: Int64
'''

 # creating a pandas.array of integer values
int_array = pd.array([1,2,3,4,5], dtype='int')
print(int_array)

#Pandas series
array1=pd.array([10,20,30,40])
series1 = pd.Series(array1)
print(series1)

print('######################################################')
# Small DataFrame to demonstrate head, tail and info
df = pd.DataFrame({
	'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
	'age': [25, 30, 35, 40, 22],
	'score': [85.0, 92.5, 88.0, 91.0, 79.5]
})
print('\nSample DataFrame:\n', df)

print('\nDataFrame.head(3):')
print(df.head(3))

print('\nDataFrame.tail(2):')
print(df.tail(2))

print('\nDataFrame.info() output:')
# df.info() prints directly to stdout and returns None; show it as a separate call
df.info()


print('\n######################################################')
print('\n=== Adding new columns examples ===')

# 1) Add a column by assignment (broadcast a scalar or array)
df['passed'] = df['score'] >= 90
print('\nAfter df["passed"] = df["score"] >= 90:\n', df)

# 2) Use assign (returns a new DataFrame; can chain)
df2 = df.assign(score_scaled=lambda x: x['score'] / 100)
print('\nUsing assign to add score_scaled (df2):\n', df2)

# 3) Add column with np.where (conditional values)
df['grade'] = np.where(df['score'] >= 90, 'A', np.where(df['score'] >= 80, 'B', 'C'))
print('\nAfter adding grade with np.where:\n', df)

# 4) Add a column using map/dict (useful for lookups)
bonus_map = {'Alice': 5, 'Bob': 10, 'Charlie': 0, 'Diana': 7, 'Eve': 3}
df['bonus'] = df['name'].map(bonus_map)
print('\nAfter mapping bonus from dict:\n', df)

# 5) Insert a column at a specific position
df.insert(2, 'age_plus_1', df['age'] + 1)
print('\nAfter insert at pos 2 (age_plus_1):\n', df)

# 6) Create a datetime-derived column
df['now'] = pd.to_datetime('2025-11-02')
df['month'] = df['now'].dt.month
print('\nAfter adding datetime-derived columns now and month:\n', df[['name', 'now', 'month']].head())

# 7) Add a categorical column (memory & performance hint)
df['size'] = pd.Categorical(pd.cut(df['score'], bins=[0, 80, 90, 100], labels=['low', 'medium', 'high']))
print('\nAfter adding categorical size column:\n', df[['name', 'score', 'size']])

# 8) Add column from another DataFrame via merge/join example
other = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], 'country': ['IN', 'US', 'UK', 'AU', 'IN']})
df = df.merge(other, on='name', how='left')
print('\nAfter merging country from other DataFrame:\n', df[['name', 'country']])

print('\n######################################################')
# Examples: access rows and columns using .loc
data = {
	'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
	'Age': [25, 32, 18, 47, 33],
	'City': ['New York', 'Paris', 'London', 'Tokyo', 'Sydney']
}
df2 = pd.DataFrame(data)

# access a single row (by label/index)
single_row = df2.loc[2]
print('\nSingle row:')
print(single_row)
# Expected output (conceptual):
# Name    Charlie
# Age          18
# City     London
# Name: 2, dtype: object

print()

# access rows 0, 3 and 4
row_list = df2.loc[[0, 3, 4]]
print('List of Rows:')
print(row_list)
# Expected output (conceptual): rows with indices 0,3,4 shown as a DataFrame

print()

# access a list of columns
column_list = df2.loc[:, ['Name', 'Age']]
print('List of Columns:')
print(column_list)





