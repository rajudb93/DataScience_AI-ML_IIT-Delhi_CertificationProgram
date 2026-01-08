import numpy as np
 # create a list named list1
list1 = [2,4,6,8]
 # create numpy array using list1
array1 = np.array(list1)
print(array1)
 # Output - [2,4,6,8
print("######################################################")
array1 = np.zeros(4)
print(array1)
 # Output - [0. 0. 0. 0.]
 
array1 = np.random.rand(5)
print(array1)

array1 = np.arange(5)
print("Using np.arange(5):", array1)
 # create an array with values from 1 to 8 with a step 2
array2 = np.arange(1,9,2)
print("Using np.arange(1,9,2):", array2)
''' Output - Using np.arange(5): [0 1 2 3 4]
 Using np.arange(1,9,2): [1 3 5 7]
'''

array1=np.array([[1,2,3,4],[5,6,7,8]])
print(array1)

 # create a 3D arrays with 2 "slices", each of 3 rows and 4 columns
array1 = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], 
                    [9, 10, 11, 12]],[[13, 14, 15, 16], 
                    [17, 18, 19, 20], [21, 22, 23, 24]]])
print(array1)
print(array1.ndim)

print("######################################################")

array1 = np.array([1,3,5,7,2,4,6,8])
 # reshaping a 1D array into a 2D array with 2 rows and 4 columns
result = np.reshape(array1, (2,4))
print(result)
''' Output - [[1 3 5 7]
 [2 4 6 8]]
'''

array1 = np.array([9, 12, 21])
array2 = np.array([21, 12, 9])
 # use of greater_equal()
result = np.greater_equal(array1, array2)
print("Using greater_equal():",result)

