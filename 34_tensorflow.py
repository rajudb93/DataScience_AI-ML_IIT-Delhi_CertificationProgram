import numpy as np

# Scalar (0D tensor)
scalar_tensor = np.array(5)
print(f"Scalar Tensor: {scalar_tensor}")

# Vector (1D tensor)
vector_tensor = np.array([1, 2, 3, 4])
print(f"Vector Tensor: {vector_tensor}")

# Matrix (2D tensor)
matrix_tensor = np.array([[1, 2], [3, 4]])
print(f"Matrix Tensors:\n{matrix_tensor}")

# 3D Tensor
tensor_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"Tensors 3D:\n{tensor_3d}")

# Tensor of zeros (2D tensor)
zeros_tensor = np.zeros([3, 3])
print(f"Zero Tensors:\n{zeros_tensor}")

# Tensor of ones (2D tensor)
ones_tensor = np.ones([2, 2])
print(f"Ones Tensors:\n{ones_tensor}")
