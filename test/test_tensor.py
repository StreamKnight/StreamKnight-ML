from skml.core.tensor import tensor

# Create a 2D tensor
x = tensor([[1, 2, 3], [4, 5, 6]], device = 'gpu')
print(x)
print("Shape:", x.shape)
print("Num dims:", x.ndim)
print("Size:", x.size)
print(x.dtype)
print(x.device)
