import numpy as np
from skml.core.tensor import tensor

# --------------------------
# Zeros
# --------------------------
def zeros(shape, dtype=np.float32) -> tensor:
    """
    Create a tensor filled with zeros
    """
    return tensor(np.zeros(shape, dtype=dtype))

# --------------------------
# Ones
# --------------------------
def ones(shape, dtype=np.float32) -> tensor:
    """
    Create a tensor filled with ones
    """
    return tensor(np.ones(shape, dtype=dtype))

# --------------------------
# Random normal
# --------------------------
def randn(shape, mean=0.0, std=1.0, dtype=np.float32) -> tensor:
    """
    Create a tensor with random values from normal distribution
    """
    return tensor(np.random.normal(loc=mean, scale=std, size=shape).astype(dtype))

# --------------------------
# Random uniform
# --------------------------
def rand(shape, low=0.0, high=1.0, dtype=np.float32) -> tensor:
    """
    Create a tensor with random values from uniform distribution
    """
    return tensor(np.random.uniform(low=low, high=high, size=shape).astype(dtype))

# --------------------------
# Arange
# --------------------------
def arange(start, stop=None, step=1, dtype=np.float32) -> tensor:
    """
    Create a tensor with evenly spaced values within a range
    """
    if stop is None:
        stop = start
        start = 0
    return tensor(np.arange(start, stop, step, dtype=dtype))
