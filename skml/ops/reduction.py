import numpy as np
from skml.core.tensor import tensor

# --------------------------
# Sum
# --------------------------
def sum(a: tensor, axis=None, keepdims=False) -> tensor:
    """
    Sum of tensor elements along given axis
    """
    return tensor(np.sum(a.data, axis=axis, keepdims=keepdims))

# --------------------------
# Mean
# --------------------------
def mean(a: tensor, axis=None, keepdims=False) -> tensor:
    """
    Mean of tensor elements along given axis
    """
    return tensor(np.mean(a.data, axis=axis, keepdims=keepdims))

# --------------------------
# Max
# --------------------------
def max(a: tensor, axis=None, keepdims=False) -> tensor:
    """
    Maximum value of tensor along axis
    """
    return tensor(np.max(a.data, axis=axis, keepdims=keepdims))

# --------------------------
# Min
# --------------------------
def min(a: tensor, axis=None, keepdims=False) -> tensor:
    """
    Minimum value of tensor along axis
    """
    return tensor(np.min(a.data, axis=axis, keepdims=keepdims))

# --------------------------
# Norm
# --------------------------
def norm(a: tensor, ord=2, axis=None, keepdims=False) -> tensor:
    """
    Compute vector or matrix norm.
    ord=1 -> L1 norm
    ord=2 -> L2 norm (default)
    ord=np.inf -> max(abs(a))
    """
    return tensor(np.linalg.norm(a.data, ord=ord, axis=axis, keepdims=keepdims))
