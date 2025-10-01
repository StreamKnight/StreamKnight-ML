import numpy as np
from skml.core.tensor import tensor

# --------------------------
# Matrix multiplication
# --------------------------
def matmul(a: tensor, b: tensor) -> tensor:
    """
    Matrix multiplication: a @ b
    """
    return tensor(a.data @ b.data)

# --------------------------
# Dot product
# --------------------------
def dot(a: tensor, b: tensor) -> tensor:
    """
    Dot product of two tensors
    """
    return tensor(np.dot(a.data, b.data))

# --------------------------
# Transpose
# --------------------------
def transpose(a: tensor, axes=None) -> tensor:
    """
    Transpose tensor. If axes is None, reverse dimensions.
    """
    return tensor(a.data.T if axes is None else a.data.transpose(axes))

# --------------------------
# Linear layer forward
# --------------------------
def linear(x: tensor, weight: tensor, bias: tensor = None) -> tensor:
    """
    Computes linear transformation: x @ weight + bias
    """
    out = tensor(x.data @ weight.data)
    if bias is not None:
        out = tensor(out.data + bias.data)
    return out
