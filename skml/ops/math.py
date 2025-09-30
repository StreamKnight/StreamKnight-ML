import numpy as np
from skml.core.tensor import tensor

# --------------------------
# Binary operations
# --------------------------
def add(a: tensor, b: tensor) -> tensor:
    return tensor(a.data + b.data)

def sub(a: tensor, b: tensor) -> tensor:
    return tensor(a.data - b.data)

def mul(a: tensor, b: tensor) -> tensor:
    return tensor(a.data * b.data)

def div(a: tensor, b: tensor) -> tensor:
    return tensor(a.data / b.data)

def pow(a: tensor, b: tensor) -> tensor:
    return tensor(a.data ** b.data)

def mod(a: tensor, b: tensor) -> tensor:
    return tensor(a.data % b.data)

# --------------------------
# Unary operations
# --------------------------
def neg(a: tensor) -> tensor:
    return tensor(-a.data)

def abs(a: tensor) -> tensor:
    return tensor(np.abs(a.data))

def sqrt(a: tensor) -> tensor:
    return tensor(np.sqrt(a.data))

def exp(a: tensor) -> tensor:
    return tensor(np.exp(a.data))

def log(a: tensor) -> tensor:
    return tensor(np.log(a.data))
