# tests/test_linear_tensor.py
import pytest
import numpy as np
from skml.core.tensor import tensor
from skml.ops.linear import matmul, dot, transpose, linear

# --------------------------
# Matrix multiplication test
# --------------------------
def test_matmul():
    a = tensor([[1, 2], [3, 4]])
    b = tensor([[5, 6], [7, 8]])
    c = matmul(a, b)
    expected = np.array([[19, 22], [43, 50]])
    assert np.all(c.data == expected)

# --------------------------
# Dot product test
# --------------------------
def test_dot():
    a = tensor([1, 2, 3])
    b = tensor([4, 5, 6])
    c = dot(a, b)
    expected = 32  # 1*4 + 2*5 + 3*6
    assert c.data == expected

# --------------------------
# Transpose test
# --------------------------
def test_transpose():
    a = tensor([[1, 2, 3], [4, 5, 6]])
    t1 = transpose(a)  # default transpose
    expected1 = np.array([[1,4],[2,5],[3,6]])
    assert np.all(t1.data == expected1)

    t2 = transpose(a, axes=(1,0))
    expected2 = expected1
    assert np.all(t2.data == expected2)

# --------------------------
# Linear layer forward test
# --------------------------
def test_linear():
    x = tensor([[1, 2], [3, 4]])        # 2x2 input
    weight = tensor([[1, 0], [0, 1]])   # identity matrix
    bias = tensor([1, 1])               # bias added to each row
    out = linear(x, weight, bias)
    expected = np.array([[2, 3], [4, 5]])  # x@I + bias
    assert np.all(out.data == expected)

    # test linear without bias
    out2 = linear(x, weight)
    expected2 = np.array([[1, 2], [3, 4]])
    assert np.all(out2.data == expected2)
