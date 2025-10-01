import pytest
import numpy as np
from skml.core.tensor import tensor
from skml.ops.creation import zeros, ones, randn, rand, arange

# --------------------------
# Zeros test
# --------------------------
def test_zeros():
    a = zeros((2,3))
    expected = np.zeros((2,3))
    assert np.all(a.data == expected)
    assert isinstance(a, tensor)

# --------------------------
# Ones test
# --------------------------
def test_ones():
    a = ones((2,3))
    expected = np.ones((2,3))
    assert np.all(a.data == expected)
    assert isinstance(a, tensor)

# --------------------------
# Random normal test
# --------------------------
def test_randn():
    a = randn((1000,), mean=0, std=1)
    # check shape
    assert a.data.shape == (1000,)
    # check roughly mean/std
    assert abs(np.mean(a.data) - 0) < 0.1
    assert abs(np.std(a.data) - 1) < 0.1
    assert isinstance(a, tensor)

# --------------------------
# Random uniform test
# --------------------------
def test_rand():
    a = rand((1000,), low=0, high=10)
    # check range
    assert np.all(a.data >= 0)
    assert np.all(a.data <= 10)
    assert a.data.shape == (1000,)
    assert isinstance(a, tensor)

# --------------------------
# Arange test
# --------------------------
def test_arange():
    a = arange(5)
    expected = np.array([0,1,2,3,4])
    assert np.all(a.data == expected)

    b = arange(1,6)
    expected = np.array([1,2,3,4,5])
    assert np.all(b.data == expected)

    c = arange(0, 10, 2)
    expected = np.array([0,2,4,6,8])
    assert np.all(c.data == expected)
