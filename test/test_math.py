import pytest
import numpy as np
from skml.core.tensor import tensor
from skml.ops.math import *


# --------------------------
# Binary operations tests
# --------------------------
def test_add():
    a = tensor([1, 2, 3])
    b = tensor([4, 5, 6])
    c = add(a, b)
    expected = np.array([5, 7, 9])
    assert np.all(c.data == expected)


def test_sub():
    a = tensor([5, 7, 9])
    b = tensor([1, 2, 3])
    c = sub(a, b)
    expected = np.array([4, 5, 6])
    assert np.all(c.data == expected)


def test_mul():
    a = tensor([1, 2, 3])
    b = tensor([4, 5, 6])
    c = mul(a, b)
    expected = np.array([4, 10, 18])
    assert np.all(c.data == expected)


def test_div():
    a = tensor([4, 9, 16])
    b = tensor([2, 3, 4])
    c = div(a, b)
    expected = np.array([2, 3, 4])
    assert np.all(c.data == expected)


def test_pow():
    a = tensor([2, 3, 4])
    b = tensor([3, 2, 1])
    c = pow(a, b)
    expected = np.array([8, 9, 4])
    assert np.all(c.data == expected)


def test_mod():
    a = tensor([5, 7, 9])
    b = tensor([3, 5, 4])
    c = mod(a, b)
    expected = np.array([2, 2, 1])
    assert np.all(c.data == expected)


# --------------------------
# Unary operations tests
# --------------------------
def test_neg_abs_sqrt_exp_log():
    a = tensor([1, 4, 9])

    n = neg(a)
    assert np.all(n.data == np.array([-1, -4, -9]))

    ab = abs(a)
    assert np.all(ab.data == np.array([1, 4, 9]))

    s = sqrt(a)
    assert np.all(s.data == np.sqrt(np.array([1, 4, 9])))

    e = exp(a)
    assert np.allclose(e.data, np.exp(np.array([1, 4, 9])))

    l = log(a)
    assert np.allclose(l.data, np.log(np.array([1, 4, 9])))


# --------------------------
# Broadcasting test
# --------------------------
def test_broadcasting_add():
    a = tensor([[1, 2], [3, 4]])
    b = tensor([1, 2])
    c = add(a, b)
