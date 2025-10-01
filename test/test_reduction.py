# tests/test_reduction_tensor.py
import pytest
import numpy as np
from skml.core.tensor import tensor
from skml.ops.reduction import sum, mean, max, min, norm

# --------------------------
# Sum tests
# --------------------------
def test_sum():
    a = tensor([[1, 2], [3, 4]])
    s_all = sum(a)
    s_axis0 = sum(a, axis=0)
    s_axis1 = sum(a, axis=1)

    assert np.all(s_all.data == 10)
    assert np.all(s_axis0.data == np.array([4, 6]))
    assert np.all(s_axis1.data == np.array([3, 7]))

# --------------------------
# Mean tests
# --------------------------
def test_mean():
    a = tensor([[1, 2], [3, 4]])
    m_all = mean(a)
    m_axis0 = mean(a, axis=0)
    m_axis1 = mean(a, axis=1)

    assert np.all(m_all.data == 2.5)
    assert np.all(m_axis0.data == np.array([2., 3.]))
    assert np.all(m_axis1.data == np.array([1.5, 3.5]))

# --------------------------
# Max tests
# --------------------------
def test_max():
    a = tensor([[1, 5], [3, 4]])
    m_all = max(a)
    m_axis0 = max(a, axis=0)
    m_axis1 = max(a, axis=1)

    assert m_all.data == 5
    assert np.all(m_axis0.data == np.array([3, 5]))
    assert np.all(m_axis1.data == np.array([5, 4]))

# --------------------------
# Min tests
# --------------------------
def test_min():
    a = tensor([[1, 5], [3, 4]])
    mn_all = min(a)
    mn_axis0 = min(a, axis=0)
    mn_axis1 = min(a, axis=1)

    assert mn_all.data == 1
    assert np.all(mn_axis0.data == np.array([1, 4]))
    assert np.all(mn_axis1.data == np.array([1, 3]))

# --------------------------
# Norm tests
# --------------------------
def test_norm():
    a = tensor([[3, 4], [0, 5]])
    n2_all = norm(a)  # default L2 norm
    n1_all = norm(a, ord=1)
    n_inf_all = norm(a, ord=np.inf)

    assert np.isclose(n2_all.data, np.linalg.norm(a.data))
    assert np.isclose(n1_all.data, np.linalg.norm(a.data, ord=1))
    assert np.isclose(n_inf_all.data, np.linalg.norm(a.data, ord=np.inf))

    # axis-specific
    n2_axis0 = norm(a, axis=0)
    expected = np.linalg.norm(a.data, axis=0)
    assert np.allclose(n2_axis0.data, expected)
