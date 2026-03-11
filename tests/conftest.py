import numpy as np
import pytest
from test_functions import test_function_1 as _tf1, test_function_2 as _tf2

collect_ignore = ["test_functions.py"]


@pytest.fixture
def setup_1d():
    """Common 1D setup: domain [-1,1], 11 points, _tf1 values."""
    shape = (11,)
    x = np.linspace(-1, 1, shape[0])
    y, _, _, _ = _tf1(x)
    interval = ((-1, 1, shape[0]),)
    bc = (("not-a-knot", "not-a-knot"),)
    bc_value = ((0.0, 0.0),)
    return dict(shape=shape, x=x, y=y, interval=interval, bc=bc, bc_value=bc_value)


@pytest.fixture
def setup_2d():
    """Common 2D setup: domain [0,1]^2, shape (11,13)."""
    shape = (11, 13)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, grad, hess, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    bc = (("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot"))
    bc_value = ((0.0, 0.0), (0.0, 0.0))
    return dict(shape=shape, x1=x1, x2=x2, y=y, interval=interval, bc=bc, bc_value=bc_value)


@pytest.fixture
def setup_3d():
    """Common 3D setup: domain [0,1]^3, shape (7,9,8)."""
    shape = (7, 9, 8)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    x3 = np.linspace(0, 1, shape[2])
    y, grad, hess, _ = _tf1(x1, x2, x3)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]), (0, 1, shape[2]))
    bc = (("not-a-knot", "not-a-knot"),) * 3
    bc_value = ((0.0, 0.0),) * 3
    return dict(shape=shape, x1=x1, x2=x2, x3=x3, y=y, interval=interval, bc=bc, bc_value=bc_value)
