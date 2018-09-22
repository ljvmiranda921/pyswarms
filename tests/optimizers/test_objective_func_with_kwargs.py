#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from pyswarms.single import GlobalBestPSO, LocalBestPSO
from pyswarms.utils.functions.single_obj import rosenbrock


def rosenbrock_with_args(x, a, b):

    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2
    return f


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_global_kwargs(func):
    """Tests if kwargs are passed properly to the objective function for when kwargs are present"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    cost, pos = opt_ps.optimize(func, 1000, a=1, b=100)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_global_kwargs_without_named_arguments(func):
    """Tests if kwargs are passed properly to the objective function for when kwargs are present and
    other named arguments are not passed, such as print_step"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    cost, pos = opt_ps.optimize(func, 1000, a=1, b=100)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize("func", [rosenbrock])
def test_global_no_kwargs(func):
    """Tests if args are passed properly to the objective function for when no args are present"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    cost, pos = opt_ps.optimize(func, 1000)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_local_kwargs(func):
    """Tests if kwargs are passed properly to the objective function for when kwargs are present"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.7, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    cost, pos = opt_ps.optimize(func, 8000, a=1, b=100)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize("func", [rosenbrock])
def test_local_no_kwargs(func):
    """Tests if no kwargs/args are passed properly to the objective function for when kwargs are present"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.7, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    cost, pos = opt_ps.optimize(func, iters=8000)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize("func", [rosenbrock])
def test_global_uneeded_kwargs(func):
    """Tests kwargs are passed the objective function for when kwargs do not exist"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    with pytest.raises(TypeError) as excinfo:
        cost, pos = opt_ps.optimize(func, 1000, a=1)
        assert "unexpected keyword" in str(excinfo.value)


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_global_missed_kwargs(func):
    """Tests kwargs are passed the objective function for when kwargs do not exist"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    with pytest.raises(TypeError) as excinfo:
        cost, pos = opt_ps.optimize(func, 1000, a=1)
        assert "missing 1 required positional argument" in str(excinfo.value)


@pytest.mark.parametrize("func", [rosenbrock])
def test_local_uneeded_kwargs(func):
    """Tests kwargs are passed the objective function for when kwargs do not exist"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    with pytest.raises(TypeError) as excinfo:
        cost, pos = opt_ps.optimize(func, 1000, a=1)
        assert "unexpected keyword" in str(excinfo.value)


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_local_missed_kwargs(func):
    """Tests kwargs are passed the objective function for when kwargs do not exist"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    with pytest.raises(TypeError) as excinfo:
        cost, pos = opt_ps.optimize(func, 1000, a=1)
        assert "missing 1 required positional argument" in str(excinfo.value)


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_local_wrong_kwargs(func):
    """Tests kwargs are passed the objective function for when kwargs do not exist"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    with pytest.raises(TypeError) as excinfo:
        cost, pos = opt_ps.optimize(func, 3000, print_step=10, c=1, d=100)
        assert "unexpected keyword" in str(excinfo.value)


@pytest.mark.parametrize("func", [rosenbrock_with_args])
def test_global_wrong_kwargs(func):
    """Tests kwargs are passed the objective function for when kwargs do not exist"""

    # setup optimizer
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9, "k": 2, "p": 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(
        n_particles=50, dimensions=2, options=options, bounds=bounds
    )

    # run it
    with pytest.raises(TypeError) as excinfo:
        cost, pos = opt_ps.optimize(func, 1000, c=1, d=100)
        assert "unexpected keyword" in str(excinfo.value)
