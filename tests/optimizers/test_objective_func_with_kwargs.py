#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.single import GlobalBestPSO, LocalBestPSO
from pyswarms.utils.functions.single_obj import rosenbrock_func


def rosenbrock_with_args(x, a, b):

    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2
    return f


@pytest.mark.parametrize('func', [
    rosenbrock_with_args
])
@pytest.mark.parametrize('kwargs', [
    {"a": 1, "b": 100}
])
def test_global_kwargs(func, kwargs):
    """Tests if kwargs are passed properly to the objective function for when kwargs are present"""

    # setup optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

    # run it
    cost, pos = opt_ps.optimize(func, iters=1000, print_step=10, verbose=3, **kwargs)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize('func', [
    rosenbrock_with_args
])
@pytest.mark.parametrize('args', [
    (1, 100)
])
def test_global_args(func, args):
    """Tests if args are passed properly to the objective function for when args are present"""

    # setup optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

    # run it
    cost, pos = opt_ps.optimize(func, iters=1000, print_step=10, verbose=3, *args)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize('func', [
    rosenbrock_func
])
def test_global_no_kwargs(func):
    """Tests if args are passed properly to the objective function for when no args are present"""

    # setup optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

    # run it
    cost, pos = opt_ps.optimize(func, iters=1000, print_step=10, verbose=3)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize('func', [
    rosenbrock_with_args
])
@pytest.mark.parametrize('kwargs', [
    {"a": 1, "b": 100}
])
def test_local_kwargs(func, kwargs):
    """Tests if kwargs are passed properly to the objective function for when kwargs are present"""

    # setup optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

    # run it
    cost, pos = opt_ps.optimize(func, iters=1000, print_step=10, verbose=3, **kwargs)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize('func', [
    rosenbrock_with_args
])
@pytest.mark.parametrize('args', [
    (1, 100)
])
def test_local_args(func, args):
    """Tests if args are passed properly to the objective function """

    # setup optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = LocalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

    # run it
    cost, pos = opt_ps.optimize(func, iters=1000, print_step=10, verbose=3, *args)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)


@pytest.mark.parametrize('func', [
    rosenbrock_func
])
def test_local_no_kwargs(func):
    """Tests if no kwargs/args are passed properly to the objective function for when kwargs are present"""

    # setup optimizer
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}

    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    opt_ps = GlobalBestPSO(n_particles=100, dimensions=2, options=options, bounds=bounds)

    # run it
    cost, pos = opt_ps.optimize(func, iters=1000, print_step=10, verbose=3)

    assert np.isclose(cost, 0, rtol=1e-03)
    assert np.isclose(pos[0], 1.0, rtol=1e-03)
    assert np.isclose(pos[1], 1.0, rtol=1e-03)
