#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func


@pytest.mark.parametrize(
    "options",
    [{"c2": 0.7, "w": 0.5}, {"c1": 0.5, "w": 0.5}, {"c1": 0.5, "c2": 0.7}],
)
def test_keyword_exception(options):
    """Tests if exceptions are thrown when keywords are missing"""
    with pytest.raises(KeyError):
        GlobalBestPSO(5, 2, options)


@pytest.mark.parametrize(
    "bounds",
    [
        tuple(np.array([-5, -5])),
        (np.array([-5, -5, -5]), np.array([5, 5])),
        (np.array([-5, -5, -5]), np.array([5, 5, 5])),
    ],
)
def test_bounds_size_exception(bounds, options):
    """Tests if exceptions are raised when bound sizes are wrong"""
    with pytest.raises(IndexError):
        GlobalBestPSO(5, 2, options=options, bounds=bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        (np.array([5, 5]), np.array([-5, -5])),
        (np.array([5, -5]), np.array([-5, 5])),
    ],
)
def test_bounds_maxmin_exception(bounds, options):
    """Tests if the max bounds is less than min bounds and vice-versa"""
    with pytest.raises(ValueError):
        GlobalBestPSO(5, 2, options=options, bounds=bounds)


@pytest.mark.parametrize(
    "bounds",
    [
        [np.array([-5, -5]), np.array([5, 5])],
        np.array([np.array([-5, -5]), np.array([5, 5])]),
    ],
)
def test_bound_type_exception(bounds, options):
    """Tests if exception is raised when bound type is not a tuple"""
    with pytest.raises(TypeError):
        GlobalBestPSO(5, 2, options=options, bounds=bounds)


@pytest.mark.parametrize("velocity_clamp", [(1, 1, 1), (2, 3, 1)])
def test_vclamp_shape_exception(velocity_clamp, options):
    """Tests if exception is raised when velocity_clamp's size is not equal
    to 2"""
    with pytest.raises(IndexError):
        GlobalBestPSO(5, 2, velocity_clamp=velocity_clamp, options=options)


@pytest.mark.parametrize("velocity_clamp", [(3, 2), (10, 8)])
def test_vclamp_maxmin_exception(velocity_clamp, options):
    """Tests if the max velocity_clamp is less than min velocity_clamp and
    vice-versa"""
    with pytest.raises(ValueError):
        GlobalBestPSO(5, 2, velocity_clamp=velocity_clamp, options=options)


@pytest.mark.parametrize("err, center", [(IndexError, [1.5, 3.2, 2.5])])
def test_center_exception(err, center, options):
    """Tests if exception is thrown when center is not a list or of different shape"""
    with pytest.raises(err):
        GlobalBestPSO(5, 2, center=center, options=options)


def test_reset_default_values(gbest_reset):
    """Tests if best cost and best pos are set properly when the reset()
    method is called"""
    assert gbest_reset.swarm.best_cost == np.inf
    assert set(gbest_reset.swarm.best_pos) == set(np.array([]))


@pytest.mark.parametrize(
    "history, expected_shape",
    [
        ("cost_history", (1000,)),
        ("mean_pbest_history", (1000,)),
        ("mean_neighbor_history", (1000,)),
        ("pos_history", (1000, 10, 2)),
        ("velocity_history", (1000, 10, 2)),
    ],
)
def test_training_history_shape(gbest_history, history, expected_shape):
    """Test if training histories are of expected shape"""
    pso = vars(gbest_history)
    assert np.array(pso[history]).shape == expected_shape


def test_ftol_effect(options):
    """Test if setting the ftol breaks the optimization process accodingly"""
    pso = GlobalBestPSO(10, 2, options=options, ftol=1e-1)
    pso.optimize(sphere_func, 2000, verbose=0)
    assert np.array(pso.cost_history).shape != (2000,)
