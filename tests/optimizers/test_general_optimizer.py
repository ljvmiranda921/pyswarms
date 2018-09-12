#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.backend.topology import Random, Ring, VonNeumann
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere


@pytest.mark.parametrize("topology", [object(), int(), dict()])
def test_topology_type_exception(options, topology):
    """Tests if exceptions are thrown when the topology has the wrong type"""
    with pytest.raises(TypeError):
        GeneralOptimizerPSO(5, 2, options, topology)


@pytest.mark.parametrize(
    "bounds",
    [
        tuple(np.array([-5, -5])),
        (np.array([-5, -5, -5]), np.array([5, 5])),
        (np.array([-5, -5, -5]), np.array([5, 5, 5])),
    ],
)
def test_bounds_size_exception(bounds, options, topology):
    """Tests if exceptions are raised when bound sizes are wrong"""
    with pytest.raises(IndexError):
        GeneralOptimizerPSO(
            5, 2, options=options, topology=topology, bounds=bounds
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (np.array([5, 5]), np.array([-5, -5])),
        (np.array([5, -5]), np.array([-5, 5])),
    ],
)
def test_bounds_maxmin_exception(bounds, options, topology):
    """Tests if the max bounds is less than min bounds and vice-versa"""
    with pytest.raises(ValueError):
        GeneralOptimizerPSO(
            5, 2, options=options, topology=topology, bounds=bounds
        )


@pytest.mark.parametrize(
    "bounds",
    [
        [np.array([-5, -5]), np.array([5, 5])],
        np.array([np.array([-5, -5]), np.array([5, 5])]),
    ],
)
def test_bound_type_exception(bounds, options, topology):
    """Tests if exception is raised when bound type is not a tuple"""
    with pytest.raises(TypeError):
        GeneralOptimizerPSO(
            5, 2, options=options, topology=topology, bounds=bounds
        )


@pytest.mark.parametrize("velocity_clamp", [(1, 1, 1), (2, 3, 1)])
def test_vclamp_shape_exception(velocity_clamp, options, topology):
    """Tests if exception is raised when velocity_clamp's size is not equal
    to 2"""
    with pytest.raises(IndexError):
        GeneralOptimizerPSO(
            5,
            2,
            velocity_clamp=velocity_clamp,
            options=options,
            topology=topology,
        )


@pytest.mark.parametrize("velocity_clamp", [(3, 2), (10, 8)])
def test_vclamp_maxmin_exception(velocity_clamp, options, topology):
    """Tests if the max velocity_clamp is less than min velocity_clamp and
    vice-versa"""
    with pytest.raises(ValueError):
        GeneralOptimizerPSO(
            5,
            2,
            velocity_clamp=velocity_clamp,
            options=options,
            topology=topology,
        )


@pytest.mark.parametrize("err, center", [(IndexError, [1.5, 3.2, 2.5])])
def test_center_exception(err, center, options, topology):
    """Tests if exception is thrown when center is not a list or of different shape"""
    with pytest.raises(err):
        GeneralOptimizerPSO(
            5, 2, center=center, options=options, topology=topology
        )


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


def test_ftol_effect(options, topology):
    """Test if setting the ftol breaks the optimization process accordingly"""
    pso = GeneralOptimizerPSO(
        10, 2, options=options, topology=topology, ftol=1e-1
    )
    pso.optimize(sphere, 2000)
    assert np.array(pso.cost_history).shape != (2000,)
