#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.backend.topology import Ring


@pytest.mark.parametrize('k', [1,2,3])
@pytest.mark.parametrize('p', [1,2])
def test_update_gbest_neighborhood(swarm, p, k):
    """Test if update_gbest_neighborhood gives the expected return values"""
    topology = Ring()
    pos, cost = topology.compute_gbest(swarm, p=p, k=k)
    expected_pos = np.array([1,2,3])
    expected_cost = 1
    assert (pos == expected_pos).all()
    assert cost == expected_cost

@pytest.mark.parametrize('clamp', [None, (0,1), (-1,1)])
def test_compute_velocity_return_values(swarm, clamp):
    """Test if compute_velocity() gives the expected shape and range"""
    topology = Ring()
    v = topology.compute_velocity(swarm, clamp)
    assert v.shape == swarm.position.shape
    if clamp is not None:
        assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

@pytest.mark.parametrize('bounds', [None,  ([-5,-5,-5],[5,5,5]),
    ([-10, -10, -10],[10, 10, 10])])
def test_compute_position_return_values(swarm, bounds):
    """Test if compute_position() gives the expected shape and range"""
    topology = Ring()
    p = topology.compute_position(swarm, bounds)
    assert p.shape == swarm.velocity.shape
    if bounds is not None:
        assert (bounds[0] <= p).all() and (bounds[1] >= p).all()
