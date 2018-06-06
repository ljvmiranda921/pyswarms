#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.backend.topology import Star


def test_compute_gbest_return_values(swarm):
    """Test if compute_gbest() gives the expected return values"""
    topology = Star()
    expected_cost = 1
    expected_pos = np.array([1,2,3])
    pos, cost = topology.compute_gbest(swarm)
    assert cost == expected_cost
    assert (pos == expected_pos).all()

@pytest.mark.parametrize('clamp', [None, (0,1), (-1,1)])
def test_compute_velocity_return_values(swarm, clamp):
    """Test if compute_velocity() gives the expected shape and range"""
    topology = Star()
    v = topology.compute_velocity(swarm, clamp)
    assert v.shape == swarm.position.shape
    if clamp is not None:
        assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

@pytest.mark.parametrize('bounds', [None,  ([-5,-5,-5],[5,5,5]),
    ([-10, -10, -10],[10, 10, 10])])
def test_compute_position_return_values(swarm, bounds):
    """Test if compute_position() gives the expected shape and range"""
    topology = Star()
    p = topology.compute_position(swarm, bounds)
    assert p.shape == swarm.velocity.shape
    if bounds is not None:
        assert (bounds[0] <= p).all() and (bounds[1] >= p).all()