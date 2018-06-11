#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
import pyswarms.backend as P


def test_compute_pbest_return_values(swarm):
    """Test if compute_pbest() gives the expected return values"""
    expected_cost = np.array([1,2,2])
    expected_pos = np.array([[1,2,3], [4,5,6], [1,1,1]])
    pos, cost = P.compute_pbest(swarm)
    assert (pos == expected_pos).all()
    assert (cost == expected_cost).all()

@pytest.mark.parametrize('clamp', [None, (0,1), (-1,1)])
def test_compute_velocity_return_values(swarm, clamp):
    """Test if compute_velocity() gives the expected shape and range"""
    v = P.compute_velocity(swarm, clamp)
    assert v.shape == swarm.position.shape
    if clamp is not None:
        assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

@pytest.mark.parametrize('bounds', [None,  ([-5,-5,-5],[5,5,5]),
    ([-10, -10, -10],[10, 10, 10])])
def test_compute_position_return_values(swarm, bounds):
    """Test if compute_position() gives the expected shape and range"""
    p = P.compute_position(swarm, bounds)
    assert p.shape == swarm.velocity.shape
    if bounds is not None:
        assert (bounds[0] <= p).all() and (bounds[1] >= p).all()
