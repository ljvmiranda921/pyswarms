#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
import pyswarms.backend as P


def test_update_pbest_return_values(swarm):
    """Test if update_pbest() gives the expected return values"""
    expected_cost = np.array([1,2,2])
    expected_pos = np.array([[1,2,3], [4,5,6], [1,1,1]])
    pos, cost = P.update_pbest(swarm)
    assert (pos == expected_pos).all()
    assert (cost == expected_cost).all()

def test_update_gbest_return_values(swarm):
    """Test if update_gbest() gives the expected return values"""
    expected_cost = 1
    expected_pos = np.array([1,2,3])
    pos, cost = P.update_gbest(swarm)
    assert cost == expected_cost
    assert (pos == expected_pos).all()

@pytest.mark.parametrize('clamp', [None, (0,1), (-1,1)])
def test_update_velocity_return_values(swarm, clamp):
    """Test if update_velocity() gives the expected shape and range"""
    v = P.update_velocity(swarm, clamp)
    assert v.shape == swarm.position.shape
    if clamp is not None:
        assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

@pytest.mark.parametrize('bounds', [None,  ([-5,-5,-5],[5,5,5]),
    ([-10, -10, -10],[10, 10, 10])])
def test_update_position_return_values(swarm, bounds):
    """Test if update_position() gives the expected shape and range"""
    p = P.update_position(swarm, bounds)
    assert p.shape == swarm.velocity.shape
    if bounds is not None:
        assert (bounds[0] <= p).all() and (bounds[1] >= p).all()

@pytest.mark.parametrize('k', [1,2,3])
@pytest.mark.parametrize('p', [1,2])
def test_update_gbest_neighborhood(swarm, p, k):
    """Test if update_gbest_neighborhood gives the expected return values"""
    pos, cost = P.update_gbest_neighborhood(swarm, p=p, k=k)
    expected_pos = np.array([1,2,3])
    expected_cost = 1
    print('k={} p={}, pos={} cost={}'.format(k,p,pos,cost))
    assert (pos == expected_pos).all()
    assert cost == expected_cost