#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
import pyswarms.backend as P


def test_update_pbest_return_values(pbest):
    """Test if update_pbest() gives the expected return values"""
    expected_cost = np.array([1,2,2])
    expected_pos = np.array([[1,2,3], [4,5,6], [3,3,3]])
    pos, cost = P.update_pbest(**pbest)
    assert (pos == expected_pos).all()
    assert (cost == expected_cost).all()

def test_update_gbest_return_values(pbest):
    """Test if update_gbest() gives the expected return values"""
    expected_cost = 1
    expected_pos = np.array([1,2,3])
    pos, cost = P.update_gbest(**{k : pbest[k] for k in {'pbest_cost', 'pbest_pos'}})
    assert cost == expected_cost 
    assert (pos == expected_pos).all()

@pytest.mark.parametrize('velocity', [np.array([[1,1],[1,1]])])
@pytest.mark.parametrize('clamp', [None, (0,1)])
def test_update_velocity_return_values(swarm_at_t, velocity, clamp):
    """Test if update_velocity() gives the expected shape and range"""
    v = P.update_velocity(velocity, clamp, **swarm_at_t)
    assert v.shape == (2,2)
    if clamp is not None:
        assert (clamp[0] <= v).all() and (clamp[1] >= v).all()