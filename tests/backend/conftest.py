#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import pytest
import numpy as np


@pytest.fixture
def pbest():
    """Returns a dictionary of pbest values"""
    return {
        'pbest_pos'  : np.array([[1,2,3], [4,5,6], [7,8,9]]),
        'pbest_cost' : np.array([1,2,3]),
        'pos' : np.array([[1,1,1], [2,2,2], [3,3,3]]),
        'cost' : np.array([2,2,2])
    }

@pytest.fixture
def swarm_at_t():
    """A contrived set of swarm parameters at time t"""
    return {
        'pos' : np.array([[5,5], [3,3]]),
        'pbest_pos' : np.array([[5,5], [2,2]]),
        'best_pos' : np.array([1,1]),
        'c1' : 0.5,
        'c2' : 1,
        'w'  : 2
    }