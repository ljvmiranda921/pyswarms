#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.backend.swarms import Swarm


@pytest.fixture
def swarm():
    """A contrived instance of the Swarm class at a certain timestep"""
    attrs_at_t = {
        "position": np.array([[5, 5, 5], [3, 3, 3], [1, 1, 1]]),
        "velocity": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        "current_cost": np.array([2, 2, 2]),
        "pbest_cost": np.array([1, 2, 3]),
        "pbest_pos": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "best_cost": 1,
        "best_pos": np.array([1, 1, 1]),
        "options": {"c1": 0.5, "c2": 1, "w": 2},
    }
    return Swarm(**attrs_at_t)
