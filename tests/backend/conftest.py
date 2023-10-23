#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

import numpy as np
import pytest

from pyswarms.backend.swarms import Swarm
from pyswarms.backend.velocity import SwarmOptions
from pyswarms.utils.types import Bounds, Clamp, Position, Velocity


@pytest.fixture
def options() -> SwarmOptions:
    return SwarmOptions({"c1": 0.5, "c2": 1, "w": 2})


@pytest.fixture
def swarm():
    """A contrived instance of the Swarm class at a certain timestep"""
    return Swarm(
        position=np.array([[5, 5, 5], [3, 3, 3], [1, 1, 1]]),
        velocity=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        current_cost=np.array([2, 2, 2]),
        pbest_cost=np.array([1, 2, 3]),
        pbest_pos=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        best_cost=1,
        best_pos=np.array([1, 1, 1]),
    )


@pytest.fixture
def bounds() -> Bounds:
    return (np.array([2, 3, 1]), np.array([4, 7, 8]))


@pytest.fixture
def clamp() -> Clamp:
    return (np.array([2, 3, 1]), np.array([4, 7, 8]))


@pytest.fixture
def positions_inbound() -> Position:
    return np.array(
        [
            [3.3, 4.4, 2.3],
            [3.7, 5.2, 7.0],
            [2.5, 6.8, 2.3],
            [2.1, 6.9, 4.7],
            [2.7, 3.2, 3.5],
            [2.5, 5.1, 1.2],
        ]
    )


@pytest.fixture
def positions_out_of_bound() -> Position:
    return np.array(
        [
            [5.3, 4.4, 2.3],
            [3.7, 9.2, 7.0],
            [8.5, 0.8, 2.3],
            [2.1, 6.9, 0.7],
            [2.7, 9.2, 3.5],
            [1.5, 5.1, 9.2],
        ]
    )


@pytest.fixture
def velocities_inbound() -> Velocity:
    return np.array(
        [
            [3.3, 4.4, 2.3],
            [3.7, 5.2, 7.0],
            [2.5, 6.8, 2.3],
            [2.1, 6.9, 4.7],
            [2.7, 3.2, 3.5],
            [2.5, 5.1, 1.2],
        ]
    )


@pytest.fixture
def velocities_out_of_bound() -> Velocity:
    return np.array(
        [
            [5.3, 4.4, 2.3],
            [3.7, 9.2, 7.0],
            [8.5, 0.8, 2.3],
            [2.1, 6.9, 0.7],
            [2.7, 9.2, 3.5],
            [1.5, 5.1, 9.2],
        ]
    )
