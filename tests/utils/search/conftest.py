#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.single import LocalBestPSO
from pyswarms.utils.functions.single_obj import sphere

# Import from package
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.utils.search.random_search import RandomSearch


@pytest.fixture
def grid():
    """Returns a GridSearch instance"""
    options = {
        "c1": [1, 2, 3],
        "c2": [1, 2, 3],
        "k": [5, 10, 15],
        "w": [0.9, 0.7, 0.4],
        "p": [1],
    }
    return GridSearch(
        LocalBestPSO,
        n_particles=40,
        dimensions=20,
        options=options,
        objective_func=sphere,
        iters=10,
        bounds=None,
    )


@pytest.fixture
def grid_mini():
    """Returns a GridSearch instance with a smaller search-space"""
    options = {"c1": [1, 2], "c2": 6, "k": 5, "w": 0.9, "p": 0}
    return GridSearch(
        LocalBestPSO,
        n_particles=40,
        dimensions=20,
        options=options,
        objective_func=sphere,
        iters=10,
        bounds=None,
    )


@pytest.fixture
def random_unbounded():
    """Returns a RandomSearch instance without bounds"""
    options = {
        "c1": [1, 5],
        "c2": [6, 10],
        "k": [11, 15],
        "w": [0.4, 0.9],
        "p": 1,
    }
    return RandomSearch(
        LocalBestPSO,
        n_particles=40,
        dimensions=20,
        options=options,
        objective_func=sphere,
        iters=10,
        n_selection_iters=100,
        bounds=None,
    )


@pytest.fixture
def random_bounded():
    """Returns a RandomSearch instance with bounds"""
    bounds = (np.array([-5, -5]), np.array([5, 5]))
    options = {
        "c1": [1, 5],
        "c2": [6, 10],
        "k": [11, 15],
        "w": [0.4, 0.9],
        "p": 1,
    }
    return RandomSearch(
        LocalBestPSO,
        n_particles=40,
        dimensions=20,
        options=options,
        objective_func=sphere,
        iters=10,
        n_selection_iters=100,
        bounds=bounds,
    )
