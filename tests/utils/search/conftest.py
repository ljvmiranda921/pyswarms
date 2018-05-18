#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single import LocalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

@pytest.fixture
def grid():
    options = {'c1': [1, 2, 3],
               'c2': [1, 2, 3],
               'k' : [5, 10, 15],
               'w' : [0.9, 0.7, 0.4],
               'p' : [1]}
    return GridSearch(LocalBestPSO, n_particles=40,
        dimensions=20, options=options, objective_func=sphere_func, iters=10, bounds=None, velocity_clamp=None)

@pytest.fixture
def grid_mini():
    options = {'c1': [1, 2],
               'c2': 6,
               'k' : 5,
               'w' : 0.9,
               'p' : 0}
    return GridSearch(LocalBestPSO, n_particles=40,
        dimensions=20, options=options, objective_func=sphere_func, iters=10, bounds=None, velocity_clamp=None)