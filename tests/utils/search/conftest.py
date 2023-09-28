#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

from typing import Dict, List, Tuple
import pytest

from pyswarms.optimizers import LocalBestPSO
from pyswarms.utils.functions.single_obj import sphere
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.utils.types import SwarmOption


@pytest.fixture
def grid():
    """Returns a GridSearch instance"""
    options: Dict[SwarmOption, float|List[float]] = {
        "c1": [1, 2, 3],
        "c2": [1, 2, 3],
        "w": [0.9, 0.7, 0.4],
        # "k": [5, 10, 15],
        # "p": [1],
    }
    
    return GridSearch(
        LocalBestPSO(40, 20, {"c1": 1, "c2": 1, "w": 1}),
        sphere,
        10,
        options
    )


@pytest.fixture
def grid_mini():
    """Returns a GridSearch instance with a smaller search-space"""
    # options = {"c1": [1, 2], "c2": 6, "k": 5, "w": 0.9, "p": 0}
    options: Dict[SwarmOption, float|List[float]] = {"c1": [1., 2.], "c2": 6., "w": 0.9}
    
    return GridSearch(
        LocalBestPSO(40, 20, {"c1": 1, "c2": 1, "w": 1}),
        sphere,
        10,
        options
    )


@pytest.fixture
def random_unbounded():
    """Returns a RandomSearch instance without bounds"""
    options: Dict[SwarmOption, float|Tuple[float, float]] = {
        "c1": (1, 5),
        "c2": (6, 10),
        "w": (0.4, 0.9),
        # "k": (11, 15),
        # "p": 1,
    }
    
    return RandomSearch(
        LocalBestPSO(40, 20, {"c1": 1, "c2": 1, "w": 1}),
        sphere,
        10,
        100,
        options
    )


@pytest.fixture
def random_bounded():
    """Returns a RandomSearch instance with bounds"""
    bounds = (-5, 5)
    options: Dict[SwarmOption, float|Tuple[float, float]] = {
        "c1": (1, 5),
        "c2": (6, 10),
        "w": (0.4, 0.9),
        # "k": [11, 15],
        # "p": 1,
    }
    
    return RandomSearch(
        LocalBestPSO(40, 20, {"c1": 1, "c2": 1, "w": 1}, bounds=bounds),
        sphere,
        10,
        100,
        options
    )
