#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import os
import pytest
import numpy as np
from mock import Mock
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "":
    mpl.use("Agg")

# Import from package
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func
from pyswarms.utils.plotters.formatters import Mesher


@pytest.fixture
def trained_optimizer():
    """Returns a trained optimizer instance with 100 iterations"""
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    optimizer.optimize(sphere_func, iters=100)
    return optimizer


@pytest.fixture
def pos_history():
    """Returns a list containing a swarms' position history"""
    return np.random.uniform(size=(10, 5, 2))


@pytest.fixture
def mesher():
    """A Mesher instance with sphere function and delta=0.1"""
    return Mesher(func=sphere_func, delta=0.1)
