#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests

isort:skip_file
"""

# Import standard library
import os

# Import modules
import matplotlib as mpl
import numpy as np
import pytest

if os.environ.get("DISPLAY", "") == "":
    mpl.use("Agg")

# Import from pyswarms
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere
from pyswarms.utils.plotters.formatters import Mesher


@pytest.fixture
def trained_optimizer():
    """Returns a trained optimizer instance with 100 iterations"""
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    optimizer.optimize(sphere, iters=100)
    return optimizer


@pytest.fixture
def pos_history():
    """Returns a list containing a swarms' position history"""
    return np.random.uniform(size=(10, 5, 2))


@pytest.fixture
def mesher():
    """A Mesher instance with sphere function and delta=0.1"""
    return Mesher(func=sphere, delta=0.1)
