#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests

isort:skip_file
"""

import os

import matplotlib as mpl
import numpy as np
import pytest
from pyswarms.backend.handlers import VelocityHandler
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.topology.star import Star
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.optimizers.optimizer import OptimizerPSO

from pyswarms.utils.types import SwarmOptions

if os.environ.get("DISPLAY", "") == "":
    mpl.use("Agg")

from pyswarms.utils.functions.single_obj import sphere
from pyswarms.utils.plotters.formatters import Mesher


@pytest.fixture
def trained_optimizer():
    """Returns a trained optimizer instance with 100 iterations"""
    options = SwarmOptions({"c1": 0.5, "c2": 0.3, "w": 0.9})
    vu = VelocityUpdater(options, None, VelocityHandler.factory("unmodified"))
    pu = PositionUpdater()
    optimizer = OptimizerPSO(10, 2, Star(), vu, pu)
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
