#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import Callable

import pytest

from pyswarms.backend.handlers import VelocityHandler
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.topology import Star
from pyswarms.backend.topology.ring import Ring
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.optimizers import OptimizerPSO
from pyswarms.optimizers.base import BaseSwarmOptimizer
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.types import SwarmOptions

# Instantiate optimizers
options = SwarmOptions({"c1": 2, "c2": 2, "w": 0.7})
n_particles = 20
dimensions = 10
velocity_updater = VelocityUpdater(options, None, VelocityHandler.factory("unmodified"))
position_updater = PositionUpdater()

optimizers = [
    lambda: OptimizerPSO(n_particles, dimensions, Star(), velocity_updater, position_updater),
    lambda: OptimizerPSO(n_particles, dimensions, Ring(2, 3), velocity_updater, position_updater),
    lambda: OptimizerPSO(n_particles, dimensions, Star(), velocity_updater, position_updater),
]


class TestToleranceOptions:
    @pytest.mark.parametrize("optimizer_func", optimizers)
    def test_verbose(self, optimizer_func: Callable[[], BaseSwarmOptimizer], capsys: pytest.CaptureFixture[str]):
        """Test verbose run"""
        optimizer = optimizer_func()
        optimizer.optimize(fx.sphere, iters=100)
        out = capsys.readouterr().err
        count = len(re.findall(r"pyswarms", out))
        assert count > 0

    @pytest.mark.parametrize("optimizer_func", optimizers)
    def test_silent(self, optimizer_func: Callable[[], BaseSwarmOptimizer], capsys: pytest.CaptureFixture[str]):
        """Test silent run"""
        optimizer = optimizer_func()
        optimizer.optimize(fx.sphere, iters=100, verbose=False)
        out = capsys.readouterr()
        assert out.err == ""
        assert out.out == ""
