#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.functions.single_obj import sphere
from tests.optimizers.abc_test_optimizer import ABCTestOptimizer


class TestDiscreteOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self, velocity_updater: VelocityUpdater, position_updater: PositionUpdater):
        return BinaryPSO(10, 2, 2, 2, velocity_updater, position_updater)

    def test_binary_correct_pos(self, optimizer: BinaryPSO):
        """Test to check binary optimiser returns the correct position
        corresponding to the best cost"""
        _, pos = optimizer.optimize(sphere, 10)
        # find best pos from history
        min_cost_idx = np.argmin(optimizer.cost_history)
        min_pos_idx = np.argmin(sphere(optimizer.pos_history[min_cost_idx]))
        assert np.array_equal(optimizer.pos_history[min_cost_idx][min_pos_idx], pos)
