#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.optimizers import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_optimizer import ABCTestOptimizer


class TestGlobalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self, velocity_updater: VelocityUpdater, position_updater: PositionUpdater):
        return GlobalBestPSO(10, 2, velocity_updater, position_updater)

    def test_global_correct_pos(self, velocity_updater: VelocityUpdater, position_updater: PositionUpdater):
        """Test to check global optimiser returns the correct position corresponding to the best cost"""
        opt = GlobalBestPSO(10, 2, velocity_updater, position_updater)
        _, pos = opt.optimize(sphere, iters=5)

        # find best pos from history
        min_cost_idx = np.argmin(opt.cost_history)
        min_pos_idx = np.argmin(sphere(opt.pos_history[min_cost_idx]))
        assert np.array_equal(opt.pos_history[min_cost_idx][min_pos_idx], pos)
