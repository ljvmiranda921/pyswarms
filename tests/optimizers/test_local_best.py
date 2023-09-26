#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.velocity import VelocityUpdater

from pyswarms.single import LocalBestPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_optimizer import ABCTestOptimizer


class TestLocalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self, velocity_updater: VelocityUpdater, position_updater: PositionUpdater):
        return LocalBestPSO(10, 2, 2, 2, velocity_updater, position_updater)

    def test_local_correct_pos(self, velocity_updater: VelocityUpdater, position_updater: PositionUpdater):
        """Test to check local optimiser returns the correct position corresponding to the best cost"""
        opt = LocalBestPSO(10, 2, 2, 2, velocity_updater, position_updater)
        _, pos = opt.optimize(sphere, iters=5)
        
        # find best pos from history
        min_cost_idx = np.argmin(opt.cost_history)
        min_pos_idx = np.argmin(sphere(opt.pos_history[min_cost_idx]))
        assert np.array_equal(opt.pos_history[min_cost_idx][min_pos_idx], pos)
