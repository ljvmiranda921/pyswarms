#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.optimizers import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere
from pyswarms.utils.types import SwarmOptions

from .abc_test_optimizer import ABCTestOptimizer


class TestGlobalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self, options: SwarmOptions): # type: ignore
        return GlobalBestPSO(10, 2, options)

    def test_global_correct_pos(self, options: SwarmOptions):
        """Test to check global optimiser returns the correct position corresponding to the best cost"""
        opt = GlobalBestPSO(10, 2, options)
        _, pos = opt.optimize(sphere, iters=5)

        # find best pos from history
        min_cost_idx = np.argmin(opt.cost_history)
        min_pos_idx = np.argmin(sphere(opt.pos_history[min_cost_idx]))
        assert np.array_equal(opt.pos_history[min_cost_idx][min_pos_idx], pos)
