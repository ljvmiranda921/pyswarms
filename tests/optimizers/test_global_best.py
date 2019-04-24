#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from pyswarms
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_optimizer import ABCTestOptimizer


class TestGlobalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self):
        return GlobalBestPSO

    @pytest.fixture
    def optimizer_history(self, options):
        opt = GlobalBestPSO(10, 2, options=options)
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture
    def optimizer_reset(self, options):
        opt = GlobalBestPSO(10, 2, options=options)
        opt.optimize(sphere, 10)
        opt.reset()
        return opt

    def test_global_correct_pos(self, options):
        """ Test to check global optimiser returns the correct position corresponding to the best cost """
        opt = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
        cost, pos = opt.optimize(sphere, iters=5)
        # find best pos from history
        min_cost_idx = np.argmin(opt.cost_history)
        min_pos_idx = np.argmin(sphere(opt.pos_history[min_cost_idx]))
        assert np.array_equal(opt.pos_history[min_cost_idx][min_pos_idx], pos)
