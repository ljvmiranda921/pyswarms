#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest

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
        assert sum(pos ** 2) == cost
