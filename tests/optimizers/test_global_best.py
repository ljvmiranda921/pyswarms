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

    def global_eg(self, options):
        opt = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
        return opt.optimize(sphere, iters=5)

    def test_global_correct_pos(self, options):
        print("Running local test")
        cost, pos = self.global_eg(options)
        assert sum(pos ** 2) == cost
