#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from pyswarms
from pyswarms.single import LocalBestPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_optimizer import ABCTestOptimizer


class TestLocalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self):
        return LocalBestPSO

    @pytest.fixture
    def optimizer_history(self, options):
        opt = LocalBestPSO(10, 2, options)
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture
    def optimizer_reset(self, options):
        opt = LocalBestPSO(10, 2, options)
        opt.optimize(sphere, 10)
        opt.reset()
        return opt
