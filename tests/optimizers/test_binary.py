#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest

# Import from pyswarms
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_discrete_optimizer import ABCTestDiscreteOptimizer


class TestDiscreteOptimizer(ABCTestDiscreteOptimizer):
    @pytest.fixture
    def optimizer(self):
        return BinaryPSO

    @pytest.fixture
    def optimizer_history(self, options):
        opt = BinaryPSO(10, 2, options=options)
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture
    def optimizer_reset(self, options):
        opt = BinaryPSO(10, 2, options=options)
        opt.optimize(sphere, 10)
        opt.reset()
        return opt
