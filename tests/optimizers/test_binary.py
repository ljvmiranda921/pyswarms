#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

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

    def test_binary_correct_pos(self, options):
        dim = 10
        x = np.random.rand(dim, dim)

        def f_per_particle(m):
            if np.count_nonzero(m) == 0:
                return sum(x)
            return sum(x[:, m == 1]).mean()

        def binary_eg(options):
            def f(x):
                n_particles = x.shape[0]
                j = [f_per_particle(x[i]) for i in range(n_particles)]
                return np.array(j)

            opt = BinaryPSO(n_particles=5, dimensions=dim, options=options)
            return opt.optimize(f, iters=5)

        cost, pos = binary_eg(options)
        assert f_per_particle(pos) == cost
