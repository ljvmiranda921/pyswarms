#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest

from .abc_test_optimizer import ABCTestOptimizer

# from pyswarms.utils.functions.single_obj import sphere


class ABCTestDiscreteOptimizer(ABCTestOptimizer):
    """Abstract class that defines various tests for high-level optimizers

    Whenever an optimizer implementation inherits from ABCTestOptimizer,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_with_kwargs(self, obj_with_args, optimizer, options):
        """Test if kwargs are passed properly in objfunc"""
        opt = optimizer(100, 2, options=options)
        cost, pos = opt.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03)
        assert np.isclose(pos[0], 1.0, rtol=1e-03)
        assert np.isclose(pos[1], 1.0, rtol=1e-03)

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_unnecessary_kwargs(
        self, obj_without_args, optimizer, options
    ):
        """Test if error is raised given unnecessary kwargs"""
        opt = optimizer(100, 2, options=options)
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            cost, pos = opt.optimize(obj_without_args, 1000, a=1)

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_missing_kwargs(self, obj_with_args, optimizer, options):
        """Test if error is raised with incomplete kwargs"""
        opt = optimizer(100, 2, options=options)
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            cost, pos = opt.optimize(obj_with_args, 1000, a=1)

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_incorrect_kwargs(self, obj_with_args, optimizer, options):
        """Test if error is raised with wrong kwargs"""
        opt = optimizer(100, 2, options=options)
        with pytest.raises(TypeError):
            # Wrong kwargs
            cost, pos = opt.optimize(obj_with_args, 1000, c=1, d=100)
