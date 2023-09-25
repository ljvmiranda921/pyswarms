#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
from abc import ABC
from typing import Any, Callable, Type
import numpy as np
import numpy.typing as npt
import pytest
from pyswarms.base.base import Options

from pyswarms.base.discrete import DiscreteSwarmOptimizer


class ABCTestDiscreteOptimizer(ABC):
    """Abstract class that defines various tests for high-level optimizers

    Whenever an optimizer implementation inherits from ABCTestOptimizer,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_with_kwargs(self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: Type[DiscreteSwarmOptimizer], options: Options):
        """Test if kwargs are passed properly in objfunc"""
        opt = optimizer(100, 2, options=options)
        cost, pos = opt.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03)
        assert np.isclose(pos[0], 1.0, rtol=1e-03)
        assert np.isclose(pos[1], 1.0, rtol=1e-03)

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_unnecessary_kwargs(self, obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]], optimizer: Type[DiscreteSwarmOptimizer], options: Options):
        """Test if error is raised given unnecessary kwargs"""
        opt = optimizer(100, 2, options=options)
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            opt.optimize(obj_without_args, 1000, a=1)

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_missing_kwargs(self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: Type[DiscreteSwarmOptimizer], options: Options):
        """Test if error is raised with incomplete kwargs"""
        opt = optimizer(100, 2, options=options)
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            opt.optimize(obj_with_args, 1000, a=1)

    @pytest.mark.skip("No way of testing this yet")
    def test_obj_incorrect_kwargs(self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: Type[DiscreteSwarmOptimizer], options: Options):
        """Test if error is raised with wrong kwargs"""
        opt = optimizer(100, 2, options=options)
        with pytest.raises(TypeError):
            # Wrong kwargs
            opt.optimize(obj_with_args, 1000, c=1, d=100)
