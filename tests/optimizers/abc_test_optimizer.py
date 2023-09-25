#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
import abc
from typing import Any, Callable, Tuple, Type

# Import modules
import numpy as np
import numpy.typing as npt
import pytest
from pyswarms.base.base import BaseSwarmOptimizer, Options
from pyswarms.single.general_optimizer import GeneralOptions

# Import from pyswarms
from pyswarms.utils.functions.single_obj import rosenbrock, sphere


class ABCTestOptimizer(abc.ABC):
    """Abstract class that defines various tests for high-level optimizers

    Whenever an optimizer implementation inherits from ABCTestOptimizer,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.fixture
    @abc.abstractmethod
    def optimizer(self) -> Type[BaseSwarmOptimizer]:
        """Return an instance of the optimizer"""
        ...

    @pytest.fixture
    @abc.abstractmethod
    def optimizer_history(self, options: GeneralOptions) -> BaseSwarmOptimizer:
        """Run the optimizer for 1000 iterations and return its instance"""
        ...

    @pytest.fixture
    @abc.abstractmethod
    def optimizer_reset(self, options: GeneralOptions) -> BaseSwarmOptimizer:
        """Reset the optimizer and return its instance"""
        ...

    @pytest.fixture
    def options(self) -> GeneralOptions:
        """Default options dictionary for most PSO use-cases"""
        return GeneralOptions({"c1": 0.3, "c2": 0.7, "w": 0.9, "k": 2, "p": 2, "r": 1})

    @pytest.fixture
    def obj_with_args(self):
        """Objective function with arguments"""

        def obj_with_args_(x: npt.NDArray[Any], a: int, b: int):
            f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2
            return f

        return obj_with_args_

    @pytest.fixture
    def obj_without_args(self):
        """Objective function without arguments"""
        return rosenbrock

    @pytest.mark.parametrize(
        "history, expected_shape",
        [
            ("cost_history", (1000,)),
            ("mean_pbest_history", (1000,)),
            ("mean_neighbor_history", (1000,)),
            ("pos_history", (1000, 10, 2)),
            ("velocity_history", (1000, 10, 2)),
        ],
    )
    def test_train_history(self, optimizer_history: BaseSwarmOptimizer, history: str, expected_shape: Tuple[int, ...]):
        """Test if training histories are of expected shape"""
        opt = vars(optimizer_history)
        assert np.array(opt[history]).shape == expected_shape

    def test_reset_default_values(self, optimizer_reset: BaseSwarmOptimizer):
        """Test if best cost and best pos are set properly when the reset()
        method is called"""
        assert optimizer_reset.swarm.best_cost == np.inf
        assert set(optimizer_reset.swarm.best_pos) == set()

    @pytest.mark.skip(reason="The Ring topology converges too slowly")
    def test_ftol_effect(self, options: Options, optimizer: Type[BaseSwarmOptimizer]):
        """Test if setting the ftol breaks the optimization process"""
        opt = optimizer(10, 2, options=options, ftol=1e-1)
        opt.optimize(sphere, 2000)
        assert np.array(opt.cost_history).shape != (2000,)

    def test_parallel_evaluation(self, obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]], optimizer: Type[BaseSwarmOptimizer], options: Options):
        """Test if parallelization breaks the optimization process"""
        # Import standard library
        import multiprocessing

        opt = optimizer(100, 2, options=options)
        opt.optimize(obj_without_args, 2000, n_processes=multiprocessing.cpu_count())
        assert np.array(opt.cost_history).shape == (2000,)

    def test_obj_with_kwargs(self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: Type[BaseSwarmOptimizer], options: Options):
        """Test if kwargs are passed properly in objfunc"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        cost, pos = opt.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03)
        assert np.isclose(pos[0], 1.0, rtol=1e-03)
        assert np.isclose(pos[1], 1.0, rtol=1e-03)

    def test_obj_unnecessary_kwargs(self, obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]], optimizer: Type[BaseSwarmOptimizer], options: Options):
        """Test if error is raised given unnecessary kwargs"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            opt.optimize(obj_without_args, 1000, a=1)

    def test_obj_missing_kwargs(self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: Type[BaseSwarmOptimizer], options: Options):
        """Test if error is raised with incomplete kwargs"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            opt.optimize(obj_with_args, 1000, a=1)

    def test_obj_incorrect_kwargs(self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: Type[BaseSwarmOptimizer], options: Options):
        """Test if error is raised with wrong kwargs"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        opt = optimizer(100, 2, options=options, bounds=bounds)
        with pytest.raises(TypeError):
            # Wrong kwargs
            opt.optimize(obj_with_args, 1000, c=1, d=100)
