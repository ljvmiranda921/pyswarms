#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from pyswarms.backend.handlers import VelocityHandler
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.velocity import VelocityUpdater

from pyswarms.base.base import BaseSwarmOptimizer
from pyswarms.utils.functions.single_obj import rosenbrock, sphere
from pyswarms.utils.types import SwarmOptions


class ABCTestOptimizer(ABC):
    """Abstract class that defines various tests for high-level optimizers

    Whenever an optimizer implementation inherits from ABCTestOptimizer,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.fixture
    @abstractmethod
    def optimizer(self, velocity_updater: VelocityUpdater, position_updater: PositionUpdater) -> BaseSwarmOptimizer:
        """Return an instance of the optimizer"""
        ...

    @pytest.fixture
    def optimizer_history(self, optimizer: BaseSwarmOptimizer) -> BaseSwarmOptimizer:
        """Run the optimizer for 1000 iterations and return its instance"""
        optimizer.optimize(sphere, 1000)
        return optimizer

    @pytest.fixture
    def optimizer_reset(self, optimizer: BaseSwarmOptimizer) -> BaseSwarmOptimizer:
        """Reset the optimizer and return its instance"""
        optimizer.optimize(sphere, 10)
        optimizer.optimize(sphere, 10)
        optimizer.reset()
        return optimizer

    @pytest.fixture
    def options(self):
        """Default options dictionary for most PSO use-cases"""
        return {"c1": 0.3, "c2": 0.7, "w": 0.9}
    
    @pytest.fixture
    def position_updater(self):
        return PositionUpdater()

    @pytest.fixture
    def velocity_updater(self, options: SwarmOptions) -> VelocityUpdater:
        return VelocityUpdater(
            options,
            None,
            VelocityHandler.factory("unmodified")
        )

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
    def test_ftol_effect(self, optimizer: BaseSwarmOptimizer):
        """Test if setting the ftol breaks the optimization process"""
        optimizer.ftol = 1e-1
        optimizer.optimize(sphere, 2000)
        assert np.array(optimizer.cost_history).shape != (2000,)

    def test_parallel_evaluation(
        self,
        obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
        optimizer: BaseSwarmOptimizer,
    ):
        """Test if parallelization breaks the optimization process"""
        import multiprocessing

        optimizer.optimize(obj_without_args, 2000, n_processes=multiprocessing.cpu_count())
        assert np.array(optimizer.cost_history).shape == (2000,)

    def test_obj_with_kwargs(
        self,
        obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]],
        optimizer: BaseSwarmOptimizer,
    ):
        """Test if kwargs are passed properly in objfunc"""
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        optimizer.position_updater.bounds = bounds
        cost, pos = optimizer.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03), f"cost (={cost}) should be ~0"
        assert np.isclose(pos[0], 1.0, rtol=1e-03), f"pos[0] (={pos[0]}) should be ~1.0"
        assert np.isclose(pos[1], 1.0, rtol=1e-03), f"pos[1] (={pos[1]}) should be ~1.0"

    def test_obj_unnecessary_kwargs(
        self,
        obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]],
        optimizer: BaseSwarmOptimizer,
    ):
        """Test if error is raised given unnecessary kwargs"""
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            optimizer.optimize(obj_without_args, 1000, a=1)

    def test_obj_missing_kwargs(
        self,
        obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]],
        optimizer: BaseSwarmOptimizer,
    ):
        """Test if error is raised with incomplete kwargs"""
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            optimizer.optimize(obj_with_args, 1000, a=1)

    def test_obj_incorrect_kwargs(
        self,
        obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]],
        optimizer: BaseSwarmOptimizer,
    ):
        """Test if error is raised with wrong kwargs"""
        with pytest.raises(TypeError):
            # Wrong kwargs
            optimizer.optimize(obj_with_args, 1000, c=1, d=100)
