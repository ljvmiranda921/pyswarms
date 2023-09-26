#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.topology import Pyramid, Random, Ring, Star, VonNeumann
from pyswarms.backend.topology.base import Topology
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.base.base import BaseSwarmOptimizer
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_optimizer import ABCTestOptimizer


def istopology(x: object):
    """Helper predicate to check if it's a subclass"""
    return inspect.isclass(x) and not inspect.isabstract(x)


# Get all classes in the topology module, then
# Instatiate topologies, no need to suppy static param
topologies = [Pyramid(), Random(2), Ring(2, 2), Star(), VonNeumann(2, 1, 2)]


if TYPE_CHECKING:

    class FixtureRequest:
        param: Topology

else:
    FixtureRequest = Any


class TestGeneralOptimizer(ABCTestOptimizer):
    @pytest.fixture(params=topologies)
    def optimizer(  # type: ignore
        self, request: FixtureRequest, velocity_updater: VelocityUpdater, position_updater: PositionUpdater
    ):
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        position_updater.bounds = bounds
        velocity_updater.bounds = bounds
        return GeneralOptimizerPSO(
            10,
            2,
            request.param,
            velocity_updater,
            position_updater,
        )

    def test_ftol_effect(self, optimizer: BaseSwarmOptimizer):
        """Test if setting the ftol breaks the optimization process"""
        # Set optimizer tolerance
        optimizer.ftol = 1e-1
        optimizer.optimize(sphere, 2000)
        assert np.array(optimizer.cost_history).shape != (2000,)

    def test_parallel_evaluation(
        self, obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]], optimizer: BaseSwarmOptimizer
    ):
        """Test if parallelization breaks the optimization process"""
        import multiprocessing

        optimizer.optimize(obj_without_args, 2000, n_processes=multiprocessing.cpu_count())
        assert np.array(optimizer.cost_history).shape == (2000,)

    @pytest.mark.skip(reason="Some topologies converge too slowly")
    def test_obj_with_kwargs(
        self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: BaseSwarmOptimizer
    ):
        """Test if kwargs are passed properly in objfunc"""
        cost, pos = optimizer.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03)
        assert np.isclose(pos[0], 1.0, rtol=1e-03)
        assert np.isclose(pos[1], 1.0, rtol=1e-03)

    def test_obj_unnecessary_kwargs(
        self, obj_without_args: Callable[[npt.NDArray[Any]], npt.NDArray[Any]], optimizer: BaseSwarmOptimizer
    ):
        """Test if error is raised given unnecessary kwargs"""
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            optimizer.optimize(obj_without_args, 1000, a=1)

    def test_obj_missing_kwargs(
        self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: BaseSwarmOptimizer
    ):
        """Test if error is raised with incomplete kwargs"""
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            optimizer.optimize(obj_with_args, 1000, a=1)

    def test_obj_incorrect_kwargs(
        self, obj_with_args: Callable[[npt.NDArray[Any], int, int], npt.NDArray[Any]], optimizer: BaseSwarmOptimizer
    ):
        """Test if error is raised with wrong kwargs"""
        with pytest.raises(TypeError):
            # Wrong kwargs
            optimizer.optimize(obj_with_args, 1000, c=1, d=100)

    def test_general_correct_pos(self, optimizer: GeneralOptimizerPSO):
        """Test to check general optimiser returns the correct position corresponding to the best cost"""
        _, pos = optimizer.optimize(sphere, iters=5)
        # find best pos from history
        min_cost_idx = np.argmin(optimizer.cost_history)
        min_pos_idx = np.argmin(sphere(optimizer.pos_history[min_cost_idx]))
        assert np.array_equal(optimizer.pos_history[min_cost_idx][min_pos_idx], pos)

    def test_ftol_iter_effect(self, optimizer: GeneralOptimizerPSO):
        """Test if setting the ftol breaks the optimization process after the set number of iterations"""
        # Set optimizer tolerance
        optimizer.ftol = 1e-1
        optimizer.ftol_iter = 5
        optimizer.optimize(sphere, 2000)
        assert np.array(optimizer.cost_history).shape[0] >= optimizer.ftol_iter
