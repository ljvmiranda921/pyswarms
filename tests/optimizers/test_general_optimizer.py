#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
import inspect

# Import modules
import numpy as np
import pytest

# Import from pyswarms
import pyswarms.backend.topology as t
from pyswarms.single import GeneralOptimizerPSO
from pyswarms.utils.functions.single_obj import sphere

from .abc_test_optimizer import ABCTestOptimizer


def istopology(x):
    """Helper predicate to check if it's a subclass"""
    return inspect.isclass(x) and not inspect.isabstract(x)


# Get all classes in the topology module, then
# Instatiate topologies, no need to suppy static param
topologies = [topo() for _, topo in inspect.getmembers(t, istopology)]


class TestGeneralOptimizer(ABCTestOptimizer):
    @pytest.fixture(params=topologies)
    def optimizer(self, request, options):
        x_max = 10 * np.ones(2)
        x_min = -1 * x_max
        bounds = (x_min, x_max)
        return GeneralOptimizerPSO(
            n_particles=10,
            dimensions=2,
            options=options,
            bounds=bounds,
            topology=request.param,
        )

    @pytest.fixture(params=topologies)
    def optimizer_history(self, request, options):
        opt = GeneralOptimizerPSO(
            n_particles=10,
            dimensions=2,
            options=options,
            topology=request.param,
        )
        opt.optimize(sphere, 1000)
        return opt

    @pytest.fixture(params=topologies)
    def optimizer_reset(self, request, options):
        opt = GeneralOptimizerPSO(
            n_particles=10,
            dimensions=2,
            options=options,
            topology=request.param,
        )
        opt.optimize(sphere, 1000)
        opt.reset()
        return opt

    def test_ftol_effect(self, optimizer):
        """Test if setting the ftol breaks the optimization process"""
        # Set optimizer tolerance
        optimizer.ftol = 1e-1
        optimizer.optimize(sphere, 2000)
        assert np.array(optimizer.cost_history).shape != (2000,)

    def test_parallel_evaluation(self, obj_without_args, optimizer):
        """Test if parallelization breaks the optimization process"""
        import multiprocessing
        optimizer.optimize(obj_without_args, 2000, n_processes=multiprocessing.cpu_count())
        assert np.array(optimizer.cost_history).shape == (2000,)

    @pytest.mark.skip(reason="Some topologies converge too slowly")
    def test_obj_with_kwargs(self, obj_with_args, optimizer):
        """Test if kwargs are passed properly in objfunc"""
        cost, pos = optimizer.optimize(obj_with_args, 1000, a=1, b=100)
        assert np.isclose(cost, 0, rtol=1e-03)
        assert np.isclose(pos[0], 1.0, rtol=1e-03)
        assert np.isclose(pos[1], 1.0, rtol=1e-03)

    def test_obj_unnecessary_kwargs(self, obj_without_args, optimizer):
        """Test if error is raised given unnecessary kwargs"""
        with pytest.raises(TypeError):
            # kwargs `a` should not be supplied
            cost, pos = optimizer.optimize(obj_without_args, 1000, a=1)

    def test_obj_missing_kwargs(self, obj_with_args, optimizer):
        """Test if error is raised with incomplete kwargs"""
        with pytest.raises(TypeError):
            # kwargs `b` is missing here
            cost, pos = optimizer.optimize(obj_with_args, 1000, a=1)

    def test_obj_incorrect_kwargs(self, obj_with_args, optimizer):
        """Test if error is raised with wrong kwargs"""
        with pytest.raises(TypeError):
            # Wrong kwargs
            cost, pos = optimizer.optimize(obj_with_args, 1000, c=1, d=100)

    def test_general_correct_pos(self, options, optimizer):
        """ Test to check general optimiser returns the correct position corresponding to the best cost """
        cost, pos = optimizer.optimize(sphere, iters=5)
        # find best pos from history
        min_cost_idx = np.argmin(optimizer.cost_history)
        min_pos_idx = np.argmin(sphere(optimizer.pos_history[min_cost_idx]))
        assert np.array_equal(optimizer.pos_history[min_cost_idx][min_pos_idx], pos)
