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

<<<<<<< HEAD

@pytest.mark.parametrize("topology", [object(), int(), dict()])
def test_topology_type_exception(options, topology):
    """Tests if exceptions are thrown when the topology has the wrong type"""
    with pytest.raises(TypeError):
        GeneralOptimizerPSO(5, 2, options, topology)


@pytest.mark.parametrize(
    "bounds",
    [
        tuple(np.array([-5, -5])),
        (np.array([-5, -5, -5]), np.array([5, 5])),
        (np.array([-5, -5, -5]), np.array([5, 5, 5])),
    ],
)
def test_bounds_size_exception(bounds, options, topology):
    """Tests if exceptions are raised when bound sizes are wrong"""
    with pytest.raises(IndexError):
        GeneralOptimizerPSO(
            5, 2, options=options, topology=topology, bounds=bounds
        )
=======
from .abc_test_optimizer import ABCTestOptimizer
>>>>>>> upstream/refactor/general


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
            n_particles=100,
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
<<<<<<< HEAD

def test_reset_default_values(gbest_reset):
    """Tests if best cost and best pos are set properly when the reset()
    method is called"""
    assert gbest_reset.swarm.best_cost == np.inf
    assert set(gbest_reset.swarm.best_pos) == set(np.array([]))


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
def test_training_history_shape(gbest_history, history, expected_shape):
    """Test if training histories are of expected shape"""
    pso = vars(gbest_history)
    assert np.array(pso[history]).shape == expected_shape


def test_ftol_effect(options, topology):
    """Test if setting the ftol breaks the optimization process accordingly"""
    pso = GeneralOptimizerPSO(
        10, 2, options=options, topology=topology, ftol=1e-1
    )
    pso.optimize(sphere, 2000)
    assert np.array(pso.cost_history).shape != (2000,)
=======
        opt.optimize(sphere, 1000)
        opt.reset()
        return opt

    def test_ftol_effect(self, optimizer):
        """Test if setting the ftol breaks the optimization process"""
        # Set optimizer tolerance
        optimizer.ftol = 1e-1
        optimizer.optimize(sphere, 2000)
        assert np.array(optimizer.cost_history).shape != (2000,)

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
>>>>>>> upstream/refactor/general
