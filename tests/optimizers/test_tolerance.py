#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import Any, Callable, List

import numpy as np
import numpy.typing as npt
import pytest
from loguru import logger

from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.topology import Star
from pyswarms.backend.topology.ring import Ring
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.optimizers import OptimizerPSO
from pyswarms.optimizers.base import BaseSwarmOptimizer
from pyswarms.utils.types import SwarmOptions

random.seed(0)

# Knapsack parameters
capacity = 50
number_of_items = 10
item_range = range(number_of_items)

# PARAMETERS
value = [random.randint(1, number_of_items) for _ in item_range]
weight = [random.randint(1, number_of_items) for _ in item_range]

# PSO parameters
n_particles = 10
iterations = 200
dimensions = number_of_items
LB = [0] * dimensions
UB = [1] * dimensions
constraints = (np.array(LB), np.array(UB))
kwargs = {"value": value, "weight": weight, "capacity": capacity}


# Instantiate optimizers
options = SwarmOptions({"c1": 2, "c2": 2, "w": 0.7})
velocity_updater = VelocityUpdater(options, (-0.5, 0.5), "invert", constraints)
position_updater = PositionUpdater(constraints, "periodic")

optimizers = [
    lambda: OptimizerPSO(n_particles, dimensions, Star(), velocity_updater, position_updater),
    lambda: OptimizerPSO(n_particles, dimensions, Ring(2, 3), velocity_updater, position_updater),
    lambda: OptimizerPSO(n_particles, dimensions, Star(), velocity_updater, position_updater),
]


def get_particle_obj(X: npt.NDArray[Any], **kwargs: Any):
    """Calculates the objective function value which is
    total revenue minus penalty of capacity violations"""
    # X is the decision variable. X is vector in the lenght of number of items
    # $ value of items
    value = kwargs["value"]
    # Total revenue
    x: List[int] = [value[i] * np.round(X[i]) for i in item_range]
    revenue = sum(x)
    # Total weight of selected items
    x = [kwargs["weight"][i] * np.round(X[i]) for i in item_range]
    used_capacity = sum(x)
    # Total capacity violation with 100 as a penalty cofficient
    capacity_violation = 100 * min(0, capacity - used_capacity)
    # the objective function minimizes the negative revenue, which is the same
    # as maximizing the positive revenue
    return -1 * (revenue + capacity_violation)


def objective_function(X: npt.NDArray[Any], **kwargs: Any):
    """Objective function with arguments"""
    n_particles_ = X.shape[0]
    dist = [get_particle_obj(X[i], **kwargs) for i in range(n_particles_)]
    return np.array(dist)


class TestToleranceOptions:
    @pytest.mark.parametrize("optimizer_func", optimizers)
    def test_no_ftol(self, optimizer_func: Callable[[], BaseSwarmOptimizer]):
        """Test complete run"""
        optimizer = optimizer_func()
        optimizer.optimize(objective_function, iterations, None, False, **kwargs)
        assert len(optimizer.cost_history) == iterations

    @pytest.mark.parametrize("optimizer_func", optimizers)
    def test_ftol_effect(self, optimizer_func: Callable[[], BaseSwarmOptimizer]):
        """Test early stopping with ftol"""
        optimizer = optimizer_func()
        optimizer.ftol = 0.01
        logger.critical(optimizer.ftol)
        # logger.critical(optimizer.__dict__)
        optimizer.optimize(objective_function, iterations, None, False, **kwargs)
        assert len(optimizer.cost_history) <= iterations

    @pytest.mark.parametrize("optimizer_func", optimizers)
    def test_ftol_iter_effect(self, optimizer_func: Callable[[], BaseSwarmOptimizer]):
        """Test early stopping with ftol and ftol_iter;
        must run for a minimum of ftol_iter iterations"""
        optimizer = optimizer_func()
        optimizer.ftol_iter = 50
        optimizer.optimize(objective_function, iterations, None, False, **kwargs)
        assert len(optimizer.cost_history) >= optimizer.ftol_iter
