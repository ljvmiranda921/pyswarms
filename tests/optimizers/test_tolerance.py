#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type

import numpy as np
import numpy.typing as npt
import pytest

from pyswarms.backend.topology import Star
from pyswarms.base.base import BaseSwarmOptimizer
from pyswarms.single import GeneralOptimizerPSO, GlobalBestPSO, LocalBestPSO

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
iterations = 1000
options = {"c1": 2, "c2": 2, "w": 0.7, "k": 3, "p": 2}
dim = number_of_items
LB = [0] * dim
UB = [1] * dim
constraints = (np.array(LB), np.array(UB))
kwargs = {"value": value, "weight": weight, "capacity": capacity}


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


# Instantiate optimizers
optimizers = [GlobalBestPSO, LocalBestPSO, GeneralOptimizerPSO]
parameters = dict(
    n_particles=n_particles,
    dimensions=dim,
    options=options,
    bounds=constraints,
    bh_strategy="periodic",
    velocity_clamp=(-0.5, 0.5),
    vh_strategy="invert",
)


if TYPE_CHECKING:

    class FixtureRequest:
        param: Type[BaseSwarmOptimizer]

else:
    FixtureRequest = Any


class TestToleranceOptions:
    @pytest.fixture(params=optimizers)
    def optimizer(self, request: FixtureRequest):
        global parameters
        if request.param.__name__ == "GeneralOptimizerPSO":
            return request.param, {**parameters, **{"topology": Star()}}
        return request.param, parameters

    def test_no_ftol(self, optimizer: Tuple[Type[BaseSwarmOptimizer], Dict[str, Any]]):
        """Test complete run"""
        optm, params = optimizer
        opt = optm(**params)
        opt.optimize(objective_function, iters=iterations, n_processes=None, **kwargs)
        assert len(opt.cost_history) == iterations

    def test_ftol_effect(self, optimizer: Tuple[Type[BaseSwarmOptimizer], Dict[str, Any]]):
        """Test early stopping with ftol"""
        optm, params = optimizer
        params["ftol"] = 0.01
        opt = optm(**params)
        opt.optimize(objective_function, iters=iterations, n_processes=None, **kwargs)
        assert len(opt.cost_history) <= iterations

    def test_ftol_iter_assertion(self, optimizer: Tuple[Type[BaseSwarmOptimizer], Dict[str, Any]]):
        """Assert ftol_iter type and value"""
        with pytest.raises(AssertionError):
            optm, params = optimizer
            params["ftol_iter"] = 0
            optm(**params)

    def test_ftol_iter_effect(self, optimizer: Tuple[Type[BaseSwarmOptimizer], Dict[str, Any]]):
        """Test early stopping with ftol and ftol_iter;
        must run for a minimum of ftol_iter iterations"""
        optm, params = optimizer
        params["ftol_iter"] = 50
        opt = optm(**params)
        opt.optimize(objective_function, iters=iterations, n_processes=None, **kwargs)
        assert len(opt.cost_history) >= opt.ftol_iter
