#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard libraries
import pytest
import random

random.seed(0)

# Import modules
import numpy as np

# Import from pyswarms
from pyswarms.backend.topology import Star
from pyswarms.single import GlobalBestPSO, LocalBestPSO, GeneralOptimizerPSO

# Knapsack parameters
capacity = 50
number_of_items = 10
item_range = range(number_of_items)

# PARAMETERS
value = [random.randint(1, number_of_items) for i in item_range]
weight = [random.randint(1, number_of_items) for i in item_range]

# PSO parameters
n_particles = 10
iterations = 1000
options = {"c1": 2, "c2": 2, "w": 0.7, "k": 3, "p": 2}
dim = number_of_items
LB = [0] * dim
UB = [1] * dim
constraints = (np.array(LB), np.array(UB))
kwargs = {"value": value, "weight": weight, "capacity": capacity}


def get_particle_obj(X, **kwargs):
    """Calculates the objective function value which is
    total revenue minus penalty of capacity violations"""
    # X is the decision variable. X is vector in the lenght of number of items
    # $ value of items
    value = kwargs["value"]
    # weight of items
    weight = kwargs["weight"]
    # Total revenue
    revenue = sum([value[i] * np.round(X[i]) for i in item_range])
    # Total weight of selected items
    used_capacity = sum(
        [kwargs["weight"][i] * np.round(X[i]) for i in item_range]
    )
    # Total capacity violation with 100 as a penalty cofficient
    capacity_violation = 100 * min(0, capacity - used_capacity)
    # the objective function minimizes the negative revenue, which is the same
    # as maximizing the positive revenue
    return -1 * (revenue + capacity_violation)


def objective_function(X, **kwargs):
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


class TestToleranceOptions:
    @pytest.fixture(params=optimizers)
    def optimizer(self, request):
        global parameters
        if request.param.__name__ == "GeneralOptimizerPSO":
            return request.param, {**parameters, **{"topology": Star()}}
        return request.param, parameters

    def test_no_ftol(self, optimizer):
        """Test complete run"""
        optm, params = optimizer
        opt = optm(**params)
        opt.optimize(objective_function, iters=iterations, **kwargs)
        assert len(opt.cost_history) == iterations

    def test_ftol_effect(self, optimizer):
        """Test early stopping with ftol"""
        optm, params = optimizer
        params["ftol"] = 0.01
        opt = optm(**params)
        opt.optimize(objective_function, iters=iterations, **kwargs)
        assert len(opt.cost_history) <= iterations

    def test_ftol_iter_assertion(self, optimizer):
        """Assert ftol_iter type and value"""
        with pytest.raises(AssertionError):
            optm, params = optimizer
            params["ftol_iter"] = 0
            opt = optm(**params)

    def test_ftol_iter_effect(self, optimizer):
        """Test early stopping with ftol and ftol_iter;
        must run for a minimum of ftol_iter iterations"""
        optm, params = optimizer
        params["ftol_iter"] = 50
        opt = optm(**params)
        opt.optimize(objective_function, iters=iterations, **kwargs)
        assert len(opt.cost_history) >= opt.ftol_iter
