#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.single import (GlobalBestPSO, LocalBestPSO)
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.functions.single_obj import sphere_func

@pytest.fixture(scope='module')
def gbest_history():
    """Returns a GlobalBestPSO instance run for 1000 iterations for checking
    history"""
    pso = GlobalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5})
    pso.optimize(sphere_func, 1000, verbose=0)
    return pso

@pytest.fixture(scope='module')
def gbest_reset():
    """Returns a GlobalBestPSO instance that has been run and reset to check
    default value"""
    pso = GlobalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5})
    pso.optimize(sphere_func, 10, verbose=0)
    pso.reset()
    return pso

@pytest.fixture(scope='module')
def lbest_history():
    """Returns a LocalBestPSO instance run for 1000 iterations for checking
    history"""
    pso = LocalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k':2, 'p':2})
    pso.optimize(sphere_func, 1000, verbose=0)
    return pso

@pytest.fixture(scope='module')
def lbest_reset():
    """Returns a LocalBestPSO instance that has been run and reset to check
    default value"""
    pso = LocalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k': 2, 'p': 2})
    pso.optimize(sphere_func, 10, verbose=0)
    pso.reset()
    return pso

@pytest.fixture(scope='module')
def binary_history():
    """Returns a BinaryPSO instance run for 1000 iterations for checking
    history"""
    pso = BinaryPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k':2, 'p':2})
    pso.optimize(sphere_func, 1000, verbose=0)
    return pso

@pytest.fixture(scope='module')
def binary_reset():
    """Returns a BinaryPSO instance that has been run and reset to check
    default value"""
    pso = BinaryPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k': 2, 'p': 2})
    pso.optimize(sphere_func, 10, verbose=0)
    pso.reset()
    return pso

@pytest.fixture
def options():
    """Default options dictionary for most PSO use-cases"""
    options_ = {'c1':0.5, 'c2':0.7, 'w':0.5, 'k':2, 'p':2}
    return options_