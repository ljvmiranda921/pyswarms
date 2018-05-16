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
    pso = GlobalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5})
    pso.optimize(sphere_func, 1000, verbose=0)
    return pso

@pytest.fixture(scope='module')
def gbest_reset():
    pso = GlobalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5})
    pso.optimize(sphere_func, 10, verbose=0)
    pso.reset()
    return pso

@pytest.fixture(scope='module')
def lbest_history():
    pso = LocalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k':2, 'p':2})
    pso.optimize(sphere_func, 1000, verbose=0)
    return pso

@pytest.fixture(scope='module')
def lbest_reset():
    pso = LocalBestPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k': 2, 'p': 2})
    pso.optimize(sphere_func, 10, verbose=0)
    pso.reset()
    return pso

@pytest.fixture(scope='module')
def binary_history():
    pso = BinaryPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k':2, 'p':2})
    pso.optimize(sphere_func, 1000, verbose=0)
    return pso

@pytest.fixture(scope='module')
def binary_reset():
    pso = BinaryPSO(10, 2, {'c1': 0.5, 'c2': 0.7, 'w': 0.5, 'k': 2, 'p': 2})
    pso.optimize(sphere_func, 10, verbose=0)
    pso.reset()
    return pso

@pytest.fixture
def options():
    options_ = {'c1':0.5, 'c2':0.7, 'w':0.5, 'k':2, 'p':2}
    return options_