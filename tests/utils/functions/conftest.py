#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import numpy as np
import pytest


@pytest.fixture
def outbound():
    """Returns a function that generates a matrix out of bounds a given
    range"""

    def _outbound(low, high, size, tol=1000, nums=100):
        """Generates a matrix that is out of bounds"""
        low_end = -np.random.uniform(tol, low, (nums,))
        high_end = np.random.uniform(tol, high, (nums,))
        choices = np.hstack([low_end, high_end])
        return np.random.choice(choices, size=size, replace=True)

    return _outbound


@pytest.fixture
def outdim():
    """Returns a matrix of bad shape (3D matrix)"""
    return np.zeros(shape=(3, 3))


@pytest.fixture
def common_minima():
    """Returns a zero-matrix with a common-minima for most objective
    functions"""
    return np.zeros(shape=(3, 2))


@pytest.fixture
def common_minima2():
    """Returns a one-matrix with a common-minima for most objective
    functions"""
    return np.ones(shape=(3, 2))


@pytest.fixture
def targetdim():
    """Returns a baseline target dimension for most objective functions"""
    return (3,)
