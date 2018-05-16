#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import pytest
import numpy as np
from collections import namedtuple

@pytest.fixture
def outbound():

    def _outbound(low, high, size, tol=100, nums=100):
        """Generates a matrix that is out of bounds"""
        low_end = - np.random.uniform(tol, low, (nums,))
        high_end = np.random.uniform(tol, high, (nums,))
        choices = np.hstack([low_end, high_end])
        return np.random.choice(choices, size=size, replace=True)

    return _outbound

@pytest.fixture
def outdim():
    return np.zeros(shape=(3,3))

@pytest.fixture
def common_minima():
    return np.zeros(shape=(3,2))

@pytest.fixture
def common_minima2():
    return np.ones(shape=(3,2))

@pytest.fixture
def targetdim():
    return (3, )