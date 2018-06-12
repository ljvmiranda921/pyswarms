#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fixtures for tests"""

# Import modules
import os
import pytest
import numpy as np
from mock import Mock
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')

# Import from package
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.environments import PlotEnvironment
from pyswarms.utils.functions.single_obj import sphere_func

@pytest.fixture
def trained_optimizer():
    """Returns a trained optimizer instance with 100 iterations"""
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    optimizer.optimize(sphere_func, iters=100)
    return optimizer
