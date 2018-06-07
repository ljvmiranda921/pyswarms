#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.discrete import BinaryPSO
from pyswarms.utils.functions.single_obj import sphere_func

@pytest.mark.parametrize('options', [
    {'c2':0.7, 'w':0.5, 'k': 2, 'p': 2},
    {'c1':0.5, 'w':0.5, 'k': 2, 'p': 2},
    {'c1':0.5, 'c2':0.7, 'k': 2, 'p': 2},
    {'c1':0.5, 'c2':0.7, 'w':0.5, 'p': 2},
    {'c1':0.5, 'c2':0.7, 'w':0.5, 'k': 2}
])
def test_keyword_exception(options):
    """Tests if exceptions are thrown when keywords are missing"""
    with pytest.raises(KeyError):
        BinaryPSO(5, 2, options)

@pytest.mark.parametrize('options', [
    {'c1':0.5, 'c2':0.7, 'w':0.5, 'k':-1, 'p':2},
    {'c1':0.5, 'c2':0.7, 'w':0.5, 'k':6, 'p':2},
    {'c1':0.5, 'c2':0.7, 'w':0.5, 'k':2, 'p':5}
])
def test_invalid_k_or_p_values(options):
    """Tests if exception is thrown when passing
    an invalid value for k or p"""
    with pytest.raises(ValueError):
        BinaryPSO(5, 2, options)

@pytest.mark.parametrize('velocity_clamp', [[1, 3],np.array([1, 3])])
def test_vclamp_type_exception(velocity_clamp, options):
    """Tests if exception is raised when velocity_clamp type is not a
    tuple"""
    with pytest.raises(TypeError):
        BinaryPSO(5, 2, velocity_clamp=velocity_clamp, options=options)

@pytest.mark.parametrize('velocity_clamp', [(1,1,1), (2,3,1)])
def test_vclamp_shape_exception(velocity_clamp, options):
    """Tests if exception is raised when velocity_clamp's size is not equal
    to 2"""
    with pytest.raises(IndexError):
        BinaryPSO(5, 2, velocity_clamp=velocity_clamp, options=options)

@pytest.mark.parametrize('velocity_clamp', [(3,2),(10,8)])
def test_vclamp_maxmin_exception(velocity_clamp, options):
    """Tests if the max velocity_clamp is less than min velocity_clamp and
    vice-versa"""
    with pytest.raises(ValueError):
        BinaryPSO(5, 2, velocity_clamp=velocity_clamp, options=options)

def test_reset_default_values(binary_reset):
    """Tests if best cost and best pos are set properly when the reset()
    method is called"""
    assert binary_reset.swarm.best_cost == np.inf
    assert set(binary_reset.swarm.best_pos) == set(np.array([]))

def test_training_history_shape(binary_history):
    """Test if training histories are of expected shape"""
    assert binary_history.get_cost_history.shape == (1000,)
    assert binary_history.get_mean_pbest_history.shape == (1000,)
    assert binary_history.get_mean_neighbor_history.shape == (1000,)
    assert binary_history.get_pos_history.shape == (1000, 10, 2)
    assert binary_history.get_velocity_history.shape == (1000, 10, 2)
