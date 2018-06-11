#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyswarms` package."""

# Import modules
import pytest
import numpy as np
from collections import namedtuple

# Import from package
from pyswarms.utils.functions import single_obj as fx

Bounds = namedtuple('Bounds', 'low high')
b = {
    # Define all bounds here
    'rastrigin' : Bounds(low=-5.12, high=5.12),
    'ackley'    : Bounds(low=-32, high=32),
    'beale'     : Bounds(low=-4.5, high=4.5),
    'goldstein' : Bounds(low=-2, high=-2),
    'booth'     : Bounds(low=-10, high=10),
    'matyas'    : Bounds(low=-10, high=10),
    'levi'      : Bounds(low=-10, high=10),
    'schaffer2' : Bounds(low=-100, high=100)

}

def test_rastrigin_bound_fail(outbound):
    """Test rastrigin bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['rastrigin'].low, b['rastrigin'].high, size=(3,2))
        fx.rastrigin_func(x)

def test_ackley_bound_fail(outbound):
    """Test ackley bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['ackley'].low, b['ackley'].high, size=(3,2))
        fx.ackley_func(x)

def test_beale_bound_fail(outbound):
    """Test beale bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['beale'].low, b['beale'].high, size=(3,2))
        fx.beale_func(x)

def test_goldstein_bound_fail(outbound):
    """Test goldstein bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['goldstein'].low, b['goldstein'].high, size=(3,2))
        fx.goldstein_func(x)

def test_booth_bound_fail(outbound):
    """Test booth bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['booth'].low, b['booth'].high, size=(3,2))
        fx.booth_func(x)

@pytest.mark.parametrize("x", [
    -np.random.uniform(15.001, 100, (3,2)),
    np.random.uniform(-5.001, -3.001, (3,2)),
    np.random.uniform(-3.001, -100, (3,2))
])
def test_bukin6_bound_fail(x):
    """Test bukin6 bound exception"""
    with pytest.raises(ValueError):
        fx.bukin6_func(x)

def test_matyas_bound_fail(outbound):
    """Test matyas bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['matyas'].low, b['matyas'].high, size=(3,2))
        fx.matyas_func(x)

def test_levi_bound_fail(outbound):
    """Test levi bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['levi'].low, b['levi'].high, size=(3,2))
        fx.levi_func(x)

def test_schaffer2_bound_fail(outbound):
    """Test schaffer2 bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b['schaffer2'].low, b['schaffer2'].high, tol=200, size=(3,2))
        fx.schaffer2_func(x)