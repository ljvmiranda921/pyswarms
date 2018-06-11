#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np
from collections import namedtuple

# Import from package
from pyswarms.utils.functions import single_obj as fx

def test_sphere_output(common_minima):
    """Tests sphere function output."""
    assert np.array_equal(fx.sphere_func(common_minima), np.zeros((3,)))

def test_rastrigin_output(common_minima):
    """Tests rastrigin function output."""
    assert np.array_equal(fx.rastrigin_func(common_minima), np.zeros(3))

def test_ackley_output(common_minima):
    """Tests ackley function output."""
    assert np.isclose(fx.ackley_func(common_minima), np.zeros(3)).all()

def test_rosenbrock_output(common_minima2):
    """Tests rosenbrock function output."""
    assert np.array_equal(fx.rosenbrock_func(common_minima2).all(),np.zeros(3).all())

def test_beale_output(common_minima2):
    """Tests beale function output."""
    assert np.isclose(fx.beale_func([3, 0.5] * common_minima2), np.zeros(3)).all()

def test_goldstein_output(common_minima2):
    """Tests goldstein-price function output."""
    assert np.isclose(fx.goldstein_func([0, -1] * common_minima2), (3 * np.ones(3))).all()

def test_booth_output(common_minima2):
    """Test booth function output."""
    assert np.isclose(fx.booth_func([1, 3] * common_minima2), np.zeros(3)).all()

def test_bukin6_output(common_minima2):
    """Test bukin function output."""
    assert np.isclose(fx.bukin6_func([-10, 1] * common_minima2), np.zeros(3)).all()

def test_bukin6_output(common_minima):
    """Test bukin function output."""
    assert np.isclose(fx.matyas_func(common_minima), np.zeros(3)).all()

def test_levi_output(common_minima2):
    """Test levi function output."""
    assert np.isclose(fx.levi_func(common_minima2), np.zeros(3)).all()

def test_schaffer2_output(common_minima):
    """Test schaffer2 function output."""
    assert np.isclose(fx.schaffer2_func(common_minima), np.zeros(3)).all()