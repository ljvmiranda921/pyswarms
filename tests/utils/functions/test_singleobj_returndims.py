#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
from collections import namedtuple

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.utils.functions import single_obj as fx


def test_ackley_output_size(common_minima, targetdim):
    """Tests ackley output size."""
    assert fx.ackley(common_minima).shape == targetdim


def test_beale_output_size(common_minima, targetdim):
    """Tests beale output size."""
    assert fx.beale(common_minima).shape == targetdim


def test_booth_output_size(common_minima, targetdim):
    """Test booth output size."""
    assert fx.booth(common_minima).shape == targetdim


def test_bukin6_output_size(common_minima2, targetdim):
    """Test bukin6 output size."""
    assert fx.bukin6([-10, 0] * common_minima2).shape == targetdim


def test_crossintray_output_size(common_minima2, targetdim):
    """Test crossintray output size."""
    assert fx.crossintray([-10, 0] * common_minima2).shape == targetdim


def test_easom_output_size(common_minima2, targetdim):
    """Test easom output size."""
    assert fx.easom([-10, 0] * common_minima2).shape == targetdim


def test_eggholder_output_size(common_minima2, targetdim):
    """Test eggholder output size."""
    assert fx.eggholder([-10, 0] * common_minima2).shape == targetdim


def test_goldstein_output_size(common_minima, targetdim):
    """Test goldstein output size."""
    assert fx.goldstein(common_minima).shape == targetdim


def test_himmelblau_output_size(common_minima, targetdim):
    """Test himmelblau output size."""
    assert fx.himmelblau(common_minima).shape == targetdim


def test_holdertable_output_size(common_minima, targetdim):
    """Test holdertable output size."""
    assert fx.holdertable(common_minima).shape == targetdim


def test_levi_output_size(common_minima, targetdim):
    """Test levi output size."""
    assert fx.levi(common_minima).shape == targetdim


def test_rastrigin_output_size(common_minima, targetdim):
    """Tests rastrigin output size."""
    assert fx.rastrigin(common_minima).shape == targetdim


def test_rosenbrock_output_size(common_minima, targetdim):
    """Tests rosenbrock output size."""
    assert fx.rosenbrock(common_minima).shape == targetdim


def test_schaffer2_output_size(common_minima, targetdim):
    """Test schaffer2 output size."""
    assert fx.schaffer2(common_minima).shape == targetdim


def test_sphere_output_size(common_minima, targetdim):
    """Tests sphere output size."""
    assert fx.sphere(common_minima).shape == targetdim


def test_threehump_output_size(common_minima, targetdim):
    """Test threehump output size."""
    assert fx.threehump(common_minima).shape == targetdim
