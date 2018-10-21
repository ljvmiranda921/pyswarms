#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyswarms` package."""

# Import standard library
from collections import namedtuple

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.utils.functions import single_obj as fx

Bounds = namedtuple("Bounds", "low high")
b = {
    # Define all bounds here
    "ackley": Bounds(low=-32, high=32),
    "beale": Bounds(low=-4.5, high=4.5),
    "booth": Bounds(low=-10, high=10),
    # bukin_6 is not symmetrical
    "crossintray": Bounds(low=-1, high=10),
    "easom": Bounds(low=-100, high=100),
    "eggholder": Bounds(low=-512, high=512),
    "goldstein": Bounds(low=-2, high=-2),
    "himmelblau": Bounds(low=-5, high=5),
    "holdertable": Bounds(low=-10, high=10),
    "levi": Bounds(low=-10, high=10),
    "matyas": Bounds(low=-10, high=10),
    "rastrigin": Bounds(low=-5.12, high=5.12),
    # rosenbrock has an infinite domain
    "schaffer2": Bounds(low=-100, high=100),
    "threehump": Bounds(low=-5, high=5),
}


def test_ackley_bound_fail(outbound):
    """Test ackley bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["ackley"].low, b["ackley"].high, size=(3, 2))
        fx.ackley(x)


def test_beale_bound_fail(outbound):
    """Test beale bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["beale"].low, b["beale"].high, size=(3, 2))
        fx.beale(x)


def test_booth_bound_fail(outbound):
    """Test booth bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["booth"].low, b["booth"].high, size=(3, 2))
        fx.booth(x)


@pytest.mark.parametrize(
    "x",
    [
        -np.random.uniform(15.001, 100, (3, 2)),
        np.random.uniform(-5.001, -3.001, (3, 2)),
        np.random.uniform(-3.001, -100, (3, 2)),
    ],
)
def test_bukin6_bound_fail(x):
    """Test bukin6 bound exception"""
    with pytest.raises(ValueError):
        fx.bukin6(x)


def test_crossintray_bound_fail(outbound):
    """Test crossintray bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["crossintray"].low, b["crossintray"].high, size=(3, 2))
        fx.crossintray(x)


def test_easom_bound_fail(outbound):
    """Test easom bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["easom"].low, b["easom"].high, size=(3, 2))
        fx.easom(x)


def test_eggholder_bound_fail(outbound):
    """Test eggholder bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["eggholder"].low, b["eggholder"].high, size=(3, 2))
        fx.eggholder(x)


def test_goldstein_bound_fail(outbound):
    """Test goldstein bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["goldstein"].low, b["goldstein"].high, size=(3, 2))
        fx.goldstein(x)


def test_himmelblau_bound_fail(outbound):
    """Test himmelblau bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["himmelblau"].low, b["himmelblau"].high, size=(3, 2))
        fx.himmelblau(x)


def test_holdertable_bound_fail(outbound):
    """Test holdertable bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["holdertable"].low, b["holdertable"].high, size=(3, 2))
        fx.holdertable(x)


def test_levi_bound_fail(outbound):
    """Test levi bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["levi"].low, b["levi"].high, size=(3, 2))
        fx.levi(x)


def test_matyas_bound_fail(outbound):
    """Test matyas bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["matyas"].low, b["matyas"].high, size=(3, 2))
        fx.matyas(x)


def test_rastrigin_bound_fail(outbound):
    """Test rastrigin bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["rastrigin"].low, b["rastrigin"].high, size=(3, 2))
        fx.rastrigin(x)


def test_schaffer2_bound_fail(outbound):
    """Test schaffer2 bound exception"""
    with pytest.raises(ValueError):
        x = outbound(
            b["schaffer2"].low, b["schaffer2"].high, tol=200, size=(3, 2)
        )
        fx.schaffer2(x)


def test_threehump_bound_fail(outbound):
    """Test threehump bound exception"""
    with pytest.raises(ValueError):
        x = outbound(b["threehump"].low, b["threehump"].high, size=(3, 2))
        fx.threehump(x)
