#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
from collections import namedtuple

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.utils.functions import single_obj as fx


def test_ackley_output(common_minima):
    """Tests ackley function output."""
    assert np.isclose(fx.ackley(common_minima), np.zeros(3)).all()


def test_beale_output(common_minima2):
    """Tests beale function output."""
    assert np.isclose(fx.beale([3, 0.5] * common_minima2), np.zeros(3)).all()


def test_booth_output(common_minima2):
    """Test booth function output."""
    assert np.isclose(fx.booth([1, 3] * common_minima2), np.zeros(3)).all()


def test_bukin6_output(common_minima2):
    """Test bukin function output."""
    assert np.isclose(fx.bukin6([-10, 1] * common_minima2), np.zeros(3)).all()


@pytest.mark.parametrize(
    "x",
    [
        np.array(
            [
                [1.34941, -1.34941],
                [1.34941, 1.34941],
                [-1.34941, 1.34941],
                [-1.34941, -1.34941],
            ]
        )
    ],
)
@pytest.mark.parametrize(
    "minima", [np.array([-2.06261, -2.06261, -2.06261, -2.06261])]
)
def test_crossintray_output(x, minima):
    """Tests crossintray function output."""
    assert np.isclose(fx.crossintray(x), minima).all()


def test_easom_output(common_minima2):
    """Tests easom function output."""
    assert np.isclose(
        fx.easom([np.pi, np.pi] * common_minima2), (-1 * np.ones(3))
    ).all()


def test_eggholder_output(common_minima2):
    """Tests eggholder function output."""
    assert np.isclose(
        fx.eggholder([512, 404.3219] * common_minima2),
        (-959.6407 * np.ones(3)),
    ).all()


def test_goldstein_output(common_minima2):
    """Tests goldstein-price function output."""
    assert np.isclose(
        fx.goldstein([0, -1] * common_minima2), (3 * np.ones(3))
    ).all()


@pytest.mark.parametrize(
    "x",
    [
        np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )
    ],
)
def test_himmelblau_output(x):
    """Tests himmelblau function output."""
    assert np.isclose(fx.himmelblau(x), np.zeros(4)).all()


@pytest.mark.parametrize(
    "x",
    [
        np.array(
            [
                [8.05502, 9.66459],
                [-8.05502, 9.66459],
                [8.05502, -9.66459],
                [-8.05502, -9.66459],
            ]
        )
    ],
)
@pytest.mark.parametrize(
    "minima", [np.array([-19.2085, -19.2085, -19.2085, -19.2085])]
)
def test_holdertable_output(x, minima):
    """Tests holdertable function output."""
    assert np.isclose(fx.holdertable(x), minima).all()


def test_levi_output(common_minima2):
    """Test levi function output."""
    assert np.isclose(fx.levi(common_minima2), np.zeros(3)).all()


def test_matyas_output(common_minima):
    """Test matyas function output."""
    assert np.isclose(fx.matyas(common_minima), np.zeros(3)).all()


def test_rastrigin_output(common_minima):
    """Tests rastrigin function output."""
    assert np.array_equal(fx.rastrigin(common_minima), np.zeros(3))


def test_rosenbrock_output(common_minima2):
    """Tests rosenbrock function output."""
    assert np.array_equal(
        fx.rosenbrock(common_minima2).all(), np.zeros(3).all()
    )


def test_schaffer2_output(common_minima):
    """Test schaffer2 function output."""
    assert np.isclose(fx.schaffer2(common_minima), np.zeros(3)).all()


def test_sphere_output(common_minima):
    """Tests sphere function output."""
    assert np.array_equal(fx.sphere(common_minima), np.zeros((3,)))


def test_threehump_output(common_minima):
    """Tests threehump function output."""
    assert np.array_equal(fx.threehump(common_minima), np.zeros(3))
