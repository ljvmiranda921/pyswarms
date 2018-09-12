#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest

# Import from pyswarms
import pyswarms.backend as P


class TestComputePbest(object):
    """Test suite for compute_pbest()"""

    def test_return_values(self, swarm):
        """Test if method gives the expected return values"""
        expected_cost = np.array([1, 2, 2])
        expected_pos = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
        pos, cost = P.compute_pbest(swarm)
        assert (pos == expected_pos).all()
        assert (cost == expected_cost).all()

    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    def test_input_swarm(self, swarm):
        """Test if method raises AttributeError with wrong swarm"""
        with pytest.raises(AttributeError):
            P.compute_pbest(swarm)


class TestComputeVelocity(object):
    """Test suite for compute_velocity()"""

    @pytest.mark.parametrize("clamp", [None, (0, 1), (-1, 1)])
    def test_return_values(self, swarm, clamp):
        """Test if method gives the expected shape and range"""
        v = P.compute_velocity(swarm, clamp)
        assert v.shape == swarm.position.shape
        if clamp is not None:
            assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    def test_input_swarm(self, swarm):
        """Test if method raises AttributeError with wrong swarm"""
        with pytest.raises(AttributeError):
            P.compute_velocity(swarm, clamp=(0, 1))

    @pytest.mark.parametrize("options", [{"c1": 0.5, "c2": 0.3}])
    def test_missing_kwargs(self, swarm, options):
        """Test if method raises KeyError with missing kwargs"""
        with pytest.raises(KeyError):
            swarm.options = options
            clamp = (0, 1)
            P.compute_velocity(swarm, clamp)


class TestComputePosition(object):
    """Test suite for compute_position()"""

    @pytest.mark.parametrize(
        "bounds",
        [None, ([-5, -5, -5], [5, 5, 5]), ([-10, -10, -10], [10, 10, 10])],
    )
    def test_return_values(self, swarm, bounds):
        """Test if method gives the expected shape and range"""
        p = P.compute_position(swarm, bounds)
        assert p.shape == swarm.velocity.shape
        if bounds is not None:
            assert (bounds[0] <= p).all() and (bounds[1] >= p).all()

    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    def test_input_swarm(self, swarm):
        """Test if method raises AttributeError with wrong swarm"""
        with pytest.raises(AttributeError):
            P.compute_position(swarm, bounds=([-5, -5], [5, 5]))
