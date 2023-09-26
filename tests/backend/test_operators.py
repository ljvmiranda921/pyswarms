#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
import pytest

import pyswarms.backend as P
from pyswarms.backend.handlers import BoundaryStrategy, VelocityHandler, VelocityStrategy
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.velocity import SwarmOptions, VelocityUpdater
from pyswarms.utils.types import Bounds, Clamp


class TestComputePbest(object):
    """Test suite for compute_pbest()"""

    def test_return_values(self, swarm: Swarm):
        """Test if method gives the expected return values"""
        expected_cost = np.array([1, 2, 2])
        expected_pos = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
        pos, cost = P.compute_pbest(swarm)
        assert (pos == expected_pos).all()
        assert (cost == expected_cost).all()

    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    def test_input_swarm(self, swarm: Swarm):
        """Test if method raises AttributeError with wrong swarm"""
        with pytest.raises(AttributeError):
            P.compute_pbest(swarm)


class TestComputeVelocity(object):
    """Test suite for compute_velocity()"""

    @pytest.mark.parametrize("clamp", [None, (0, 1), (-1, 1)])
    def test_return_values(self, swarm: Swarm, clamp: Optional[Clamp], options: SwarmOptions):
        """Test if method gives the expected shape and range"""
        vu = VelocityUpdater(options, clamp, "unmodified")
        v = vu.compute(swarm)
        assert v.shape == swarm.position.shape
        if clamp is not None:
            assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    @pytest.mark.parametrize("vh_strat", ["unmodified", "zero", "invert", "adjust"])
    def test_input_swarm(self, swarm: Swarm, vh_strat: VelocityStrategy, options: SwarmOptions):
        """Test if method raises AttributeError with wrong swarm"""
        vu = VelocityUpdater(options, None, "unmodified")
        with pytest.raises(AttributeError):
            vu.compute(swarm)

    @pytest.mark.parametrize("options", [{"c1": 0.5, "c2": 0.3}])
    @pytest.mark.parametrize("vh_strat", ["unmodified", "zero", "invert", "adjust"])
    def test_missing_kwargs(self, swarm: Swarm, options: SwarmOptions, vh_strat: VelocityStrategy):
        """Test if method raises KeyError with missing kwargs"""
        vu = VelocityUpdater(options, (0, 1), "unmodified")
        with pytest.raises(KeyError):
            vu.compute(swarm, 10)


class TestComputePosition(object):
    """Test suite for compute_position()"""

    @pytest.mark.parametrize(
        "bounds",
        [None, ([-5, -5, -5], [5, 5, 5]), ([-10, -10, -10], [10, 10, 10])],
    )
    @pytest.mark.parametrize("bh_strat", ["nearest", "random"])
    def test_return_values(self, swarm: Swarm, bounds: Optional[Bounds], bh_strat: BoundaryStrategy):
        """Test if method gives the expected shape and range"""
        position_updater = PositionUpdater(bounds, bh_strat)
        p = position_updater.compute(swarm)
        assert p.shape == swarm.velocity.shape
        if bounds is not None:
            assert (bounds[0] <= p).all() and (bounds[1] >= p).all()

    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    @pytest.mark.parametrize("bh_strat", ["nearest", "random", "shrink", "intermediate"])
    def test_input_swarm(self, swarm: Swarm, bh_strat: BoundaryStrategy):
        """Test if method raises AttributeError with wrong swarm"""
        position_updater = PositionUpdater(([-5, -5], [5, 5]), bh_strat)
        with pytest.raises(AttributeError):
            position_updater.compute(swarm)
