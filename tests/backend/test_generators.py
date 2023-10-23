#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import numpy as np
import pytest

from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.utils.types import Bounds, Clamp, Position


class TestGenerateSwarm(object):
    """Test suite for generate_swarm() method"""

    @pytest.mark.parametrize("bounds", [None, (2, 5), ([2, 2, 2], [5, 5, 5]), ([-1, -1, 0], [2, 2, 5])])
    @pytest.mark.parametrize("center", [1, [3, 3, 3], [0.2, 0.2, 0.1]])
    def test_return_values(self, bounds: Optional[Bounds], center: float | Position):
        """Test if method returns expected values"""
        position_updater = PositionUpdater(bounds)
        pos = position_updater.generate_position(n_particles=2, dimensions=3, center=center)
        if bounds is None:
            min_bounds, max_bounds = (0.0, 1.00)
        else:
            min_bounds, max_bounds = bounds
        lower_bound = center * np.array(min_bounds)
        upper_bound = center * np.array(max_bounds)
        assert (pos <= upper_bound).all() and (pos >= lower_bound).all()

    def test_out_of_bounds(self):
        """Test if method raises ValueError when initialized with the wrong value"""
        bounds = ((1, 1, 1), (5, 5, 5))
        init_pos = np.array([[-2, 3, 3], [6, 8, 1]])
        position_updater = PositionUpdater(bounds)
        with pytest.raises(ValueError):
            position_updater.generate_position(n_particles=2, dimensions=3, init_pos=init_pos)

    @pytest.mark.parametrize("bounds", [0.1])
    def test_bounds_wrong_type(self, bounds: Bounds):
        """Test if method raises TypeError when bounds is not an array"""
        with pytest.raises(TypeError):
            PositionUpdater(bounds)

    @pytest.mark.parametrize("bounds", [(1, 1, 1), ([1, 1, 1]), ([1, 1, 1], [2, 2])])
    def test_bounds_wrong_size(self, bounds: Bounds):
        """Test if method raises ValueError when bounds is of wrong shape"""
        with pytest.raises((ValueError, AssertionError)):
            position_updater = PositionUpdater(bounds)
            position_updater.generate_position(n_particles=2, dimensions=3)


class TestDiscreteSwarm(object):
    """Test suite for generate_discrete_swarm() method"""

    @pytest.mark.parametrize("binary", [False, True])
    def test_generate_discrete_binary_swarm(self, binary: bool):
        """Test if binary=True returns expected values"""
        dims = 3
        position_updater = PositionUpdater()
        pos = position_updater.generate_discrete_position(n_particles=2, dimensions=dims, binary=binary)
        if binary:
            assert len(np.unique(pos)) <= 2  # Might generate pure 0 or 1
        else:
            assert (np.max(pos, axis=1) == dims - 1).all()

    def test_not_binary_error_discrete_swarm(self):
        """Test if method raises ValueError given wrong init_pos val"""
        init_pos = np.array([0, 1, 2])
        position_updater = PositionUpdater()
        with pytest.raises(ValueError):
            position_updater.generate_discrete_position(n_particles=2, dimensions=3, binary=True, init_pos=init_pos)

    @pytest.mark.parametrize("init_pos", [None, np.array([[4, 2, 1], [1, 4, 6]])])
    def test_generate_discrete_swarm(self, init_pos: Optional[Position]):
        """Test if init_pos actually sets the position properly"""
        dims = 3
        position_updater = PositionUpdater()
        pos = position_updater.generate_discrete_position(n_particles=2, dimensions=dims, init_pos=init_pos)
        if init_pos is None:
            assert (np.max(pos, axis=1) == dims - 1).all()
        else:
            assert np.equal(pos, init_pos).all()


class TestGenerateVelocity(object):
    """Test suite for generate_velocity()"""

    @pytest.mark.parametrize("clamp", [None, (0, 1), (2, 5), (1, 6)])
    def test_return_values(self, clamp: Optional[Clamp]):
        """Test if the method returns expected values"""
        min_clamp, max_clamp = (0, 1) if clamp is None else clamp
        velocity_updater = VelocityUpdater({"c1": 0, "c2": 0, "w": 0}, clamp)
        velocity = velocity_updater.generate_velocity(n_particles=2, dimensions=3)
        assert (velocity <= max_clamp).all() and (velocity >= min_clamp).all()

    @pytest.mark.parametrize("clamp", [(0, 2, 5), [1, 3, 5]])
    def test_invalid_clamp_value(self, clamp: Clamp):
        """Test if the method raises a ValueError given invalid clamp size"""
        velocity_updater = VelocityUpdater({"c1": 0, "c2": 0, "w": 0}, clamp)
        with pytest.raises(ValueError):
            velocity_updater.generate_velocity(n_particles=2, dimensions=3)

    @pytest.mark.parametrize("clamp", [0, 1])
    def test_invalid_clamp_type(self, clamp: Clamp):
        """Test if method raises a TypeError given invalid clamp type"""
        velocity_updater = VelocityUpdater({"c1": 0, "c2": 0, "w": 0}, clamp)
        with pytest.raises(TypeError):
            velocity_updater.generate_velocity(n_particles=2, dimensions=3)
