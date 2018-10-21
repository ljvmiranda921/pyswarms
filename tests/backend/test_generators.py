#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest

# Import from pyswarms
import pyswarms.backend as P


class TestGenerateSwarm(object):
    """Test suite for generate_swarm() method"""

    @pytest.mark.parametrize(
        "bounds", [None, ([2, 2, 2], [5, 5, 5]), ([-1, -1, 0], [2, 2, 5])]
    )
    @pytest.mark.parametrize("center", [1, [3, 3, 3], [0.2, 0.2, 0.1]])
    def test_return_values(self, bounds, center):
        """Test if method returns expected values"""
        pos = P.generate_swarm(
            n_particles=2, dimensions=3, bounds=bounds, center=center
        )
        if bounds is None:
            min_bounds, max_bounds = (0.0, 1.00)
        else:
            min_bounds, max_bounds = bounds
        lower_bound = center * np.array(min_bounds)
        upper_bound = center * np.array(max_bounds)
        assert (pos <= upper_bound).all() and (pos >= lower_bound).all()

    def test_out_of_bounds(self):
        """Test if method raises ValueError when initialized with the wrong value"""
        bounds = ([1, 1, 1], [5, 5, 5])
        init_pos = np.array([[-2, 3, 3], [6, 8, 1]])
        with pytest.raises(ValueError):
            P.generate_swarm(
                n_particles=2, dimensions=3, bounds=bounds, init_pos=init_pos
            )

    @pytest.mark.parametrize("bounds", [0.1])
    def test_bounds_wrong_type(self, bounds):
        """Test if method raises TypeError when bounds is not an array"""
        with pytest.raises(TypeError):
            P.generate_swarm(n_particles=2, dimensions=3, bounds=bounds)

    @pytest.mark.parametrize(
        "bounds", [(1, 1, 1), ([1, 1, 1]), ([1, 1, 1], [2, 2])]
    )
    def test_bounds_wrong_size(self, bounds):
        """Test if method raises ValueError when bounds is of wrong shape"""
        with pytest.raises(ValueError):
            P.generate_swarm(n_particles=2, dimensions=3, bounds=bounds)


class TestDiscreteSwarm(object):
    """Test suite for generate_discrete_swarm() method"""

    @pytest.mark.parametrize("binary", [False, True])
    def test_generate_discrete_binary_swarm(self, binary):
        """Test if binary=True returns expected values"""
        dims = 3
        pos = P.generate_discrete_swarm(
            n_particles=2, dimensions=dims, binary=binary
        )
        if binary:
            assert len(np.unique(pos)) <= 2  # Might generate pure 0 or 1
        else:
            assert (np.max(pos, axis=1) == dims - 1).all()

    def test_not_binary_error_discrete_swarm(self):
        """Test if method raises ValueError given wrong init_pos val"""
        init_pos = [0, 1, 2]
        with pytest.raises(ValueError):
            P.generate_discrete_swarm(
                n_particles=2, dimensions=3, binary=True, init_pos=init_pos
            )

    @pytest.mark.parametrize(
        "init_pos", [None, np.array([[4, 2, 1], [1, 4, 6]])]
    )
    def test_generate_discrete_swarm(self, init_pos):
        """Test if init_pos actually sets the position properly"""
        dims = 3
        pos = P.generate_discrete_swarm(
            n_particles=2, dimensions=dims, init_pos=init_pos
        )
        if init_pos is None:
            assert (np.max(pos, axis=1) == dims - 1).all()
        else:
            assert np.equal(pos, init_pos).all()


class TestGenerateVelocity(object):
    """Test suite for generate_velocity()"""

    @pytest.mark.parametrize("clamp", [None, (0, 1), (2, 5), (1, 6)])
    def test_return_values(self, clamp):
        """Test if the method returns expected values"""
        min_clamp, max_clamp = (0, 1) if clamp is None else clamp
        velocity = P.generate_velocity(
            n_particles=2, dimensions=3, clamp=clamp
        )
        assert (velocity <= max_clamp).all() and (velocity >= min_clamp).all()

    @pytest.mark.parametrize("clamp", [(0, 2, 5), [1, 3, 5]])
    def test_invalid_clamp_value(self, clamp):
        """Test if the method raises a ValueError given invalid clamp size"""
        with pytest.raises(ValueError):
            P.generate_velocity(n_particles=2, dimensions=3, clamp=clamp)

    @pytest.mark.parametrize("clamp", [0, 1])
    def test_invalid_clamp_type(self, clamp):
        """Test if method raises a TypeError given invalid clamp type"""
        with pytest.raises(TypeError):
            P.generate_velocity(n_particles=2, dimensions=3, clamp=clamp)
