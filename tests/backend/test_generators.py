#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
import pyswarms.backend as P


@pytest.mark.parametrize(
    "bounds", [None, ([2, 2, 2], [5, 5, 5]), ([-1, -1, 0], [2, 2, 5])]
)
@pytest.mark.parametrize("center", [1, [3, 3, 3], [0.2, 0.2, 0.1]])
def test_generate_swarm_return_values(bounds, center):
    """Tests if generate_swarm() returns expected values"""
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


def test_generate_swarm_out_of_bounds():
    """Tests if generate_swarm() raises ValueError when initialized with the wrong value"""
    bounds = ([1, 1, 1], [5, 5, 5])
    init_pos = np.array([[-2, 3, 3], [6, 8, 1]])
    with pytest.raises(ValueError):
        pos = P.generate_swarm(
            n_particles=2, dimensions=3, bounds=bounds, init_pos=init_pos
        )


@pytest.mark.parametrize("binary", [False, True])
def test_generate_discrete_binary_swarm(binary):
    """Tests if generate_discrete_swarm(binary=True) returns expected values"""
    dims = 3
    pos = P.generate_discrete_swarm(
        n_particles=2, dimensions=dims, binary=binary
    )
    if binary:
        assert len(np.unique(pos)) <= 2  # Might generate pure 0 or 1
    else:
        assert (np.max(pos, axis=1) == dims - 1).all()


@pytest.mark.parametrize("init_pos", [None, np.array([[4, 2, 1], [1, 4, 6]])])
def test_generate_discrete_swarm(init_pos):
    """Tests if init_pos actually sets the position properly"""
    dims = 3
    pos = P.generate_discrete_swarm(
        n_particles=2, dimensions=dims, init_pos=init_pos
    )
    if init_pos is None:
        assert (np.max(pos, axis=1) == dims - 1).all()
    else:
        assert np.equal(pos, init_pos).all()


@pytest.mark.parametrize("clamp", [None, (0, 1), (2, 5), (1, 6)])
def test_generate_velocity_return_values(clamp):
    """Tests if generate_velocity() returns expected values"""
    min_clamp, max_clamp = (0, 1) if clamp == None else clamp
    velocity = P.generate_velocity(n_particles=2, dimensions=3, clamp=clamp)
    assert (velocity <= max_clamp).all() and (velocity >= min_clamp).all()
