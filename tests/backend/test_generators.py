#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
import pyswarms.backend as P


@pytest.mark.parametrize('bounds', [None, ([2,2,2], [5,5,5]), ([-1,-1,0], [2,2,5])])
@pytest.mark.parametrize('center', [1, [3,3,3], [0.2,0.2,0.1]])
def test_generate_swarm_return_values(bounds, center):
    """Tests if generate_swarm() returns expected values"""
    pos = P.generate_swarm(n_particles=2, dimensions=3, bounds=bounds,
                           center=center)
    if bounds is None:
        min_bounds, max_bounds = (0.0, 1.00)
    else:
        min_bounds, max_bounds = bounds
    lower_bound = center * np.array(min_bounds)
    upper_bound = center * np.array(max_bounds)
    assert (pos <= upper_bound).all() and (pos >= lower_bound).all()

@pytest.mark.parametrize('clamp', [None, (0,1), (2,5), (1,6)])
def test_generate_velocity_return_values(clamp):
    """Tests if generate_velocity() returns expected values"""
    min_clamp, max_clamp = (0,1) if clamp == None else clamp
    velocity = P.generate_velocity(n_particles=2, dimensions=3, clamp=clamp)
    assert (velocity <= max_clamp).all() and (velocity >= min_clamp).all()