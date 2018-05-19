# -*- coding: utf-8 -*-

"""
Swarm Generation Backend

This module abstracts how a swarm is generated. You can see its
implementation in our base classes. In addition, you can use all the methods
here to dictate how a swarm is initialized for your custom PSO.

"""

# Import modules
import numpy as np


def generate_swarm(n_particles, dimensions, bounds=None, init_pos=1.00):
    """Generates a swarm

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm
    bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    init_pos : :code:`numpy.ndarrray` (default is :code:`1`)
        a list of initial positions for generating the swarm

    Returns
    -------
    numpy.ndarray
        swarm matrix of shape (n_particles, n_dimensions)
    """
    min_bounds, max_bounds = (0.0, 1.0)
    try:
        if bounds is not None:
            lb, ub = bounds
            min_bounds = np.repeat(np.array(lb)[np.newaxis, :], n_particles, axis=0)
            max_bounds = np.repeat(np.array(ub)[np.newaxis, :], n_particles, axis=0)
        pos = init_pos * np.random.uniform(low=min_bounds, high=max_bounds,
                                           size=(n_particles, dimensions))
    except ValueError:
        raise
    else:
        return pos

def generate_velocity(n_particles, dimensions, clamp=None):
    """Initializes a velocity vector

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm
    clamp : tuple of floats (default is :code:`(1,0)`)
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.

    Returns
    -------
    numpy.ndarray
        velocity matrix of shape (n_particles, dimensions)
    """
    try:
        min_velocity, max_velocity = (0,1) if clamp==None else clamp
        velocity = ((max_velocity - min_velocity) 
                 * np.random.random_sample(size=(n_particles, dimensions)) 
                 + min_velocity)
    except (ValueError, TypeError):
        raise
    else:
        return velocity