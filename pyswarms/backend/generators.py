# -*- coding: utf-8 -*-

"""
Swarm Generation Backend

This module abstracts how a swarm is generated. You can see its
implementation in our base classes. In addition, you can use all the methods
here to dictate how a swarm is initialized for your custom PSO.

"""

# Import modules
import numpy as np

# Import from package
from .swarms import Swarm


def generate_swarm(
    n_particles, dimensions, bounds=None, center=1.00, init_pos=None
):
    """Generate a swarm

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
    center : :code:`numpy.ndarray` or :code:`float` (default is :code:`1`)
        controls the mean or center whenever the swarm is generated randomly.
    init_pos : :code:`numpy.ndarray` (default is :code:`None`)
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.

    Returns
    -------
    numpy.ndarray
        swarm matrix of shape (n_particles, n_dimensions)
    """
    try:
        if (init_pos is not None) and (bounds is None):
            pos = init_pos
        elif (init_pos is not None) and (bounds is not None):
            if not (
                np.all(bounds[0] <= init_pos) and np.all(init_pos <= bounds[1])
            ):
                raise ValueError("User-defined init_pos is out of bounds.")
            pos = init_pos
        elif (init_pos is None) and (bounds is None):
            pos = center * np.random.uniform(
                low=0.0, high=1.0, size=(n_particles, dimensions)
            )
        else:
            lb, ub = bounds
            min_bounds = np.repeat(
                np.array(lb)[np.newaxis, :], n_particles, axis=0
            )
            max_bounds = np.repeat(
                np.array(ub)[np.newaxis, :], n_particles, axis=0
            )
            pos = center * np.random.uniform(
                low=min_bounds, high=max_bounds, size=(n_particles, dimensions)
            )
    except ValueError:
        raise
    else:
        return pos


def generate_discrete_swarm(
    n_particles, dimensions, binary=False, init_pos=None
):
    """Generate a discrete swarm

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm.
    binary : bool (default is :code:`False`)
        generate a binary matrix
    init_pos : :code:`numpy.ndarray` (default is :code:`None`)
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.
    """
    try:
        if (init_pos is not None) and binary:
            if not len(np.unique(init_pos)) == 2:
                raise ValueError("User-defined init_pos is not binary!")
            pos = init_pos
        elif (init_pos is not None) and not binary:
            pos = init_pos
        elif (init_pos is None) and binary:
            pos = np.random.randint(2, size=(n_particles, dimensions))
        else:
            pos = np.random.random_sample(
                size=(n_particles, dimensions)
            ).argsort(axis=1)
    except ValueError:
        raise
    else:
        return pos


def generate_velocity(n_particles, dimensions, clamp=None):
    """Initialize a velocity vector

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm.
    clamp : tuple of floats (default is :code:`None`)
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.

    Returns
    -------
    numpy.ndarray
        velocity matrix of shape (n_particles, dimensions)
    """
    try:
        min_velocity, max_velocity = (0, 1) if clamp is None else clamp
        velocity = (max_velocity - min_velocity) * np.random.random_sample(
            size=(n_particles, dimensions)
        ) + min_velocity
    except (ValueError, TypeError):
        raise
    else:
        return velocity


def create_swarm(
    n_particles,
    dimensions,
    discrete=False,
    binary=False,
    options={},
    bounds=None,
    center=1.0,
    init_pos=None,
    clamp=None,
):
    """Abstract the generate_swarm() and generate_velocity() methods

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm
    options : dict (default is empty dict :code:`{}`)
        Swarm options, for example, c1, c2, etc.
    discrete : bool (default is :code:`False`)
        Creates a discrete swarm
    binary : bool (default is :code:`False`)
        generate a binary matrix
    bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    center : :code:`numpy.ndarray` (default is :code:`1`)
        a list of initial positions for generating the swarm
    init_pos : :code:`numpy.ndarray` (default is :code:`None`)
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.
    clamp : tuple of floats (default is :code:`None`)
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.

    Returns
    -------
    pyswarms.backend.swarms.Swarm
        a Swarm class
    """
    if discrete:
        position = generate_discrete_swarm(
            n_particles, dimensions, binary=binary
        )
    else:
        position = generate_swarm(
            n_particles,
            dimensions,
            bounds=bounds,
            center=center,
            init_pos=init_pos,
        )

    velocity = generate_velocity(n_particles, dimensions, clamp=clamp)
    return Swarm(position, velocity, options=options)
