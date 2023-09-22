# -*- coding: utf-8 -*-

"""
Swarm Generation Backend

This module abstracts how a swarm is generated. You can see its
implementation in our base classes. In addition, you can use all the methods
here to dictate how a swarm is initialized for your custom PSO.

"""

# Import standard library
from typing import Optional
from loguru import logger

# Import modules
import numpy as np

# Import from pyswarms
from pyswarms.utils.types import Bounds, Clamp, Position, Velocity


def generate_swarm(
    n_particles: int,
    dimensions: int,
    bounds: Optional[Bounds] = None,
    center: float | Position = 1.00,
    init_pos: Optional[Position] = None,
) -> Position:
    """Generate a swarm

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`. Default is :code:`None`
    center : numpy.ndarray or float, optional
        controls the mean or center whenever the swarm is generated randomly.
        Default is :code:`1`
    init_pos : numpy.ndarray, optional
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.
        Default is :code:`None`.

    Returns
    -------
    numpy.ndarray
        swarm matrix of shape (n_particles, n_dimensions)

    Raises
    ------
    ValueError
        When the shapes and values of bounds, dimensions, and init_pos
        are inconsistent.
    TypeError
        When the argument passed to bounds is not an iterable.
    """
    if bounds is not None:
        if init_pos is not None:
            if not (np.all(bounds[0] <= init_pos) and np.all(init_pos <= bounds[1])):
                raise ValueError("User-defined init_pos is out of bounds.")
            pos = init_pos
        else:
            try:
                lb, ub = bounds
                min_bounds = np.repeat(np.array(lb)[np.newaxis, :], n_particles, axis=0)
                max_bounds = np.repeat(np.array(ub)[np.newaxis, :], n_particles, axis=0)
                pos = center * np.random.uniform(low=min_bounds, high=max_bounds, size=(n_particles, dimensions))
            except ValueError as e:
                msg = "Bounds and/or init_pos should be of size ({},)"
                logger.error(msg.format(dimensions))
                raise e
    else:
        if init_pos is not None:
            pos = init_pos
        else:
            pos = center * np.random.uniform(low=0.0, high=1.0, size=(n_particles, dimensions))

    return pos


def generate_discrete_swarm(
    n_particles: int, dimensions: int, binary: bool = False, init_pos: Optional[Position] = None
) -> Position:
    """Generate a discrete swarm

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm.
    binary : bool
        generate a binary matrix. Default is :code:`False`
    init_pos : numpy.ndarray, optional
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.
        Default is :code:`None`

    Returns
    -------
    numpy.ndarray
        swarm matrix of shape (n_particles, n_dimensions)

    Raises
    ------
    ValueError
        When init_pos during binary=True does not contain two unique values.
        When init_pos has the incorrect number of dimensions
    AssertionError
        When init_pos has an incorrect shape
    """
    if init_pos is not None:
        if binary and len(np.unique(init_pos)) > 2:
            raise ValueError("User-defined init_pos is not binary!")
        
        if init_pos.ndim == 1:
            assert init_pos.shape[0] == dimensions
            pos = np.repeat([init_pos], n_particles, axis=0)
        elif init_pos.ndim == 2:
            assert init_pos.shape[0] == n_particles
            assert init_pos.shape[1] == dimensions
            pos = init_pos
        else:
            raise ValueError("init_pos must be 1D or 2D")
        
    else:
        if binary:
            pos = np.random.randint(2, size=(n_particles, dimensions)) # type: ignore
        else:
            pos = np.random.random_sample(size=(n_particles, dimensions)).argsort(axis=1)

    return pos


def generate_velocity(n_particles: int, dimensions: int, clamp: Optional[Clamp] = None) -> Velocity:
    """Initialize a velocity vector

    Parameters
    ----------
    n_particles : int
        number of particles to be generated in the swarm.
    dimensions: int
        number of dimensions to be generated in the swarm.
    clamp : tuple of floats, optional
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping. Default is :code:`None`

    Returns
    -------
    numpy.ndarray
        velocity matrix of shape (n_particles, dimensions)
    """
    min_velocity, max_velocity = (0, 1) if clamp is None else np.array(clamp)

    velocity = (max_velocity - min_velocity) * np.random.random_sample(
        size=(n_particles, dimensions)
    ) + min_velocity

    return velocity
