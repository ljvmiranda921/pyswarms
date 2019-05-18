# -*- coding: utf-8 -*-

"""
Swarm Generation Backend

This module abstracts how a swarm is generated. You can see its
implementation in our base classes. In addition, you can use all the methods
here to dictate how a swarm is initialized for your custom PSO.

"""

# Import standard library
import logging

# Import modules
import numpy as np

from ..utils.reporter import Reporter
from .swarms import Swarm

rep = Reporter(logger=logging.getLogger(__name__))


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
        msg = "Bounds and/or init_pos should be of size ({},)"
        rep.logger.exception(msg.format(dimensions))
        raise
    except TypeError:
        msg = "generate_swarm() takes an int for n_particles and dimensions and an array for bounds"
        rep.logger.exception(msg)
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
    TypeError
        When the argument passed to n_particles or dimensions is incorrect.
    """
    try:
        if (init_pos is not None) and binary:
            if not len(np.unique(init_pos)) <= 2:
                raise ValueError("User-defined init_pos is not binary!")
            # init_pos maybe ones
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
        rep.logger.exception("Please check the size and value of dimensions")
        raise
    except TypeError:
        msg = "generate_discrete_swarm() takes an int for n_particles and dimensions"
        rep.logger.exception(msg)
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
    clamp : tuple of floats, optional
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping. Default is :code:`None`

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
    except ValueError:
        msg = "Please check clamp shape: {} != {}"
        rep.logger.exception(msg.format(len(clamp), dimensions))
        raise
    except TypeError:
        msg = "generate_velocity() takes an int for n_particles and dimensions and an array for clamp"
        rep.logger.exception(msg)
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
    discrete : bool
        Creates a discrete swarm. Default is `False`
    options : dict, optional
        Swarm options, for example, c1, c2, etc.
    binary : bool
        generate a binary matrix, Default is `False`
    bounds : tuple of np.ndarray or list
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`. Default is `None`
    center : numpy.ndarray, optional
        a list of initial positions for generating the swarm. Default is `1`
    init_pos : numpy.ndarray, optional
        option to explicitly set the particles' initial positions. Set to
        :code:`None` if you wish to generate the particles randomly.
    clamp : tuple of floats, optional
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
            n_particles, dimensions, binary=binary, init_pos=init_pos
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
