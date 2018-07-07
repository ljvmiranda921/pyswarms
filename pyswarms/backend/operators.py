# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

# Import from stdlib
import logging

# Import modules
import numpy as np

# Create a logger
logger = logging.getLogger(__name__)


def compute_pbest(swarm):
    """Update the personal best score of a swarm instance

    You can use this method to update your personal best positions.

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        # Inside the for-loop...
        for i in range(iters):
            # It updates the swarm internally
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.update_pbest(my_swarm)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.

    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best
    position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarm.Swarm
        a Swarm instance

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    try:
        # Infer dimensions from positions
        dimensions = swarm.dimensions
        # Create a 1-D and 2-D mask based from comparisons
        mask_cost = swarm.current_cost < swarm.pbest_cost
        mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(
            ~mask_cost, swarm.pbest_cost, swarm.current_cost
        )
    except AttributeError:
        msg = "Please pass a Swarm class. You passed {}".format(type(swarm))
        logger.error(msg)
        raise
    else:
        return (new_pbest_pos, new_pbest_cost)


def compute_velocity(swarm, clamp):
    """Update the velocity matrix

    This method updates the velocity matrix using the best and current
    positions of the swarm. The velocity matrix is computed using the
    cognitive and social terms of the swarm.

    A sample usage can be seen with the following:

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        for i in range(iters):
            # Inside the for-loop
            my_swarm.velocity = update_velocity(my_swarm, clamp)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    clamp : tuple of floats (default is :code:`None`)
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.

    Returns
    -------
    numpy.ndarray
        Updated velocity matrix
    """
    try:
        # Prepare parameters
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        # Compute for cognitive and social terms
        cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
        )
        social = (
            c2
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.best_pos - swarm.position)
        )
        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * swarm.velocity) + cognitive + social

        if clamp is None:
            updated_velocity = temp_velocity
        else:
            min_velocity, max_velocity = clamp
            mask = np.logical_and(
                temp_velocity >= min_velocity, temp_velocity <= max_velocity
            )
            updated_velocity = np.where(~mask, swarm.velocity, temp_velocity)
    except AttributeError:
        msg = "Please pass a Swarm class. You passed {}".format(type(swarm))
        logger.error(msg)
        raise
    except KeyError:
        msg = "Missing keyword in swarm.options"
        logger.error(msg)
        raise
    else:
        return updated_velocity


def compute_position(swarm, bounds):
    """Update the position matrix

    This method updates the position matrix given the current position and
    the velocity. If bounded, it waives updating the position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.

    Returns
    -------
    numpy.ndarray
        New position-matrix
    """
    try:
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            lb, ub = bounds
            min_bounds = np.repeat(
                np.array(lb)[np.newaxis, :], swarm.n_particles, axis=0
            )
            max_bounds = np.repeat(
                np.array(ub)[np.newaxis, :], swarm.n_particles, axis=0
            )
            mask = np.all(min_bounds <= temp_position, axis=1) * np.all(
                temp_position <= max_bounds, axis=1
            )
            mask = np.repeat(mask[:, np.newaxis], swarm.dimensions, axis=1)
            temp_position = np.where(~mask, swarm.position, temp_position)
        position = temp_position
    except AttributeError:
        msg = "Please pass a Swarm class. You passed {}".format(type(swarm))
        logger.error(msg)
        raise
    else:
        return position
