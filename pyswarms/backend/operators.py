# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

from functools import partial
from multiprocessing.pool import Pool
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from pyswarms.backend.swarms import Swarm


def compute_pbest(swarm: Swarm):
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
    # Infer dimensions from positions
    dimensions = swarm.dimensions
    # Create a 1-D and 2-D mask based from comparisons
    mask_cost = swarm.current_cost < swarm.pbest_cost
    mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
    # Apply masks
    new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
    new_pbest_cost = np.where(~mask_cost, swarm.pbest_cost, swarm.current_cost)

    return (new_pbest_pos, new_pbest_cost)


def compute_objective_function(
    swarm: Swarm, objective_func: Callable[..., npt.NDArray[Any]], pool: Optional[Pool] = None, **kwargs: Any
):
    """Evaluate particles using the objective function

    This method evaluates each particle in the swarm according to the objective
    function passed.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_func : function
        objective function to be evaluated
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective function

    Returns
    -------
    numpy.ndarray
        Cost-matrix for the given swarm
    """
    if pool is None:
        return objective_func(swarm.position, **kwargs)
    else:
        results = pool.map(
            partial(objective_func, **kwargs),
            np.array_split(swarm.position, pool._processes),  # type: ignore
        )
        return np.concatenate(results)
