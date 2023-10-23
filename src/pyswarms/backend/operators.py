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
