# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

<<<<<<< HEAD
=======
# Import from stdlib
import logging

>>>>>>> f0e6a49... [WIP-operator] Add update_gbest_neighborhood to operators
# Import modules
import numpy as np
from scipy.spatial import cKDTree


def update_pbest(pbest_pos, pbest_cost, pos, cost):
    """Updates the personal best, pbest_pos, given the costs
    
    You can use this method to update your personal best positions.

    ..code-block :: python

        import pyswarms.backend as P

        for i in range(iters):
            current_cost = compute_fitness(current_pos)
            current_pbest_cost = compute_fitness(current_pbest)
        
            current_pbest_pos, pbest_cost = P.update_pbest(pbest_pos=current_pbest,
                                                           pbest_cost=current_pbest_cost,
                                                           pos=current_pos,
                                                           cost=current_cost)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.
    
    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best position.
    If you wish, you can supply these values as keyword arguments:

    ..code-block :: python

        import pyswarms.backend as P

        current_pbest = P.update_pbest(**{'pbest_pos': current_pbest,
                                          'pbest_cost': current_pbest_cost,
                                          'pos': current_pos,
                                          'cost': current_cost})
        current_pbest_pos, pbest_cost = current_pbest

    Parameters
    ----------
    pbest_pos : numpy.ndarray
        Current personal best positions of the swarm. Must be of shape
        :code:`(n_particles, n_dimensions)`
    pbest_cost : numpy.ndarray
        Cost or fitness of the personal best positions. Must be of shape
        :code:`(n_particles,)`
    pos : numpy.ndarray
        Current positions of the swarm. Must be of shape
        :code:`(n_particles, n_dimensions)`
    cost : numpy.ndarray
        Cost or fitness of the current positions. Must be of shape
        :code:`(n_particles, )`

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    # Infer dimensions from positions
    dimensions = pos.shape[1]
    # Create a 1-D and 2-D mask based from comparisons
    mask_cost = (cost < pbest_cost)
    mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
    # Apply masks
    new_pbest_pos = np.where(~mask_pos, pbest_pos, pos)
    new_pbest_cost = np.where(~mask_cost, pbest_cost, cost)
    return (new_pbest_pos, new_pbest_cost)

def update_gbest(pbest_pos, pbest_cost):
    """Updates the global best given the cost and the position

    This method takes the current pbest_pos and pbest_cost, then returns
    the minimum cost and position from the matrix. It should be used in
    tandem with an if statement

    .. code-block:: python

        import pyswarms.backend as P

        # If the minima of the pbest_cost is less than the best_cost
        if np.min(pbest_cost) < best_cost:
            # Update best_cost
            best_pos, best_cost = P.update_gbest(pbest_pos, pbest_cost)

    Parameters
    ----------
    pbest_pos : numpy.ndarray
        Current personal best positions of the swarm. Must be of shape
        :code:`(n_particles, n_dimensions)`
    pbest_cost : numpy.ndarray
        Cost or fitness of the personal best positions. Must be of shape
        :code:`(n_particles,)`

    Returns
    -------
    numpy.ndarray
        Best position of shape :code:`(n_dimensions, )`
    float
        Best cost
    """
    return (pbest_pos[np.argmin(pbest_cost)], np.min(pbest_cost))

def update_gbest_neighborhood(swarm, p, k):
    """Updates the global best using a neighborhood approach

    This uses the cKDTree method from :code:`scipy` to obtain the nearest
    neighbours

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    k : int
        number of neighbors to be considered. Must be a
        positive integer less than :code:`n_particles`
    p: int {1,2}
        the Minkowski p-norm to use. 1 is the
        sum-of-absolute values (or L1 distance) while 2 is
        the Euclidean (or L2) distance.

    Returns
    -------
    numpy.ndarray
        Best position of shape :code:`(n_dimensions, )`
    float
        Best cost
    """
    try:
        # Obtain the nearest-neighbors for each particle
        tree = cKDTree(swarm.position)
        _, idx = tree.query(swarm.position, p=p, k=k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = swarm.pbest_cost[idx][:, np.newaxis].argmin(axis=1)
        else:
            idx_min = swarm.pbest_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]
        # Obtain best cost and position
        best_cost = np.min(swarm.pbest_cost[best_neighbor])
        best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost[best_neighbor])]
    except AttributeError:
        msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
        logger.error(msg)
        raise
    else:
        return (best_pos, best_cost)

def update_velocity(swarm, clamp):
    """Updates the velocity matrix

    This method updates the velocity matrix using the best and current
    positions of the swarm. A sample usage can be seen with the following:

    .. code-block :: python

        import pyswarms.backend as P

        new_velocity = update_velocity(velocity, pos, pbest_pos, best_pos,
                                       **{'c1':0.5, 'c2':0.4, 'w':0.3})

    Parameters
    ----------
    velocity : numpy.ndarray
        Velocity-matrix of shape :code:`(n_samples, n_dimensions)`
    clamp : None or iterable
        Clamping value
    pos : numpy.ndarray
        Current positions of shape :code:`(n_samples, n_dimensions)`
    pbest_pos : numpy.ndarray
        Personal best positions of the particles of shape :code:`(n_samples, n_dimensions)`
    best_pos : numpy.ndarray
        Best position of the swarm of shape :code:`(n_dimensions, )`
    c1 : float
        Cognitive parameter
    c2 : float
        Social parameter
    w : float
        Inertia parameter

    Returns
    -------
    numpy.ndarray
        Updated velocity matrix
    """
    swarm_size = pos.shape
    cognitive = (c1 * np.random.uniform(0,1, swarm_size) * (pbest_pos - pos))
    social = (c2 * np.random.uniform(0, 1, swarm_size) * (best_pos - pos))

    # Compute temp velocity (subject to clamping if possible)
    temp_velocity = (w * velocity) + cognitive + social

    if clamp is None:
        updated_velocity = temp_velocity
    else:
        min_velocity, max_velocity = clamp
        mask = np.logical_and(temp_velocity >= min_velocity,
                              temp_velocity <= max_velocity)
        updated_velocity = np.where(~mask, velocity, temp_velocity)
    
    return updated_velocity

def update_position(velocity, position, bounds):
    """Updates the position matrix
    
    Parameters
    ----------
    velocity : numpy.ndarray
        velocity-matrix of shape (n_samples, dimensions)
    position : numpy.ndarray
        position-matrix of shape (n_samples, dimensions)

    """
    temp_position = position.copy()
    temp_position += velocity

    if bounds is not None:
        lb, ub = bounds
        min_bounds = np.repeat(np.array(lb)[np.newaxis, :], n_particles, axis=0)
        max_bounds = np.repeat(np.array(ub)[np.newaxis, :], n_particles, axis=0)
        mask = (np.all(min_bounds <= temp_position, axis=1)
               * np.all(temp_position <= max_bounds, axis=1))
        mask = np.repeat(mask[:, np.newaxis], position.shape[1], axis=1)
        temp_position = np.where(~mask, position, temp_position)
    position = temp_position
    return position