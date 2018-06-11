# -*- coding: utf-8 -*-

"""
A Star Network Topology

This class implements a star topology where all particles are connected to
one another. This social behavior is often found in GlobalBest PSO
optimizers.
"""

# Import from stdlib
import logging

# Import modules
import numpy as np

# Import from package
from .. import operators as ops
from .base import Topology

# Create a logger
logger = logging.getLogger(__name__)

class Star(Topology):

    def __init__(self):
        super(Star, self).__init__()

    def compute_gbest(self, swarm):
        """Obtains the global best cost and position based on a star topology

        This method takes the current pbest_pos and pbest_cost, then returns
        the minimum cost and position from the matrix. It should be used in
        tandem with an if statement

        .. code-block:: python

            import pyswarms.backend as P
            from pyswarms.backend.swarms import Swarm
            from pyswarm.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()

            # If the minima of the pbest_cost is less than the best_cost
            if np.min(pbest_cost) < best_cost:
                # Update best_cost and position
                swarm.best_pos, swarm.best_cost = my_topology.compute_best_particle(my_swarm)

        Parameters
        ----------
        swarm : pyswarms.backend.swarm.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
            best_cost = np.min(swarm.pbest_cost)
        except AttributeError:
            msg = 'Please pass a Swarm class. You passed {}'.format(type(swarm))
            logger.error(msg)
            raise
        else:
            return (best_pos, best_cost)

    def compute_velocity(self, swarm, clamp=None):
        """Computes the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.
        
        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.swarms.backend import Swarm
            from pyswarms.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp)

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
        return ops.compute_velocity(swarm, clamp)

    def compute_position(self, swarm, bounds=None):
        """Updates the position matrix

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
        return ops.compute_position(swarm, bounds)