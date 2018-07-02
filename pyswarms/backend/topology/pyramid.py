# -*- coding: utf-8 -*-

"""
A Pyramid Network Topology

This class implements a pyramid topology where all particles are connected in a N-dimensional simplex fashion.
"""

# Import from stdlib
import logging

# Import modules
import numpy as np
from scipy.spatial import Delaunay

# Import from package
from .. import operators as ops
from .base import Topology

# Create a logger
logger = logging.getLogger(__name__)


class Pyramid(Topology):
    def __init__(self):
        super(Pyramid, self).__init__()

    def compute_gbest(self, swarm):
        """Updates the global best using a pyramid neighborhood approach

        This uses the Delaunay method from :code:`scipy` to triangulate N-dimensional space
        with simplices consisting of swarm particles

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            # If there are less than 5 particles they are all connected
            if swarm.n_particles < 5:
                best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
                best_cost = np.min(swarm.pbest_cost)
            else:
                pyramid = Delaunay(swarm.position)
                indices, index_pointer = pyramid.vertex_neighbor_vertices
                # Insert all the neighbors for each particle in the idx array
                idx = np.array([index_pointer[indices[i]:indices[i+1]] for i in range(swarm.n_particles)])
                idx_min = np.array([swarm.pbest_cost[idx[i]].argmin() for i in range(idx.size)])
                best_neighbor = np.array([idx[i][idx_min[i]] for i in range(len(idx))]).astype(int)

                # Obtain best cost and position
                best_cost = np.min(swarm.pbest_cost[best_neighbor])
                best_pos = swarm.pbest_pos[
                    np.argmin(swarm.pbest_cost[best_neighbor])
                ]
        except AttributeError:
            msg = "Please pass a Swarm class. You passed {}".format(
                type(swarm)
            )
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
            from pyswarms.backend.topology import Pyramid

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Pyramid()

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
