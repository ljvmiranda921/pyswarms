# -*- coding: utf-8 -*-

"""
A Pyramid Network Topology

This class implements a pyramid topology. In this topology, the particles are connected by N-dimensional simplices.
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
    def __init__(self, static=False):
        """Initialize the class

        Parameters
        ----------
        static : bool (Default is :code:`False`)
            a boolean that decides whether the topology
            is static or dynamic
        """
        super(Pyramid, self).__init__(static)

    def compute_gbest(self, swarm):
        """Update the global best using a pyramid neighborhood approach

        This topology uses the :code:`Delaunay` class from :code:`scipy`. To prevent precision errors in the Delaunay
        class, custom :code:`qhull_options` were added. Namely, :code:`QJ0.001 Qbb Qc Qx`. The meaning of those options
        is explained in [qhull]. This method is used to triangulate N-dimensional space into simplices. The vertices of
        the simplicies consist of swarm particles. This method is adapted from the work of Lane et al.[SIS2008]

        [SIS2008] J. Lane, A. Engelbrecht and J. Gain, "Particle swarm optimization with spatially
        meaningful neighbours," 2008 IEEE Swarm Intelligence Symposium, St. Louis, MO, 2008,
        pp. 1-8. doi: 10.1109/SIS.2008.4668281
        [qhull] http://www.qhull.org/html/qh-optq.htm

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
            # If there are less than (swarm.dimensions + 1) particles they are all connected
            if swarm.n_particles < swarm.dimensions + 1:
                self.neighbor_idx = np.tile(np.arange(swarm.n_particles), (swarm.n_particles, 1))
                best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
                best_cost = np.min(swarm.pbest_cost)
            else:
                # Check if the topology is static or dynamic and assign neighbors
                if (self.static and self.neighbor_idx is None) or not self.static:
                    pyramid = Delaunay(swarm.position, qhull_options="QJ0.001 Qbb Qc Qx")
                    indices, index_pointer = pyramid.vertex_neighbor_vertices
                    # Insert all the neighbors for each particle in the idx array
                    self.neighbor_idx = np.array(
                        [index_pointer[indices[i]:indices[i + 1]] for i in range(swarm.n_particles)]
                    )

                idx_min = np.array(
                    [swarm.pbest_cost[self.neighbor_idx[i]].argmin() for i in range(len(self.neighbor_idx))]
                )
                best_neighbor = np.array(
                    [self.neighbor_idx[i][idx_min[i]] for i in range(len(self.neighbor_idx))]
                ).astype(int)

                # Obtain best cost and position
                best_cost = np.min(swarm.pbest_cost[best_neighbor])
                best_pos = swarm.pbest_pos[
                    best_neighbor[np.argmin(swarm.pbest_cost[best_neighbor])]
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
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.swarms.backend import Swarm
            from pyswarms.backend.topology import Pyramid

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Pyramid(static=False)

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
        return ops.compute_position(swarm, bounds)
