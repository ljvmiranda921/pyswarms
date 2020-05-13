# -*- coding: utf-8 -*-

"""
A Ring Network Topology

This class implements a ring topology. In this topology,
the particles are connected with their k nearest neighbors.
This social behavior is often found in LocalBest PSO
optimizers.
"""

# Import standard library
import logging

# Import modules
import numpy as np
from scipy.spatial import cKDTree

from .. import operators as ops
from ..handlers import BoundaryHandler, VelocityHandler
from ...utils.reporter import Reporter
from .base import Topology


class Ring(Topology):
    def __init__(self, static=False):
        """Initializes the class

        Parameters
        ----------
        static : bool (Default is :code:`False`)
            a boolean that decides whether the topology
            is static or dynamic
        """
        super(Ring, self).__init__(static)
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def compute_gbest(self, swarm, p, k, **kwargs):
        """Update the global best using a ring-like neighborhood approach

        This uses the cKDTree method from :code:`scipy` to obtain the nearest
        neighbors.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles`

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            # Check if the topology is static or not and assign neighbors
            if (self.static and self.neighbor_idx is None) or not self.static:
                # Obtain the nearest-neighbors for each particle
                tree = cKDTree(swarm.position)
                _, self.neighbor_idx = tree.query(swarm.position, p=p, k=k)

            # Map the computed costs to the neighbour indices and take the
            # argmin. If k-neighbors is equal to 1, then the swarm acts
            # independently of each other.
            if k == 1:
                # The minimum index is itself, no mapping needed.
                self.neighbor_idx = self.neighbor_idx[:, np.newaxis]
                best_neighbor = np.arange(swarm.n_particles)
            else:
                idx_min = swarm.pbest_cost[self.neighbor_idx].argmin(axis=1)
                best_neighbor = self.neighbor_idx[
                    np.arange(len(self.neighbor_idx)), idx_min
                ]
            # Obtain best cost and position
            best_cost = np.min(swarm.pbest_cost[best_neighbor])
            best_pos = swarm.pbest_pos[best_neighbor]
        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return (best_pos, best_cost)
