# -*- coding: utf-8 -*-

"""
A Random Network Topology

This class implements a random topology. All particles are connected in a random fashion.
"""

# Import standard library
import itertools
import logging

# Import modules
import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra

from .. import operators as ops
from ..handlers import BoundaryHandler, VelocityHandler
from ...utils.reporter import Reporter
from .base import Topology


class Random(Topology):
    def __init__(self, static=False):
        """Initializes the class

        Parameters
        ----------
        static : bool
            a boolean that decides whether the topology
            is static or dynamic. Defaulg is `False`
        """
        super(Random, self).__init__(static)
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def compute_gbest(self, swarm, k, **kwargs):
        """Update the global best using a random neighborhood approach

        This uses random class from :code:`numpy` to give every particle k
        randomly distributed, non-equal neighbors. The resulting topology
        is a connected graph. The algorithm to obtain the neighbors was adapted
        from [TSWJ2013].

        [TSWJ2013] Qingjian Ni and Jianming Deng, “A New Logistic Dynamic
        Particle Swarm Optimization Algorithm Based on Random Topology,”
        The Scientific World Journal, vol. 2013, Article ID 409167, 8 pages, 2013.
        https://doi.org/10.1155/2013/409167.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles-1`

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        try:
            # Check if the topology is static or dynamic and assign neighbors
            if (self.static and self.neighbor_idx is None) or not self.static:
                adj_matrix = self.__compute_neighbors(swarm, k)
                self.neighbor_idx = np.array(
                    [
                        adj_matrix[i].nonzero()[0]
                        for i in range(swarm.n_particles)
                    ]
                )
            idx_min = np.array(
                [
                    swarm.pbest_cost[self.neighbor_idx[i]].argmin()
                    for i in range(len(self.neighbor_idx))
                ]
            )
            best_neighbor = np.array(
                [
                    self.neighbor_idx[i][idx_min[i]]
                    for i in range(len(self.neighbor_idx))
                ]
            ).astype(int)

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

    def __compute_neighbors(self, swarm, k):
        """Helper method to compute the adjacency matrix of the topology

        This method computes the adjacency matrix of the topology using
        the randomized algorithm proposed in [TSWJ2013]. The resulting
        topology is a connected graph. This is achieved by creating three
        matrices:

            * adj_matrix :  The adjacency matrix of the generated graph.
                            It's initialized as an identity matrix to
                            make sure that every particle has itself as
                            a neighbour. This matrix is the return
                            value of the method.
            * neighbor_matrix : The matrix of randomly generated neighbors.
                                This matrix is a matrix of shape
                                :code:`(swarm.n_particles, k)`:
                                with randomly generated elements. It is used
                                to create connections in the :code:`adj_matrix`.
            * dist_matrix : The distance matrix computed with Dijkstra's
                            algorithm. It is used to determine where the
                            graph needs edges to change it to a connected
                            graph.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles-1`

        Returns
        -------
        numpy.ndarray
            Adjacency matrix of the topology
        """

        adj_matrix = np.identity(swarm.n_particles, dtype=int)

        neighbor_matrix = np.array(
            [
                np.random.choice(
                    # Exclude i from the array
                    np.setdiff1d(np.arange(swarm.n_particles), np.array([i])),
                    k,
                    replace=False,
                )
                for i in range(swarm.n_particles)
            ]
        )

        # Set random elements to one using the neighbor matrix
        adj_matrix[
            np.arange(swarm.n_particles).reshape(swarm.n_particles, 1),
            neighbor_matrix,
        ] = 1
        adj_matrix[
            neighbor_matrix,
            np.arange(swarm.n_particles).reshape(swarm.n_particles, 1),
        ] = 1

        dist_matrix = dijkstra(
            adj_matrix,
            directed=False,
            return_predecessors=False,
            unweighted=True,
        )

        # Generate connected graph.
        while (
            connected_components(
                adj_matrix, directed=False, return_labels=False
            )
            != 1
        ):
            for i, j in itertools.product(range(swarm.n_particles), repeat=2):
                if dist_matrix[i][j] == np.inf:
                    adj_matrix[i][j] = 1

        return adj_matrix
