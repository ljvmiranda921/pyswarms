# -*- coding: utf-8 -*-

"""
A Random Network Topology

This class implements a random topology. All particles are connected in a random fashion.
"""

# Import standard library
import itertools
from typing import Any, Dict, List, Optional

# Import modules
import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import connected_components, dijkstra  # type: ignore

# Import from pyswarms
from pyswarms.backend import operators as ops
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.utils.types import Bounds, Clamp, Position


class Random(Topology):
    neighbor_idx: Optional[List[npt.NDArray[np.integer[Any]]]] = None

    def __init__(self, k: int, static: bool = False):
        """Initializes the class

        Parameters
        ----------
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles-1`
        static : bool
            a boolean that decides whether the topology
            is static or dynamic. Defaulg is `False`
        """
        super(Random, self).__init__(static)
        self.k = k

    def compute_gbest(self, swarm: Swarm, **kwargs: Dict[str, Any]):
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

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        # Check if the topology is static or dynamic and assign neighbors
        if self.neighbor_idx is None or not self.static:
            adj_matrix = self._compute_neighbors(swarm)
            self.neighbor_idx = [adj_matrix[i].nonzero()[0] for i in range(swarm.n_particles)]

        idx_min = np.array([swarm.pbest_cost[self.neighbor_idx[i]].argmin() for i in range(len(self.neighbor_idx))])
        best_neighbor = np.array([self.neighbor_idx[i][idx_min[i]] for i in range(len(self.neighbor_idx))]).astype(int)

        # Obtain best cost and position
        best_cost = np.min(swarm.pbest_cost[best_neighbor])
        best_pos: Position = swarm.pbest_pos[best_neighbor]

        return (best_pos, float(best_cost))

    def compute_velocity(
        self,
        swarm: Swarm,
        clamp: Optional[Clamp] = None,
        vh: Optional[VelocityHandler] = None,
        bounds: Optional[Bounds] = None,
    ):
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Random

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Random(static=False)
            my_vh = VelocityHandler(strategy="zero")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping. Default is `None`
        vh : pyswarms.backend.handlers.VelocityHandler, optional
            a VelocityHandler instance
        bounds : tuple of numpy.ndarray or list, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        vh = vh or VelocityHandler.factory("unmodified")
        return ops.compute_velocity(swarm, clamp, vh, bounds=bounds)

    def compute_position(self, swarm: Swarm, bounds: Optional[Bounds] = None, bh: Optional[BoundaryHandler] = None):
        """Update the position matrix

        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of numpy.ndarray or list, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler, optional
            a BoundaryHandler instance

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        bh = bh or BoundaryHandler(strategy="periodic")
        return ops.compute_position(swarm, bounds, bh)

    def _compute_neighbors(self, swarm: Swarm):
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
                    self.k,
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
        while connected_components(adj_matrix, directed=False, return_labels=False) != 1:
            for i, j in itertools.product(range(swarm.n_particles), repeat=2):
                if dist_matrix[i][j] == np.inf:
                    adj_matrix[i][j] = 1

        return adj_matrix
