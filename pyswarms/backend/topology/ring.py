# -*- coding: utf-8 -*-

"""
A Ring Network Topology

This class implements a ring topology. In this topology,
the particles are connected with their k nearest neighbors.
This social behavior is often found in LocalBest PSO
optimizers.
"""

from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree  # type: ignore

from pyswarms.backend import operators as ops
from pyswarms.backend.handlers import BoundaryHandler
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.utils.types import Bounds, Position


class Ring(Topology):
    neighbor_idx: Optional[npt.NDArray[np.integer[Any]]] = None

    def __init__(self, p: Literal[1, 2], k: int, static: bool = False):
        """Initializes the class

        Parameters
        ----------
        p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles`
        static : bool (Default is :code:`False`)
            a boolean that decides whether the topology
            is static or dynamic
        """
        super(Ring, self).__init__(static)
        self.p = p
        self.k = k

    def compute_gbest(self, swarm: Swarm):
        """Update the global best using a ring-like neighborhood approach

        This uses the cKDTree method from :code:`scipy` to obtain the nearest
        neighbors.

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
        # Check if the topology is static or not and assign neighbors
        if self.neighbor_idx is None or not self.static:
            # Obtain the nearest-neighbors for each particle
            tree = cKDTree(swarm.position)
            _, self.neighbor_idx = tree.query(swarm.position, p=self.p, k=self.k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if self.k == 1:
            # The minimum index is itself, no mapping needed.
            self.neighbor_idx = self.neighbor_idx[:, np.newaxis]
            best_neighbor = np.arange(swarm.n_particles)
        else:
            idx_min = swarm.pbest_cost[self.neighbor_idx].argmin(axis=1)
            best_neighbor = self.neighbor_idx[np.arange(len(self.neighbor_idx)), idx_min]

        # Obtain best cost and position
        best_cost = np.min(swarm.pbest_cost[best_neighbor])
        best_pos: Position = swarm.pbest_pos[best_neighbor]

        return (best_pos, float(best_cost))

    def compute_position(self, swarm: Swarm, bounds: Optional[Bounds] = None, bh: Optional[BoundaryHandler] = None):
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
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        bh = bh or BoundaryHandler("periodic")
        return ops.compute_position(swarm, bounds, bh)
