# -*- coding: utf-8 -*-

"""
A Ring Network Topology

This class implements a ring topology. In this topology,
the particles are connected with their k nearest neighbors.
This social behavior is often found in LocalBest PSO
optimizers.
"""

# Import standard library
from typing import Any, Dict, Literal, Optional

# Import modules
import numpy as np
from scipy.spatial import cKDTree  # type: ignore

# Import from pyswarms
from pyswarms.backend import operators as ops
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.utils.types import Bounds, Clamp, Position


class Ring(Topology):
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

    def compute_gbest(self, swarm: Swarm, **kwargs: Dict[str, Any]):
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
            from pyswarms.backend.topology import Ring

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Ring(static=False)
            my_vh = VelocityHandler(strategy="invert")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler, optional
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        vh = vh or VelocityHandler.factory("unmodified")
        return ops.compute_velocity(swarm, clamp, vh, bounds)

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
