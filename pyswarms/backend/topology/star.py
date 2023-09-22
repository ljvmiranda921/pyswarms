# -*- coding: utf-8 -*-

"""
A Star Network Topology

This class implements a star topology. In this topology,
all particles are connected to one another. This social
behavior is often found in GlobalBest PSO
optimizers.
"""

# Import standard library
from typing import Any, Dict, Optional

# Import modules
import numpy as np

# Import from pyswarms
from pyswarms.backend import operators as ops
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.utils.types import Bounds, Clamp, Position


class Star(Topology):
    def __init__(self, static: bool = True):
        # static = None is just an artifact to make the API consistent
        # Setting it will not change swarm behavior
        super(Star, self).__init__(static=True)

    def compute_gbest(self, swarm: Swarm, **kwargs: Dict[str, Any]):
        """Update the global best using a star topology

        This method takes the current pbest_pos and pbest_cost, then returns
        the minimum cost and position from the matrix.

        .. code-block:: python

            import pyswarms.backend as P
            from pyswarms.backend.swarms import Swarm
            from pyswarm.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()

            # Update best_cost and position
            swarm.best_pos, swarm.best_cost = my_topology.compute_gbest(my_swarm)

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
        if self.neighbor_idx is None:
            self.neighbor_idx = np.tile(np.arange(swarm.n_particles), (swarm.n_particles, 1))
        if np.min(swarm.pbest_cost) < swarm.best_cost:
            # Get the particle position with the lowest pbest_cost
            # and assign it to be the best_pos
            best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
            best_cost = np.min(swarm.pbest_cost)
        else:
            # Just get the previous best_pos and best_cost
            best_pos: Position = swarm.best_pos
            best_cost = swarm.best_cost

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
            from pyswarms.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()
            my_vh = VelocityHandler(strategy="adjust")

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
        vh : pyswarms.backend.handlers.VelocityHandler
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
        return ops.compute_velocity(swarm, clamp, vh, bounds=bounds)

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
