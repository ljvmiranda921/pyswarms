# -*- coding: utf-8 -*-

"""
A Star Network Topology

This class implements a star topology. In this topology,
all particles are connected to one another. This social
behavior is often found in GlobalBest PSO
optimizers.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.utils.types import Position


class Star(Topology):
    neighbor_idx: Optional[npt.NDArray[np.integer[Any]]] = None

    def __init__(self, static: bool = True):
        # static = None is just an artifact to make the API consistent
        # Setting it will not change swarm behavior
        super().__init__(static=True)

    def compute_gbest(self, swarm: Swarm):
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
