# -*- coding: utf-8 -*-

"""
A Star Network Topology

This class implements a star topology. In this topology,
all particles are connected to one another. This social
behavior is often found in GlobalBest PSO
optimizers.
"""

# Import standard library
import logging

# Import modules
import numpy as np

from .. import operators as ops
from ..handlers import BoundaryHandler, VelocityHandler
from ...utils.reporter import Reporter
from .base import Topology


class Star(Topology):
    def __init__(self, static=None, **kwargs):
        # static = None is just an artifact to make the API consistent
        # Setting it will not change swarm behavior
        super(Star, self).__init__(static=True)
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def compute_gbest(self, swarm, **kwargs):
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
        try:
            if self.neighbor_idx is None:
                self.neighbor_idx = np.tile(
                    np.arange(swarm.n_particles), (swarm.n_particles, 1)
                )
            if np.min(swarm.pbest_cost) < swarm.best_cost:
                # Get the particle position with the lowest pbest_cost
                # and assign it to be the best_pos
                best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
                best_cost = np.min(swarm.pbest_cost)
            else:
                # Just get the previous best_pos and best_cost
                best_pos, best_cost = swarm.best_pos, swarm.best_cost
        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return (best_pos, best_cost)
