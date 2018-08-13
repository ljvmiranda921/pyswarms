# -*- coding: utf-8 -*-

"""
Base class for Topologies

You can use this class to create your own topology. Note that every Topology
should implement a way to compute the (1) best particle, the (2) next
position, and the (3) next velocity given the Swarm's attributes at a given
timestep. Not implementing these methods will raise an error.

In addition, this class must interface with any class found in the
:mod:`pyswarms.backend.swarms.Swarm` module.
"""

# Import from stdlib
import abc
import logging

# Import from package
from ...utils.console_utils import cli_print


class Topology(abc.ABC):
    def __init__(self, static, **kwargs):
        """Initializes the class"""

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize attributes
        self.static = static
        self.neighbor_idx = None

        if self.static:
            cli_print(
                "Running on `dynamic` topology, neighbors are updated regularly."
                "Set `static=True` for fixed neighbors.",
                1,
                0,
                self.logger,
            )

    @abc.abstractmethod
    def compute_gbest(self, swarm):
        """Compute the best particle of the swarm and return the cost and
        position"""
        raise NotImplementedError("Topology::compute_gbest()")

    @abc.abstractmethod
    def compute_position(self, swarm):
        """Update the swarm's position-matrix"""
        raise NotImplementedError("Topology::compute_position()")

    @abc.abstractmethod
    def compute_velocity(self, swarm):
        """Update the swarm's velocity-matrix"""
        raise NotImplementedError("Topology::compute_velocity()")
