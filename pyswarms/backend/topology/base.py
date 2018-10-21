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

# Import standard library
import abc
import logging

from ...utils.reporter import Reporter


class Topology(abc.ABC):
    def __init__(self, static, **kwargs):
        """Initializes the class"""

        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))

        # Initialize attributes
        self.static = static
        self.neighbor_idx = None

        if self.static:
            self.rep.log(
                "Running on `dynamic` topology,"
                "set `static=True` for fixed neighbors.",
                lvl=logging.DEBUG,
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
