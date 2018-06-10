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

class Topology(object):

    def __init__(self, **kwargs):
        """Initializes the class"""
        pass

    def compute_gbest(self, swarm):
        """Computes the best particle of the swarm and returns the cost and
        position"""
        raise NotImplementedError("Topology::compute_gbest()")

    def compute_position(self, swarm):
        """Updates the swarm's position-matrix"""
        raise NotImplementedError("Topology::compute_position()")

    def compute_velocity(self, swarm):
        """Updates the swarm's velocity-matrix"""
        raise NotImplementedError("Topology::compute_velocity()")