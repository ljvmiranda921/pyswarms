# -*- coding: utf-8 -*-

"""
Base class for Topologies
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