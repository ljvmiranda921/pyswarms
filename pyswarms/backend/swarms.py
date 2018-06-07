# -*- coding: utf-8 -*-

"""
Swarm Class Backend

This module implements a Swarm class that holds various attributes in
the swarm such as position, velocity, options, etc. You can use this
as input to most backend cases
"""

# Import modules
import numpy as np
from attr import (attrs, attrib)
from attr.validators import instance_of

@attrs
class Swarm(object):
    """A Swarm Class
    
    This class offers a generic swarm that can be used in most use-cases
    such as single-objective optimization, etc. It contains various attributes
    that are commonly-used in most swarm implementations.
    """
    # Required attributes
    position = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    velocity = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    # With defaults
    n_particles = attrib(type=int, validator=instance_of(int))
    dimensions = attrib(type=int, validator=instance_of(int))
    options = attrib(type=dict, default={}, validator=instance_of(dict))
    pbest_pos = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    best_pos  = attrib(type=np.ndarray, default=np.array([]), validator=instance_of(np.ndarray))
    pbest_cost = attrib(type=np.ndarray, default=np.array([]), validator=instance_of(np.ndarray))
    best_cost = attrib(type=float, default=np.inf, validator=instance_of((int, float)))
    current_cost = attrib(type=np.ndarray, default=np.array([]), validator=instance_of(np.ndarray))

    @n_particles.default
    def n_particles_default(self):
        return self.position.shape[0]

    @dimensions.default
    def dimensions_default(self):
        return self.position.shape[1]

    @pbest_pos.default
    def pbest_pos_default(self):
        return self.position