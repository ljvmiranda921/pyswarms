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

import abc
from typing import Any, Tuple

from loguru import logger

from pyswarms.backend.swarms import Swarm
from pyswarms.utils.types import Position


class Topology(abc.ABC):
    def __init__(self, static: bool = False, **kwargs: Any):
        """Initializes the class"""

        # Initialize attributes
        self.static = static

        if not self.static:
            logger.debug("Running on `dynamic` topology," "set `static=True` for fixed neighbors.")

    @abc.abstractmethod
    def compute_gbest(self, swarm: Swarm) -> Tuple[Position, float]:
        """Compute the best particle of the swarm and return the cost and
        position"""
        ...
