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
from typing import Any, Dict, Optional, Tuple
from loguru import logger

# Import modules
import numpy as np
import numpy.typing as npt
from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler

# Import from pyswarms
from pyswarms.backend.swarms import Swarm
from pyswarms.utils.types import Bounds, Clamp, Position, Velocity


class Topology(abc.ABC):
    neighbor_idx: Optional[npt.NDArray[np.integer[Any]]] = None

    def __init__(self, static: bool, **kwargs: Dict[str, Any]):
        """Initializes the class"""

        # Initialize attributes
        self.static = static

        if not self.static:
            logger.debug("Running on `dynamic` topology," "set `static=True` for fixed neighbors.")

    @abc.abstractmethod
    def compute_gbest(self, swarm: Swarm, **kwargs: Dict[str, Any]) -> Tuple[Position, float]:
        """Compute the best particle of the swarm and return the cost and
        position"""
        raise NotImplementedError("Topology::compute_gbest()")

    @abc.abstractmethod
    def compute_position(self, swarm: Swarm, bounds: Optional[Bounds] = None, bh: BoundaryHandler = BoundaryHandler(strategy="periodic")) -> Position:
        """Update the swarm's position-matrix"""
        raise NotImplementedError("Topology::compute_position()")

    @abc.abstractmethod
    def compute_velocity(
        self,
        swarm: Swarm,
        clamp: Optional[Clamp] = None,
        vh: Optional[VelocityHandler] = None,
        bounds: Optional[Bounds] = None,
    ) -> Velocity:
        """Update the swarm's velocity-matrix"""
        raise NotImplementedError("Topology::compute_velocity()")
