"""
The :mod:`pyswarms.base` module implements base
swarm classes to implement variants of particle swarm optimization.
"""

from .bs import SwarmBase
from .dbs import DiscreteSwarmBase

__all__ = [
    "SwarmBase",
    "DiscreteSwarmBase"
    ]