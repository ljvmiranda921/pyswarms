"""
The :mod:`pyswarms.base` module implements base
swarm classes to implement variants of particle swarm optimization.
"""

from .base_single import SwarmBase
from .base_discrete import DiscreteSwarmBase

__all__ = [
    "SwarmBase",
    "DiscreteSwarmBase"
    ]
