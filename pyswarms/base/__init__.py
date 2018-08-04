"""
The :mod:`pyswarms.base` module implements base
swarm classes to implement variants of particle swarm optimization.
"""

from .base_single import SwarmOptimizer
from .base_discrete import DiscreteSwarmOptimizer

__all__ = ["SwarmOptimizer" "DiscreteSwarmOptimizer"]
