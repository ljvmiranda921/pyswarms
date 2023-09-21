"""
The :mod:`pyswarms.base` module implements base
swarm classes to implement variants of particle swarm optimization.
"""

from .discrete import DiscreteSwarmOptimizer
from .single import SwarmOptimizer

__all__ = ["SwarmOptimizer", "DiscreteSwarmOptimizer"]
