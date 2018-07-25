"""
The :code:`pyswarms.backend.topology` contains various topologies. They dictate
the behavior of the particles and implement three methods:
    - compute_best_particle(): gets the position and cost of the best particle   in the swarm
    - update_velocity(): updates the velocity-matrix depending on the topology.
    - update_position(): updates the position-matrix depending on the topology.
"""

from .base import Topology
from .star import Star
from .ring import Ring
from .pyramid import Pyramid
from .random import Random
from .von_neumann import VonNeumann


__all__ = ["Topology", "Star", "Ring", "Pyramid", "Random", "VonNeumann"]
