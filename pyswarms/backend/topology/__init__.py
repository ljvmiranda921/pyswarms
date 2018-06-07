"""
The :code:`pyswarms.backend.topology` contains various topologies that dictate
particle behavior. These topologies implement three methods:
    - compute_best_particle(): gets the position and cost of the best particle   in the swarm
    - update_velocity(): updates the velocity-matrix depending on the topology.
    - update_position(): updates the position-matrix depending on the topology.
"""

from .star import Star
from .ring import Ring


__all__ = [
    "Star",
    "Ring"
]