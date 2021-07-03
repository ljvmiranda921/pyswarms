"""
The :code:`pyswarms.backend` module abstracts various operations
for swarm optimization: generating boundaries, updating positions, etc.
You can use the methods implemented here to build your own PSO implementations.
"""

from .generators import generate_swarm, generate_discrete_swarm, generate_velocity, create_swarm
from .handlers import HandlerMixin, BoundaryHandler, VelocityHandler, OptionsHandler
from .operators import compute_pbest, compute_velocity, compute_position, compute_objective_function
from .swarms import Swarm

__all__ = ["generators", "handlers", "operators", "swarms"]
