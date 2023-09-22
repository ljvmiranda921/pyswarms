"""
The :code:`pyswarms.backend` module abstracts various operations
for swarm optimization: generating boundaries, updating positions, etc.
You can use the methods implemented here to build your own PSO implementations.
"""

from .generators import create_swarm, generate_discrete_swarm, generate_swarm, generate_velocity
from .handlers import BoundaryHandler, OptionsHandler, VelocityHandler
from .operators import compute_objective_function, compute_pbest, compute_position, compute_velocity
from .swarms import Swarm
