"""
The :code:`pyswarms.backend` module abstracts various operations
for swarm optimization: generating boundaries, updating positions, etc.
You can use the methods implemented here to build your own PSO implementations.
"""

from .generators import *
from .operators import *
from .swarms import *