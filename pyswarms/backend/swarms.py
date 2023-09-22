# -*- coding: utf-8 -*-

"""
Swarm Class Backend

This module implements a Swarm class that holds various attributes in
the swarm such as position, velocity, options, etc. You can use this
as input to most backend cases.
"""

# Import standard library
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Import modules
import numpy as np
import numpy.typing as npt

# Import from pyswarms
from pyswarms.utils.types import Position, Velocity


@dataclass
class Swarm:
    """A Swarm Class

    This class offers a generic swarm that can be used in most use-cases
    such as single-objective optimization, etc. It contains various attributes
    that are commonly-used in most swarm implementations.

    To initialize this class, **simply supply values for the position and
    velocity matrix**. The other attributes are automatically filled. If you want to
    initialize random values, take a look at:

    * :func:`pyswarms.backend.generators.generate_swarm`: for generating positions randomly.
    * :func:`pyswarms.backend.generators.generate_velocity`: for generating velocities randomly.

    If your swarm requires additional parameters (say c1, c2, and w in gbest
    PSO), simply pass them to the :code:`options` dictionary.

    As an example, say we want to create a swarm by generating particles
    randomly. We can use the helper methods above to do our job:

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        # Let's generate a 10-particle swarm with 10 dimensions
        init_positions = P.generate_swarm(n_particles=10, dimensions=10)
        init_velocities = P.generate_velocity(n_particles=10, dimensions=10)
        # Say, particle behavior is governed by parameters `foo` and `bar`
        my_options = {'foo': 0.4, 'bar': 0.6}
        # Initialize the swarm
        my_swarm = Swarm(position=init_positions, velocity=init_velocities, options=my_options)

    From there, you can now use all the methods in :mod:`pyswarms.backend`.
    Of course, the process above has been abstracted by the method
    :func:`pyswarms.backend.generators.create_swarm` so you don't have to
    write the whole thing down.


    Attributes
    ----------
    position : numpy.ndarray
        position-matrix at a given timestep of shape :code:`(n_particles, dimensions)`
    velocity : numpy.ndarray
        velocity-matrix at a given timestep of shape :code:`(n_particles, dimensions)`
    n_particles : int
        number of particles in a swarm.
    dimensions : int
        number of dimensions in a swarm.
    options : dict
        various options that govern a swarm's behavior.
    pbest_pos : numpy.ndarray
        personal best positions of each particle of shape :code:`(n_particles, dimensions)`
        Default is `None`
    best_pos : numpy.ndarray
        best position found by the swarm of shape :code:`(dimensions, )` for
        the :obj:`pyswarms.backend.topology.Star` topology and
        :code:`(dimensions, particles)` for the other topologies
    pbest_cost : numpy.ndarray
        personal best costs of each particle of shape :code:`(n_particles, )`
    best_cost : float
        best cost found by the swarm, default is :obj:`numpy.inf`
    current_cost : numpy.ndarray
        the current cost found by the swarm of shape :code:`(n_particles, dimensions)`
    """

    # Required attributes
    position: Position
    velocity: Velocity

    # With defaults
    n_particles: int = field(init=False)
    dimensions: int = field(init=False)
    pbest_pos: Optional[Position] = field(default=None)
    options: Dict[str, Any] = field(default_factory=dict)
    best_pos: Position = field(default_factory=lambda: np.array([], dtype=float))
    pbest_cost: Position = field(default_factory=lambda: np.array([], dtype=float))
    best_cost: int | float = np.inf
    current_cost: npt.NDArray[np.floating[Any]] = field(default_factory=lambda: np.array([], dtype=float))

    def __post_init__(self):
        self.n_particles = self.position.shape[0]
        self.dimensions = self.position.shape[1]

        if self.pbest_pos is None:
            self.pbest_pos = self.position
