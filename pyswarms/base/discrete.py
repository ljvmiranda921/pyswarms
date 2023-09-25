# -*- coding: utf-8 -*-

r"""
Base class for single-objective discrete Particle Swarm Optimization
implementations.

All methods here are abstract and raises a :code:`NotImplementedError`
when not used. When defining your own swarm implementation,
create another class,

    >>> class MySwarm(DiscreteSwarmOptimizer):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

and define all the necessary methods needed.

As a guide, check the discrete PSO implementations in this package.

.. note:: Regarding :code:`options`, it is highly recommended to
    include parameters used in position and velocity updates as
    keyword arguments. For parameters that affect the topology of
    the swarm, it may be much better to have them as positional
    arguments.

See Also
--------
:mod:`pyswarms.discrete.binary`: binary PSO implementation

"""

from typing import List, Optional

import numpy as np

from pyswarms.backend.generators import generate_discrete_swarm, generate_velocity
from pyswarms.backend.swarms import Swarm
from pyswarms.base.base import BaseSwarmOptimizer, Options
from pyswarms.utils.types import Clamp, Position, Velocity


class DiscreteSwarmOptimizer(BaseSwarmOptimizer):
    # Initialize history lists
    cost_history: List[float] = []
    mean_pbest_history: List[float] = []
    mean_neighbor_history: List[float] = []
    pos_history: List[Position] = []
    velocity_history: List[Velocity] = []

    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        options: Options,
        binary: bool = True,
        velocity_clamp: Optional[Clamp] = None,
        init_pos: Optional[Position] = None,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
    ):
        """Initialize the swarm.

        Creates a :code:`numpy.ndarray` of positions depending on the
        number of particles needed and the number of dimensions.
        The initial positions of the particles depends on the argument
        :code:`binary`, which governs if a binary matrix will be produced.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        binary : boolean
            a trigger to generate a binary matrix for the swarm's
            initial positions. When passed with a :code:`False` value,
            random integers from 0 to :code:`dimensions` are generated.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`.
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        """
        # Initialize primary swarm attributes
        self.binary = binary
        super().__init__(n_particles, dimensions, options, velocity_clamp, init_pos, ftol, ftol_iter)

    def _init_swarm(self):
        position = generate_discrete_swarm(
            self.n_particles, self.dimensions, binary=self.binary, init_pos=self.init_pos
        )
        velocity = generate_velocity(self.n_particles, self.dimensions, clamp=self.velocity_clamp)
        self.swarm = Swarm(position, velocity, options=dict(self.options))
