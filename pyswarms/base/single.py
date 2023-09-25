# -*- coding: utf-8 -*-

r"""
Base class for single-objective Particle Swarm Optimization
implementations.

All methods here are abstract and raise a :code:`NotImplementedError`
when not used. When defining your own swarm implementation,
create another class,

    >>> class MySwarm(SwarmBase):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

and define all the necessary methods needed.

As a guide, check the global best and local best implementations in this
package.

.. note:: Regarding :code:`options`, it is highly recommended to
    include parameters used in position and velocity updates as
    keyword arguments. For parameters that affect the topology of
    the swarm, it may be much better to have them as positional
    arguments.

See Also
--------
:mod:`pyswarms.single.global_best`: global-best PSO implementation
:mod:`pyswarms.single.local_best`: local-best PSO implementation
:mod:`pyswarms.single.general_optimizer`: a more general PSO implementation with a custom topology
"""

from typing import Optional

import numpy as np

from pyswarms.backend.generators import generate_swarm, generate_velocity
from pyswarms.backend.swarms import Swarm
from pyswarms.base.base import BaseSwarmOptimizer, Options
from pyswarms.utils.types import Bounds, Clamp, Position


class SwarmOptimizer(BaseSwarmOptimizer):
    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        options: Options,
        bounds: Optional[Bounds] = None,
        velocity_clamp: Optional[Clamp] = None,
        center: float = 1.0,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
        init_pos: Optional[Position] = None,
    ):
        """Initialize the swarm

        Creates a Swarm class depending on the values initialized

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        center : list, optional
            an array of size :code:`dimensions`
        ftol : float, optional
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`.
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        """
        self.bounds = bounds
        self.center = center
        super().__init__(n_particles, dimensions, options, velocity_clamp, init_pos, ftol, ftol_iter)

    def _init_swarm(self):
        position = generate_swarm(
            self.n_particles,
            self.dimensions,
            bounds=self.bounds,
            center=self.center,
            init_pos=self.init_pos,
        )
        velocity = generate_velocity(self.n_particles, self.dimensions, clamp=self.velocity_clamp)
        self.swarm = Swarm(position, velocity, options=dict(self.options))
