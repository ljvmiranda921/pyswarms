# -*- coding: utf-8 -*-

r"""
A Local-best Particle Swarm Optimization (lbest PSO) algorithm.

Similar to global-best PSO, it takes a set of candidate solutions,
and finds the best solution using a position-velocity update method.
However, it uses a ring topology, thus making the particles
attracted to its corresponding neighborhood.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

However, in local-best PSO, a particle doesn't compare itself to the
overall performance of the swarm. Instead, it looks at the performance
of its nearest-neighbours, and compares itself with them. In general,
this kind of topology takes much more time to converge, but has a more
powerful explorative feature.

In this implementation, a neighbor is selected via a k-D tree
imported from :code:`scipy`. Distance are computed with either
the L1 or L2 distance. The nearest-neighbours are then queried from
this k-D tree. They are computed for every iteration.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2,
                                       options=options)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)

This algorithm was adapted from one of the earlier works of
J. Kennedy and R.C. Eberhart in Particle Swarm Optimization
[IJCNN1995]_ [MHS1995]_

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

.. [MHS1995] J. Kennedy and R.C. Eberhart, "A New Optimizer using Particle
    Swarm Theory,"  in Proceedings of the Sixth International
    Symposium on Micromachine and Human Science, 1995, pp. 39–43.
"""

from typing import Literal, Optional

import numpy as np

from pyswarms.backend.topology.ring import Ring
from pyswarms.optimizers.general_optimizer import GeneralOptimizerPSO
from pyswarms.utils.types import (
    BoundaryStrategy,
    Bounds,
    Clamp,
    Position,
    SwarmOptions,
    VelocityStrategy,
)


class LocalBestPSO(GeneralOptimizerPSO):
    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        options: SwarmOptions,
        bounds: Optional[Bounds] = None,
        bh_strategy: BoundaryStrategy = "periodic",
        velocity_clamp: Optional[Clamp] = None,
        vh_strategy: VelocityStrategy = "unmodified",
        center: float = 1.00,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
        init_pos: Optional[Position] = None,
        p: Literal[1, 2] = 2,
        k: Optional[int] = None,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            A dictionary containing the parameters for the specific
            optimization technique. This can be a constant, OptionsStrategy
            or an OptionsHandler.
                * c1 : float|Tuple[OptionsStrategy, float]|OptionsHandler
                    cognitive parameter
                * c2 : float|Tuple[OptionsStrategy, float]|OptionsHandler
                    social parameter
                * w : float|Tuple[OptionsStrategy, float]|OptionsHandler
                    inertia parameter
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple (default is :code:`(0,1)`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list, optional
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        static: bool
            a boolean that decides whether the Ring topology
            used is static or dynamic. Default is `False`
        k : int, optional
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles`. Defaults is n_particles - 1
        p: int {1,2}, optional
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance. Default is 2
        """
        self.p = p
        self.k = k or n_particles - 1

        super().__init__(
            n_particles,
            dimensions,
            options,
            Ring(self.p, self.k),
            bounds,
            bh_strategy,
            velocity_clamp,
            vh_strategy,
            center,
            ftol,
            ftol_iter,
            init_pos,
        )

        self.name = __name__
