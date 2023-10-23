# -*- coding: utf-8 -*-

r"""
A Global-best Particle Swarm Optimization (gbest PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Uses a
star-topology where each particle is attracted to the best
performing particle.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

from typing import Optional

import numpy as np

from pyswarms.backend.topology import Star
from pyswarms.optimizers.general_optimizer import GeneralOptimizerPSO
from pyswarms.utils.types import (
    BoundaryStrategy,
    Bounds,
    Clamp,
    Position,
    SwarmOptions,
    VelocityStrategy,
)


class GlobalBestPSO(GeneralOptimizerPSO):
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
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
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
        """
        super().__init__(
            n_particles,
            dimensions,
            options,
            Star(),
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
