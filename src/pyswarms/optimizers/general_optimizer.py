# -*- coding: utf-8 -*-

r"""
A general Particle Swarm Optimization (general PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Uses a user specified
topology.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
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
    from pyswarms.backend.topology import Pyramid
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters and topology
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    my_topology = Pyramid(static=False)

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                        options=options, topology=my_topology)

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

from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.topology import Topology
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.optimizers.optimizer import OptimizerPSO
from pyswarms.utils.types import (
    BoundaryStrategy,
    Bounds,
    Clamp,
    Position,
    SwarmOptions,
    VelocityStrategy,
)


class GeneralOptimizerPSO(OptimizerPSO):
    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        options: SwarmOptions,
        topology: Topology,
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
        topology : pyswarms.backend.topology.Topology
            A :code:`Topology` object that defines the topology to use in the
            optimization process. The currently available topologies are:
                * Star
                    All particles are connected
                * Ring (static and dynamic)
                    Particles are connected to the k nearest neighbours
                * VonNeumann
                    Particles are connected in a VonNeumann topology
                * Pyramid (static and dynamic)
                    Particles are connected in N-dimensional simplices
                * Random (static and dynamic)
                    Particles are connected to k random particles
                Static variants of the topologies remain with the same
                neighbours over the course of the optimization. Dynamic
                variants calculate new neighbours every time step.
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
            topology,
            VelocityUpdater(options, velocity_clamp, vh_strategy, bounds),
            PositionUpdater(bounds, bh_strategy),
            center,
            ftol,
            ftol_iter,
            init_pos,
        )
        self.name = __name__
