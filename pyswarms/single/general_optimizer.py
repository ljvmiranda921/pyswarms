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

import multiprocessing as mp
from collections import deque
from typing import Any, Callable, Deque, Dict, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from tqdm import trange

from pyswarms.backend.handlers import (
    BoundaryHandler,
    BoundaryStrategy,
    OptionsHandler,
    OptionsStrategy,
    VelocityHandler,
    VelocityStrategy,
)
from pyswarms.backend.operators import compute_objective_function, compute_pbest
from pyswarms.backend.topology import Topology
from pyswarms.base import SwarmOptimizer
from pyswarms.base.base import Options, ToHistory
from pyswarms.utils.types import Bounds, Clamp, Position


class GeneralOptions(Options):
    p: Literal[1, 2]
    k: int


class GeneralOptimizerPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        options: GeneralOptions,
        topology: Topology,
        bounds: Optional[Bounds] = None,
        oh_strategy: Optional[Dict[str, OptionsStrategy]] = None,
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
        options : dict with keys :code:`{'c1', 'c2', 'w'}` or :code:`{'c1',
                'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                if used with the :code:`Ring`, :code:`VonNeumann` or
                :code:`Random` topology the additional parameter k must be
                included
                * k : int
                    number of neighbors to be considered. Must be a positive
                    integer less than :code:`n_particles`
                if used with the :code:`Ring` topology the additional
                parameters k and p must be included
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the sum-of-absolute
                    values (or L1 distance) while 2 is the Euclidean (or L2)
                    distance.
                if used with the :code:`VonNeumann` topology the additional
                parameters p and r must be included
                * r: int
                    the range of the VonNeumann topology.  This is used to
                    determine the number of neighbours in the topology.
        topology : pyswarms.backend.topology.Topology
            a :code:`Topology` object that defines the topology to use in the
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
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
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
        super(GeneralOptimizerPSO, self).__init__(
            n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )

        if oh_strategy is None:
            oh_strategy = {}

        # Initialize the resettable attributes
        self.reset()

        self.top = topology
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler.factory(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.name = __name__

    def optimize(
        self,
        objective_func: Callable[..., npt.NDArray[Any]],
        iters: int,
        n_processes: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any
    ) -> Tuple[float, Position]:
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        log_level = "DEBUG" if verbose else "TRACE"
        logger.debug("Obj. func. args: {}".format(kwargs))
        logger.log(log_level, "Optimize for {} iters with {}".format(iters, self.options))

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history: Deque[bool] = deque(maxlen=self.ftol_iter)

        pbar = trange(iters, desc=self.name) if verbose else range(iters)
        for i in pbar:
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            best_cost_yet_found = self.swarm.best_cost

            # Update swarm
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

            # Print to console
            # if verbose:
            #     pbar.postfix(best_cost=self.swarm.best_cost)

            hist = ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=float(np.mean(self.swarm.pbest_cost)),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)

            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break

            # Perform options update
            self.swarm.options = self.oh(self.options, iternow=i, itermax=iters)

            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp, self.vh, self.bounds)
            self.swarm.position = self.top.compute_position(self.swarm, self.bounds, self.bh)

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()

        # Write report in log and return final cost and position
        logger.log(
            log_level,
            "Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos),
        )

        # Close Pool of Processes
        if n_processes is not None:
            pool.close()  # type: ignore

        return (final_best_cost, final_best_pos)
