# -*- coding: utf-8 -*-

r"""
A Binary Particle Swarm Optimization (binary PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Unlike
:mod:`pyswarms.single.gb` and :mod:`pyswarms.single.lb`, this technique
is often applied to discrete binary problems such as job-shop scheduling,
sequencing, and the like.

The update rule for the velocity is still similar, as shown in the
proceeding equation:

.. math::

   v_{ij}(t + 1) = w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

For the velocity update rule, a particle compares its current position
with respect to its neighbours. The nearest neighbours are being
determined by a kD-tree given a distance metric, similar to local-best
PSO. The neighbours are computed for every iteration. However, this whole
behavior can be modified into a global-best PSO by changing the nearest
neighbours equal to the number of particles in the swarm. In this case,
all particles see each other, and thus a global best particle can be established.

In addition, one notable change for binary PSO is that the position
update rule is now decided upon by the following case expression:

.. math::

   X_{ij}(t+1) = \left\{\begin{array}{lr}
        0, & \text{if } \text{rand() } \geq S(v_{ij}(t+1))\\
        1, & \text{if } \text{rand() } < S(v_{ij}(t+1))
        \end{array}\right\}

Where the function :math:`S(x)` is the sigmoid function defined as:

.. math::

   S(x) = \dfrac{1}{1 + e^{-x}}

This enables the algorithm to output binary positions rather than
a stream of continuous values as seen in global-best or local-best PSO.

This algorithm was adapted from the standard Binary PSO work of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [SMC1997]_.

.. [SMC1997] J. Kennedy and R.C. Eberhart, "A discrete binary version of
    particle swarm algorithm," Proceedings of the IEEE International
    Conference on Systems, Man, and Cybernetics, 1997.
"""

import multiprocessing as mp
from collections import deque
from typing import Any, Callable, Deque, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from loguru import logger
from tqdm import trange

from pyswarms.backend.operators import compute_objective_function, compute_pbest
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology import Ring
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.optimizers.base import BaseSwarmOptimizer, ToHistory
from pyswarms.utils.types import Position


class BinaryPSO(BaseSwarmOptimizer):
    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        p: Literal[1, 2],
        k: int,
        velocity_updater: VelocityUpdater,
        position_updater: PositionUpdater,
        init_pos: Optional[Position] = None,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
        **kwargs: Any
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.
        k : int
            number of neighbors to be considered. Must be a
            positive integer less than :code:`n_particles`
        velocity_updater : VelocityUpdater
            Class for updating the velocity matrix.
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        vh_strategy : String
            a strategy for the handling of the velocity of out-of-bounds particles.
            Only the "unmodified" and the "adjust" strategies are allowed.
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        """
        self.p = p
        self.k = k
        self.binary = True

        # Initialize parent class
        super().__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            velocity_updater=velocity_updater,
            position_updater=position_updater,
            init_pos=init_pos,
            ftol=ftol,
            ftol_iter=ftol_iter,
        )
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Ring(self.p, self.k, static=False)
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
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int, optional
            number of processes to use for parallel particle evaluation
            Defaut is None with no parallelization.
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for objective function

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        log_level = "DEBUG" if verbose else "TRACE"
        logger.debug("Obj. func. args: {}".format(kwargs))

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history: Deque[bool] = deque(maxlen=self.ftol_iter)

        pbar = trange(iters, desc=self.name) if verbose else range(iters)
        for i in pbar:
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            best_cost_yet_found = np.min(self.swarm.best_cost)

            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

            # Print to console
            if verbose:
                pbar.set_postfix(best_cost=self.swarm.best_cost)  # type: ignore

            # Save to history
            hist = ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=float(np.mean(self.swarm.pbest_cost)),
                mean_neighbor_cost=float(np.mean(self.swarm.best_cost)),
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

            # Perform position velocity update
            self.swarm.velocity = self.velocity_updater.compute(self.swarm, i, iters)
            self.swarm.position = self._compute_position(self.swarm)

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        logger.log(
            log_level,
            "Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos),
        )

        # Close Pool of Processes
        if n_processes is not None:
            pool.close()  # type: ignore

        return (final_best_cost, final_best_pos)

    def _compute_position(self, swarm: Swarm):
        """Update the position matrix of the swarm

        This computes the next position in a binary swarm. It compares the
        sigmoid output of the velocity-matrix and compares it with a randomly
        generated matrix.

        Parameters
        ----------
        swarm: pyswarms.backend.swarms.Swarm
            a Swarm class
        """
        return (np.random.random_sample(size=swarm.dimensions) < self._sigmoid(swarm.velocity)) * 1

    def _sigmoid(self, x: npt.NDArray[Any]):
        """Helper method for the sigmoid function

        Parameters
        ----------
        x : numpy.ndarray
            Input vector for sigmoid computation

        Returns
        -------
        numpy.ndarray
            Output sigmoid computation
        """
        return 1 / (1 + np.exp(-x))

    def _init_swarm(self):
        position = self.position_updater.generate_discrete_position(
            self.n_particles, self.dimensions, self.binary, self.init_pos
        )
        velocity = self.velocity_updater.generate_velocity(self.n_particles, self.dimensions)
        self.swarm = Swarm(position, velocity)
