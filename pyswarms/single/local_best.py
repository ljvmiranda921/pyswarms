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

# Import standard library
import logging

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque

from ..backend.operators import compute_pbest, compute_objective_function
from ..backend.topology import Ring
from ..backend.handlers import BoundaryHandler, VelocityHandler, OptionsHandler
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter


class LocalBestPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        oh_strategy=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
        static=False,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
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
        options : dict with keys :code:`{'c1', 'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        static: bool
            a boolean that decides whether the Ring topology
            used is static or dynamic. Default is `False`
        """
        if oh_strategy is None:
            oh_strategy = {}
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options["k"], options["p"]
        # Initialize parent class
        super(LocalBestPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Ring(static=static)
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.oh = OptionsHandler(strategy=oh_strategy)
        self.name = __name__

    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
    ):
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
            the local best cost and the local best position among the
            swarm.
        """
        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            self.swarm.current_cost = compute_objective_function(
                self.swarm, objective_func, pool=pool, **kwargs
            )
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, p=self.p, k=self.k
            )
            if verbose:
                self.rep.hook(best_cost=np.min(self.swarm.best_cost))
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=np.mean(self.swarm.best_cost),
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = self.oh(
                self.options, iternow=i, itermax=iters
            )
            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        return (final_best_cost, final_best_pos)
