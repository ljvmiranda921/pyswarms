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

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

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

# Import from stdlib
import logging

# Import modules
import numpy as np

# Import from package
from ..base import DiscreteSwarmOptimizer
from ..backend.operators import compute_pbest
from ..backend.topology import Ring
from ..utils.console_utils import cli_print, end_report


class BinaryPSO(DiscreteSwarmOptimizer):
    def assertions(self):
        """Check inputs and throw assertions

        Raises
        ------
        KeyError
            When one of the required dictionary keys is missing.
        ValueError
            When the number of neighbors is not within the range :code:`[0, n_particles]`.
            When the p-value is not in the list of values :code:`[1,2]`.
        """
        super(BinaryPSO, self).assertions()

        if not all(key in self.options for key in ("k", "p")):
            raise KeyError("Missing either k or p in options")
        if not 0 <= self.k <= self.n_particles:
            raise ValueError(
                "No. of neighbors must be between 0 and no. of" "particles."
            )
        if self.p not in [1, 2]:
            raise ValueError(
                "p-value should either be 1 (for L1/Minkowski)"
                "or 2 (for L2/Euclidean)."
            )

    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        init_pos=None,
        velocity_clamp=None,
        ftol=-np.inf,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        options : dict with keys :code:`{'c1', 'c2', 'k', 'p'}`
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
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options["k"], options["p"]
        # Initialize parent class
        super(BinaryPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            binary=True,
            options=options,
            init_pos=init_pos,
            velocity_clamp=velocity_clamp,
            ftol=ftol,
        )
        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology
        self.top = Ring(static=False)

    def optimize(self, objective_func, iters, print_step=1, verbose=1, **kwargs):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        print_step : int (the default is 1)
            amount of steps for printing into console.
        verbose : int  (the default is 1)
            verbosity setting.
        kwargs : dict
            arguments for objective function

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        cli_print("Arguments Passed to Objective Function: {}".format(kwargs),
                  verbose, 2, logger=self.logger)

        for i in range(iters):
            # Compute cost for current position and personal best
            self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
            self.swarm.pbest_cost = objective_func(self.swarm.pbest_pos, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = np.min(self.swarm.best_cost)
            # Update gbest from neighborhood
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, self.p, self.k
            )
            # Print to console
            if i % print_step == 0:
                cli_print(
                    "Iteration {}/{}, cost: {}".format(i + 1, iters, np.min(self.swarm.best_cost)),
                    verbose,
                    2,
                    logger=self.logger,
                )
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
            if (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            ):
                break
            # Perform position velocity update
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp
            )
            self.swarm.position = self._compute_position(self.swarm)
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.best_pos.copy()
        end_report(
            final_best_cost, final_best_pos, verbose, logger=self.logger
        )
        return (final_best_cost, final_best_pos)

    def _compute_position(self, swarm):
        """Update the position matrix of the swarm

        This computes the next position in a binary swarm. It compares the
        sigmoid output of the velocity-matrix and compares it with a randomly
        generated matrix.

        Parameters
        ----------
        swarm: pyswarms.backend.swarms.Swarm
            a Swarm class
        """
        return (
            np.random.random_sample(size=swarm.dimensions)
            < self._sigmoid(swarm.velocity)
        ) * 1

    def _sigmoid(self, x):
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
