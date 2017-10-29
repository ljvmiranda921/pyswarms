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
PSO. However, this whole behavior can be modified into a global-best PSO
by changing the nearest neighbours equal to the number of particles in
the swarm. In this case, all particles see each other, and thus a global
best particle can be established.

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

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import logging
import numpy as np
from scipy.spatial import cKDTree
from past.builtins import xrange

# Import from package
from ..base import DiscreteSwarmBase
from ..utils.console_utils import cli_print, end_report


class BinaryPSO(DiscreteSwarmBase):

    def assertions(self):
        """Assertion method to check various inputs.

        Raises
        ------
        KeyError
            When one of the required dictionary keys is missing.
        ValueError
            When the number of neighbors is not within the range
                :code:`[0, n_particles]`.
            When the p-value is not in the list of values :code:`[1,2]`.
        """
        super(BinaryPSO, self).assertions()

        if not all(key in self.options for key in ('k', 'p')):
            raise KeyError('Missing either k or p in options')
        if not 0 <= self.k <= self.n_particles:
            raise ValueError('No. of neighbors must be between 0 and no. of'
                             'particles.')
        if self.p not in [1, 2]:
            raise ValueError('p-value should either be 1 (for L1/Minkowski)'
                             'or 2 (for L2/Euclidean).')

    def __init__(self, n_particles, dimensions, options, velocity_clamp=None):
        """Initializes the swarm.

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

        binary = True
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options['k'], options['p']
        # Initialize parent class
        super(BinaryPSO, self).__init__(n_particles, dimensions, binary,
                                        options, velocity_clamp)
        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()

    def optimize(self, objective_func, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

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

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        for i in xrange(iters):
            # Compute cost for current position and personal best
            current_cost = objective_func(self.pos)
            pbest_cost = objective_func(self.personal_best_pos)

            # Update personal bests if the current position is better
            # Create a 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)
            # Create a 2-D mask to update positions
            _m = np.repeat(m[:, np.newaxis], self.dimensions, axis=1)
            self.personal_best_pos = np.where(~_m, self.personal_best_pos,
                                              self.pos)

            # Obtain the indices of the best position for each
            # neighbour-space, and get the local best cost and
            # local best positions from it.
            nmin_idx = self._get_neighbors(pbest_cost)
            self.best_cost = pbest_cost[nmin_idx]
            self.best_pos = self.personal_best_pos[nmin_idx]

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                          (i+1, iters, np.min(self.best_cost)), verbose, 2,
                          logger=self.logger)

            # Save to history
            hist = self.ToHistory(
                best_cost=np.min(self.best_cost),
                mean_pbest_cost=np.mean(pbest_cost),
                mean_neighbor_cost=np.mean(self.best_cost),
                position=self.pos,
                velocity=self.velocity
            )
            self._populate_history(hist)

            # Perform position velocity update
            self._update_velocity()
            self._update_position()

        # Obtain the final best_cost and the final best_position
        final_best_cost_arg = np.argmin(self.best_cost)
        final_best_cost = np.min(self.best_cost)
        final_best_pos = self.best_pos[final_best_cost_arg]

        end_report(final_best_cost, final_best_pos, verbose,
                   logger=self.logger)
        return final_best_cost, final_best_pos

    def _get_neighbors(self, pbest_cost):
        """Helper function to obtain the best position found in the
        neighborhood. This uses the cKDTree method from :code:`scipy`
        to obtain the nearest neighbours

        Parameters
        ----------
        pbest_cost : numpy.ndarray of size (n_particles, )
            the cost incurred at the historically best position. Will be used
            for mapping the obtained indices to its actual cost.

        Returns
        -------
        array of size (n_particles, ) dtype=int64
            indices containing the best particles for each particle's
            neighbour-space that have the lowest cost
        """
        # Use cKDTree to get the indices of the nearest neighbors
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, p=self.p, k=self.k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if self.k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = pbest_cost[idx][:, np.newaxis].argmin(axis=1)
        else:
            idx_min = pbest_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]

        return best_neighbor

    def _update_velocity(self):
        """Updates the velocity matrix of the swarm.

        This method updates the attribute :code:`self.velocity` of
        the instantiated object. It is called by the
        :code:`self.optimize()` method.
        """
        # Define the hyperparameters from options dictionary
        c1, c2, w = self.options['c1'], self.options['c2'], self.options['w']

        # Compute for cognitive and social terms
        cognitive = (c1 * np.random.uniform(0, 1, self.swarm_size)
                     * (self.personal_best_pos - self.pos))
        social = (c2 * np.random.uniform(0, 1, self.swarm_size)
                  * (self.best_pos - self.pos))
        temp_velocity = (w * self.velocity) + cognitive + social

        # Create a mask to clamp the velocities
        if self.velocity_clamp is not None:
            # Create a mask depending on the set boundaries
            min_velocity, max_velocity = self.velocity_clamp[0], \
                                         self.velocity_clamp[1]
            _b = np.logical_and(temp_velocity >= min_velocity,
                                temp_velocity <= max_velocity)
            # Use the mask to finally clamp the velocities
            self.velocity = np.where(~_b, self.velocity, temp_velocity)
        else:
            self.velocity = temp_velocity

    def _update_position(self):
        """Updates the position matrix of the swarm.

        This method updates the attribute :code:`self.pos` of
        the instantiated object. It is called by the
        :code:`self.optimize()` method.
        """
        self.pos = (np.random.random_sample(size=self.swarm_size) <
                    self._sigmoid(self.velocity)) * 1

    def _sigmoid(self, x):
        """Helper sigmoid function.

        Inputs
        ------
        x : numpy.ndarray
            Input vector to compute the sigmoid from

        Returns
        -------
        numpy.ndarray
            Output sigmoid computation
        """
        return 1 / (1 + np.exp(x))
