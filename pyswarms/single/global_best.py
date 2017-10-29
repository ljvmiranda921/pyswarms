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

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior in choosing how to
react given two choices: (1) to follow its *personal best* or (2) follow
the swarm's *global best* position. Overall, this dictates if the swarm
is explorative or exploitative in nature. In addition, a parameter
:math:`w` controls the inertia of the swarm's movement.

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
    stats = optimizer.optimize(fx.sphere_func, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import logging
import numpy as np
from past.builtins import xrange

# Import from package
from ..base import SwarmBase
from ..utils.console_utils import cli_print, end_report


class GlobalBestPSO(SwarmBase):

    def __init__(self, n_particles, dimensions, options,
                 bounds=None, velocity_clamp=None):
        """Initializes the swarm.

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
        bounds : tuple of :code:`np.ndarray` (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        """
        super(GlobalBestPSO, self).__init__(n_particles, dimensions, options,
                                            bounds, velocity_clamp)

        # Initialize logger
        self.logger = logging.getLogger(__name__)
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
        print_step : int (default is 1)
            amount of steps for printing into console.
        verbose : int  (default is 1)
            verbosity setting.

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        for i in xrange(iters):
            # Compute cost for current position and personal best
            current_cost = objective_func(self.pos)
            pbest_cost = objective_func(self.personal_best_pos)

            # Update personal bests if the current position is better
            # Create 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)
            # Create 2-D mask to update positions
            _m = np.repeat(m[:, np.newaxis], self.dimensions, axis=1)
            self.personal_best_pos = np.where(~_m, self.personal_best_pos,
                                              self.pos)

            # Get the minima of the pbest and check if it's less than
            # the saved gbest
            if np.min(pbest_cost) < self.best_cost:
                self.best_cost = np.min(pbest_cost)
                self.best_pos = self.personal_best_pos[np.argmin(pbest_cost)]

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                          (i+1, iters, self.best_cost), verbose, 2,
                          logger=self.logger)

            # Save to history
            hist = self.ToHistory(
                best_cost=self.best_cost,
                mean_pbest_cost=np.mean(pbest_cost),
                mean_neighbor_cost=self.best_cost,
                position=self.pos,
                velocity=self.velocity
            )
            self._populate_history(hist)

            # Perform velocity and position updates
            self._update_velocity()
            self._update_position()

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.best_cost.copy()  # Make deep copies
        final_best_pos = self.best_pos.copy()

        end_report(final_best_cost, final_best_pos, verbose,
                   logger=self.logger)
        return final_best_cost, final_best_pos

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
        # Update position and store it in a temporary variable
        temp = self.pos.copy()
        temp += self.velocity

        if self.bounds is not None:
            # Create a mask depending on the set boundaries
            b = (np.all(self.min_bounds <= temp, axis=1)
                 * np.all(temp <= self.max_bounds, axis=1))
            # Broadcast the mask
            b = np.repeat(b[:, np.newaxis], self.dimensions, axis=1)
            # Use the mask to finally guide position update
            temp = np.where(~b, self.pos, temp)
        self.pos = temp
