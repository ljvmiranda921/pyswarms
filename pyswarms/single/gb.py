# -*- coding: utf-8 -*-

""" gb.py: global-best partical swarm optimization algorithm """

import numpy as np
from ..base import SwarmBase
from ..utils.console_utils import cli_print, end_report

class GBestPSO(SwarmBase):
    """A global-best Particle Swarm Optimization (PSO) algorithm.

    It takes a set of candidate solutions, and tries to find the best
    solution using a position-velocity update method. Uses a 
    star-topology where each particle is attracted to the best 
    performing particle.

    .. note:: This algorithm was adapted from the earlier works of J.
        Kennedy and R.C. Eberhart in Particle Swarm Optimization [1]_

    .. [1] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
        Proceedings of the IEEE International Joint Conference on Neural
        Networks, 1995, pp. 1942-1948.

    """
    def assertions(self):
        """Assertion method to check various inputs."""
        super(GBestPSO, self).assertions()

        # Assert keyword arguments
        assert 'c1' in self.kwargs, "Missing c1 key in kwargs."
        assert 'c2' in self.kwargs, "Missing c2 key in kwargs."
        assert 'm' in self.kwargs, "Missing m key in kwargs."

    def __init__(self, n_particles, dims, bounds=None, **kwargs):
        """Initializes the swarm. 

        Takes the same attributes as SwarmBase, but also
        initializes a velocity component by sampling from a random
        distribution with range [0,1].

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dims : int
            number of dimensions in the space.
        bounds : tuple of np.ndarray, optional (default is None)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape (dims,).
        **kwargs : dict
            Keyword argument that must contain the following dictionary
            keys:
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * m : float
                    momentum parameter
        """
        super(GBestPSO, self).__init__(n_particles, dims, bounds, **kwargs)

        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()

    def optimize(self, f, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function `f` for a number of iterations `iter.`

        Parameters
        ----------
        f : function
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
            the global best cost and the global best position.
        """
        for i in range(iters):
            # Compute cost for current position and personal best
            current_cost = f(self.pos)
            pbest_cost = f(self.pbest_pos)

            # Update personal bests if the current position is better
            # Create 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)
            # Create 2-D mask
            _m = np.repeat(m[:,np.newaxis], self.dims, axis=1)
            self.pbest_pos = np.where(~_m, self.pbest_pos, self.pos)

            # Get the minima of the pbest and check if it's less than
            # the saved gbest
            if np.min(pbest_cost) < self.gbest_cost:
                self.gbest_cost = np.min(pbest_cost)
                self.gbest_pos = self.pbest_pos[np.argmin(pbest_cost)]

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                    (i+1, iters, self.gbest_cost), verbose, 2)

            # Perform velocity and position updates
            self._update_velocity_position()

        end_report(self.gbest_cost, self.gbest_pos, verbose)
        return (self.gbest_cost, self.gbest_pos)

    def reset(self):
        """Resets the attributes of the optimizer."""
        super(GBestPSO, self).reset()

        # Initialize velocity vectors
        self.velocity = np.random.random_sample(size=self.swarm_size)

        # Initialize the global best of the swarm
        self.gbest_cost = np.inf
        self.gbest_pos = None

        # Initialize the personal best of each particle
        self.pbest_pos = self.pos

    def _update_velocity_position(self):
        """Updates the velocity and position of the swarm.

        Specifically, it updates the attributes self.velocity and
        self.pos. This function is being called by the
        self.optimize() method
        """

        # Define the hyperparameters from kwargs dictionary
        c1, c2, m = self.kwargs['c1'], self.kwargs['c2'], self.kwargs['m']

        # Compute for cognitive and social terms
        cognitive = (c1 * np.random.uniform(0,1,self.swarm_size)
                    * (self.pbest_pos - self.pos))
        social = (c2 * np.random.uniform(0,1,self.swarm_size)
                    * (self.gbest_pos - self.pos))
        self.velocity = (m * self.velocity) + cognitive + social

        # Update position and store it in a temporary variable
        temp = self.pos.copy()
        temp += self.velocity

        if self.bounds is not None:
            # Create a mask depending on the set boundaries
            b = (np.all(self.min_bounds <= temp, axis=1)
                * np.all(temp <= self.max_bounds, axis=1))
            # Broadcast the mask
            b = np.repeat(b[:,np.newaxis], self.dims, axis=1)
            # Use the mask to finally guide position update
            temp = np.where(~b, self.pos, temp)
        self.pos = temp