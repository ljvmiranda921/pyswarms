# -*- coding: utf-8 -*-

""" gb.py: global-best partical swarm optimization algorithm """

import numpy as np
from ..base import SwarmBase
from ..utils.console_utils import cli_print

class GBestPSO(SwarmBase):
    """A global-best Particle Swarm Optimization (PSO) algorithm.

    It takes a set of candidate solutions, and tries to find the best
    solution using a position-velocity update method.

    Algorithm adapted from,

    @InProceedings{Kennedy1995,
        author={J.Kennedy and R.C. Eberhart},
        title={Particle Swarm Optimization},
        book={Proceedings of the IEEE International Joint Conference
            on Neural Networks},
        pages={1942-1948},
        year={1995}
    }

    """
    def assertions(self):
        """Assertion method to check various inputs."""

        assert 'c1' in self.kwargs, "Missing c1 key in kwargs."
        assert 'c2' in self.kwargs, "Missing c2 key in kwargs."
        assert 'm' in self.kwargs, "Missing m key in kwargs."

    def __init__(self, n_particles, dims, bounds=None, **kwargs):
        """Initializes the swarm. 

        Takes the same attributes as SwarmBase, but also
        initializes a velocity component by sampling from a random
        distribution with range [0,1].

        Inputs:
            - bounds: (np.ndarray, np.ndarray)  a tuple of np.ndarrays
                where the first entry is the minimum bound while the
                second entry is the maximum bound. Each array must be
                of shape (dims,).
            - **kwargs: (dictionary) containing the following keys:
                * c1: cognitive parameter
                * c2: social parameter
                * m: momentum parameter
        """
        super(GBestPSO, self).__init__(n_particles, dims, bounds, **kwargs)

        # Initialize the resettable attributes
        self.reset()

        # Invoke assertions
        self.assertions()

    def optimize(self, f, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function `f` for a number of iterations `iter.`

        Inputs:
            - f: (method) objective function to be evaluated
            - iters: (int) nb. of iterations
            - print_step: amount of steps for printing into console.
            - verbose: verbosity setting

        Returns:
            - (tuple) the global best cost and the global best
                position.
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
                    (i+1, iters, self.gbest_cost), verbose, 1)

            # Perform velocity and position updates
            self._update_velocity_position()

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