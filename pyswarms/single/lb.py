# -*- coding: utf-8 -*-

""" lb.py: local-best partical swarm optimization algorithm """

import numpy as np 
from ..base import SwarmBase
from ..utils.console_utils import cli_print, end_report

accepted_neighborhoods = ['random', 'dist']

class LBestPSO(SwarmBase):
    """A local-best Particle Swarm Optimization algorithm.

    Similar to global-best PSO, it takes a set of candidate solutions,
    and finds the best solution using a position-velocity update method.
    However, it uses a ring topology, thus making the particles 
    attracted to its corresponding neighborhood.

    In this implementation, a neighbor is selected via two means: (1) by
    random, and (2) by closest distance. For #2, the Euclidean distance
    is computed. Empirically, this is much slower.

    .. note:: This algorithm was adapted from one of the earlier works
        of J. Kennedy and R.C. Eberhart in Particle Swarm Optimization
        [1]_ [2]_

    .. [1] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
        Proceedings of the IEEE International Joint Conference on Neural
        Networks, 1995, pp. 1942-1948.

    .. [2] J. Kennedy and R.C. Eberhart, "A New Optimizer using Particle
        Swarm Theory,"  in Proceedings of the Sixth International 
        Symposium on Micromachine and Human Science, 1995, pp. 39–43.
    """
    def assertions(self):
        """Assertion method to check various inputs."""
        super(LBestPSO, self).assertions()

        # Assert keyword arguments
        assert 'c1' in self.kwargs, "Missing c1 key in kwargs."
        assert 'c2' in self.kwargs, "Missing c2 key in kwargs."
        assert 'm' in self.kwargs, "Missing m key in kwargs."

        assert self.n_neighbors <= self.n_particles, "No. of neighbors must be less than or equal no. of particles"
        assert neighborhood in accepted_neighborhoods, "Non-understandable neighborhood type"

    def __init__(self, n_particles, dims, bounds=None, 
        n_neighbors=1, neighborhood='random', **kwargs):
        """Initializes the swarm.

        Takes the same attributes as SwarmBase, but also initializes
        a velocity component by sampling from a random distribution
        with range [0,1].

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
        n_neighbors: int (default is 1, must be less than n_particles)
            number of neighbors to be considered.
        neighborhood: str {'random', 'dist'}, (default is 'random')
            type of neighbors to be searched.
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
        super(LBestPSO, self).__init__(n_particles, dims, bounds, **kwargs)

        # Store n_neighbors and neighborhood type
        self.n_neighbors = n_neighbors
        self.neighborhood = neighborhood

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
            the local best cost and the local best position among the
            swarm.
        """
        pass

    def reset(self):
        """Resets the attributes of the optimizer."""
        super(LBestPSO, self).reset()

        # Initialize velocity vectors
        self.velocity = np.random.random_sample(size=self.swarm_size)

        # Initialize the local best of the swarm
        self.lbest_cost = np.inf
        self.lbest_pos = None

        # Initialize the personal best of each particle
        self.pbest_pos = self.pos

    def _update_velocity_position(self):
        """Updates the velocity and position of the swarm.

        Specifically, it updates the attributes self.velocity and
        self.pos. This function is being called by the
        self.optimize() method
        """
        pass