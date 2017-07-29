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
:math:`m` controls the inertia of the swarm's movement. 

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import sphere_func

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'm':0.9}

    # Call instance of GBestPSO
    optimizer = ps.single.GBestPSO(n_particles=10, dims=2, **options)

    # Perform optimization
    stats = optimizer.optimize(sphere_func, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

# Import modules
import numpy as np

# Import from package
from ..base import SwarmBase
from ..utils.console_utils import cli_print, end_report

class GBestPSO(SwarmBase):

    def assertions(self):
        """Assertion method to check various inputs.

        Raises
        ------
        KeyError
            When one of the required dictionary keys is missing.
        """
        super(GBestPSO, self).assertions()

        if 'c1' not in self.kwargs:
            raise KeyError('Missing c1 key in kwargs.')
        if 'c2' not in self.kwargs:
            raise KeyError('Missing c2 key in kwargs.')
        if 'm' not in self.kwargs:
            raise KeyError('Missing m key in kwargs.')

    def __init__(self, n_particles, dims, bounds=None, **kwargs):
        """Initializes the swarm. 

        Takes the same attributes as :code:`SwarmBase`, but also
        initializes a velocity component by sampling from a random
        distribution with range :code:`[0,1]`.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dims : int
            number of dimensions in the space.
        bounds : tuple of :code:`np.ndarray`, optional (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dims,)`.
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
        function :code:`f` for a number of iterations :code:`iter.`

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
            # Create 2-D mask to update positions
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

        Specifically, it updates the attributes :code:`self.velocity`
        and :code:`self.pos`. This function is being called by the
        :code:`self.optimize()` method
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