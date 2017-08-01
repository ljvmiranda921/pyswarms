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
this k-D tree.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'm':0.9}

    # Call instance of LBestPSO with a neighbour-size of 3 determined by
    # the L2 (p=2) distance.
    optimizer = ps.single.LBestPSO(n_particles=10, dims=2, k=3, p=2, **options)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere_func, iters=100)

This algorithm was adapted from one of the earlier works of
J. Kennedy and R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_ [MHS1995]

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

.. [MHS1995] J. Kennedy and R.C. Eberhart, "A New Optimizer using Particle
    Swarm Theory,"  in Proceedings of the Sixth International 
    Symposium on Micromachine and Human Science, 1995, pp. 39–43.
"""

# Import modules
import numpy as np 
from scipy.spatial import cKDTree

# Import from package
from ..base import SwarmBase
from ..utils.console_utils import cli_print, end_report

class LBestPSO(SwarmBase):

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
        super(LBestPSO, self).assertions()

        if not all (key in self.kwargs for key in ('c1', 'c2', 'w')):
            raise KeyError('Missing either c1, c2, or w in kwargs')

        if not 0 <= self.k <= self.n_particles:
            raise ValueError('No. of neighbors must be between 0 and no. of particles.')
        if self.p not in [1,2]:
            raise ValueError('p-value should either be 1 (for L1/Minkowski) or 2 (for L2/Euclidean).')

    def __init__(self, n_particles, dims, bounds=None, v_clamp=None,
        k=1, p=2, **kwargs):
        """Initializes the swarm.

        Takes the same attributes as SwarmBase, but also initializes
        a velocity component by sampling from a random distribution
        with range :code:`[0,1]`.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dims : int
            number of dimensions in the space.
        bounds : tuple of np.ndarray, optional (default is None)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dims,)`.
        v_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It 
            sets the limits for velocity clamping. 
        k: int (default is 1, must be less than :code:`n_particles`)
            number of neighbors to be considered.
        p: int {1,2} (default is 2) 
            the Minkowski p-norm to use. 1 is the sum-of-absolute values
            (or L1 distance) while 2 is the Euclidean (or L2) distance.
        **kwargs : dict
            Keyword argument that must contain the following dictionary
            keys:
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        """

        # Store n_neighbors and neighborhood type
        self.k = k
        self.p = p

        super(LBestPSO, self).__init__(n_particles, dims, bounds, v_clamp, **kwargs)

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
            the local best cost and the local best position among the
            swarm.
        """
        for i in range(iters):
            # Compute cost for current position and personal best
            current_cost = f(self.pos)
            pbest_cost = f(self.pbest_pos)

            # Update personal bests if the current position is better
            # Create a 1-D mask then update pbest_cost
            m = (current_cost < pbest_cost)
            pbest_cost = np.where(~m, pbest_cost, current_cost)
            # Create a 2-D mask to update positions
            _m = np.repeat(m[:, np.newaxis], self.dims, axis=1)
            self.pbest_pos = np.where(~_m, self.pbest_pos, self.pos)

            # Obtain the indices of the best position for each
            # neighbour-space, and get the local best cost and
            # local best positions from it.
            nmin_idx = self._get_neighbors(current_cost)
            self.lbest_cost = current_cost[nmin_idx]
            self.lbest_pos  = self.pos[nmin_idx]

            # Print to console
            if i % print_step == 0:
                cli_print('Iteration %s/%s, cost: %s' %
                    (i+1, iters, np.min(self.lbest_cost)), verbose, 2)

            # Perform position velocity update
            self._update_velocity_position()

        # Because we have multiple neighbor spaces, we are reporting the
        # local-best for each neighbour, thus giving us multiple values 
        # for the local-best cost and positions. What we'll do is that
        # we are going to obtain only the minimum of all these local
        # positions and then report it.

        self.best_neighbor_cost = np.argmin(self.lbest_cost)
        self.best_neighbor_pos = self.lbest_pos[self.best_neighbor_cost]

        end_report(self.best_neighbor_cost, self.best_neighbor_pos, verbose)
        return (self.best_neighbor_cost, self.best_neighbor_pos)

    def _get_neighbors(self, current_cost):
        """Helper function to obtain the best position found in the
        neighborhood. This uses the cKDTree method from :code:`scipy`
        to obtain the nearest neighbours
        
        Parameters
        ----------
        current_cost : numpy.ndarray of size (n_particles, )
            the cost incurred at the current position. Will be used for
            mapping the obtained indices to its actual cost.

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
            best_neighbor = current_cost[idx][:,np.newaxis].argmin(axis=1)
        else:
            idx_min = current_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]

        return best_neighbor

    def reset(self):
        """Resets the attributes of the optimizer."""
        super(LBestPSO, self).reset()

        # Initialize the local best of the swarm
        self.lbest_cost = np.inf
        self.lbest_pos = None

        # Initialize the personal best of each particle
        self.pbest_pos = self.pos

    def _update_velocity_position(self):
        """Updates the velocity and position of the swarm.

        Specifically, it updates the attributes :code:`self.velocity`
        and :code:`self.pos`. This function is being called by the
        :code:`self.optimize()` method
        """
        # Define the hyperparameters from kwargs dictionary
        c1, c2, w = self.kwargs['c1'], self.kwargs['c2'], self.kwargs['w']

        # Compute for cognitive and social terms
        cognitive = (c1 * np.random.uniform(0,1,self.swarm_size)
                    * (self.pbest_pos - self.pos))
        social = (c2 * np.random.uniform(0,1,self.swarm_size)
                    * (self.lbest_pos - self.pos))
        temp_velocity = (w * self.velocity) + cognitive + social

        # Create a mask to clamp the velocities
        if self.v_clamp is not None:
            # Create a mask depending on the set boundaries
            v_min, v_max = self.v_clamp[0], self.v_clamp[1]
            _b = np.logical_and(temp_velocity >= v_min, temp_velocity <= v_max)
            # Use the mask to finally clamp the velocities
            self.velocity = np.where(~_b, self.velocity, temp_velocity)
        else:
            self.velocity = temp_velocity

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