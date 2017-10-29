# -*- coding: utf-8 -*-
"""Base class for hyperparameter optimization search functions"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import operator as op


class SearchBase(object):

    def assertions(self):
        """Assertion method to check :code:`optimizer` input.

        Raises
        ------
        TypeError
            When :code:`optimizer` does not have an `'optimize'` attribute.
        """
        # Check type of optimizer object
        if not hasattr(self.optimizer, 'optimize'):
            raise TypeError('Parameter `optimizer` must have an '
                            '`\'optimize\'` attribute.')

    def __init__(self, optimizer, n_particles, dimensions, options,
                 objective_func, iters,
                 bounds=None, velocity_clamp=None):
        """Initializes the Search.

        Attributes
        ----------
        optimizer: pyswarms.single
            either LocalBestPSO or GlobalBestPSO
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
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
        objective_func: function
            objective function to be evaluated
        iters: int
            number of iterations
        bounds : tuple of np.ndarray, optional (default is None)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        """

        # Assign attributes
        self.optimizer = optimizer
        self.n_particles = n_particles
        self.dims = dimensions
        self.options = options
        self.bounds = bounds
        self.vclamp = velocity_clamp
        self.objective_func = objective_func
        self.iters = iters
        # Invoke assertions
        self.assertions()

    def generate_score(self, options):
        """Generates score for optimizer's performance on objective function.

        Parameters
        ----------

        options: dict
            a dict with the following keys: {'c1', 'c2', 'w', 'k', 'p'}
        """

        # Intialize optimizer
        f = self.optimizer(self.n_particles, self.dims, options,
                           self.bounds, self.vclamp)

        # Return score
        return f.optimize(self.objective_func, self.iters)[0]

    def search(self, maximum=False):
        """Compares optimizer's objective function performance scores
        for all combinations of provided parameters.

        Parameters
        ----------

        maximum: bool
            a bool defaulting to False, returning the minimum value for the
            objective function. If set to True, will return the maximum value
            for the objective function.
        """

        # Generate the grid of all hyperparameter value combinations
        grid = self.generate_grid()

        # Calculate scores for all hyperparameter combinations
        scores = [self.generate_score(i) for i in grid]

        # Default behavior
        idx, self.best_score = min(enumerate(scores), key=op.itemgetter(1))

        # Catches the maximum bool flag
        if maximum:
            idx, self.best_score = max(enumerate(scores),
                                       key=op.itemgetter(1))

        # Return optimum hyperparameter value property from grid using index
        self.best_options = op.itemgetter(idx)(grid)
        return self.best_score, self.best_options
