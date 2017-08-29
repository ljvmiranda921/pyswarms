# -*- coding: utf-8 -*-

"""
Hyperparameter grid search.

Compares the relative performance of hyperparameter value combinations in
reducing a specified objective function.

For each hyperparameter, user can provide either a single value or a list
of possible values, and their cartesian product is taken to produce a grid
of all possible combinations. These combinations are then tested to produce
a list of objective function scores. The default of the optimize method
returns the hyperparameters that yield the minimum score, yet maximum score
can also be evaluated.

Parameters
----------
* c1 : float
    cognitive parameter
* c2 : float
    social parameter
* w : float
    inertia parameter
* k : int
    number of neighbors to be considered. Must be a
    positive integer less than `n_particles`
* p: int {1,2}
    the Minkowski p-norm to use. 1 is the
    sum-of-absolute values (or L1 distance) while 2 is
    the Euclidean (or L2) distance.

>>> options = {'c1': [1, 2, 3],
               'c2': [1, 2, 3],
               'w' : [2, 3, 5],
               'k' : [5, 10, 15],
               'p' : 1}
>>> g = GridSearch(LocalBestPSO, n_particles=40, dimensions=20,
                   options=options, objective_func=sphere_func, iters=10)
>>> best_score, best_options = g.optimize()
>>> best_score
301.418815268
>>> best_options['c1']
1
>>> best_options['c2']
1
"""

import operator as op
import itertools
import numpy as np

class GridSearch(object):
    """Exhaustive search of optimal performance on selected objective function
    over all combinations of specified hyperparameter values."""

    def __init__(self, optimizer, n_particles, dimensions, options,
                 objective_func, iters, bounds=None, velocity_clamp=None):
        """Initializes the GridSearch.

        Attributes
        ----------
        optimizer: PySwarms class
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

    def generate_grid(self):
        """Generates the grid of all hyperparameter value combinations."""

        #Extract keys and values from options dictionary
        params = self.options.keys()
        items = [x if type(x) == list \
                 else [x] for x in list(zip(*self.options.items()))[1]]

        #Create list of cartesian products of hyperparameter values from options
        list_of_products = list(itertools.product(*items))

        #Return list of dicts for all hyperparameter value combinations
        return [dict(zip(*[params, list(x)])) for x in list_of_products]

    def generate_score(self, options):
        """Generates score for optimizer's performance on objective function."""

        #Intialize optimizer
        f = self.optimizer(self.n_particles, self.dims, options,
                        self.bounds, self.vclamp)

        #Return score
        return f.optimize(self.objective_func, self.iters)[0]

    def search(self, maximum=False):
        """Compares optimizer's objective function performance scores
        for all combinations of provided parameters."""

        #Assign parameter keys
        params = self.options.keys()

        #Generate the grid of all hyperparameter value combinations
        grid = self.generate_grid()

        #Calculate scores for all hyperparameter combinations
        scores = [self.generate_score(i) for i in grid]

        #Select optimization function
        if maximum:
            idx, self.best_score = max(enumerate(scores), key=op.itemgetter(1))
        else:
            idx, self.best_score = min(enumerate(scores), key=op.itemgetter(1))

        #Return optimum hyperparameter value property from grid using index
        self.best_options = op.itemgetter(idx)(grid)
        return self.best_score, self.best_options
