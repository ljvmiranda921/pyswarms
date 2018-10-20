# -*- coding: utf-8 -*-
"""
Hyperparameter grid search.

Compares the relative performance of hyperparameter value combinations in
optimizing a specified objective function.

For each hyperparameter, user can provide either a single value or a list
of possible values. The cartesian products of these hyperparameters are taken
to produce a grid of all possible combinations. These combinations are then
tested to produce a list of objective function scores. The search method
default returns the minimum objective function score and hyperparameters that
yield the minimum score, yet maximum score can also be evaluated.

>>> options = {'c1': [1, 2, 3],
               'c2': [1, 2, 3],
               'w' : [2, 3, 5],
               'k' : [5, 10, 15],
               'p' : 1}
>>> g = GridSearch(LocalBestPSO, n_particles=40, dimensions=20,
                   options=options, objective_func=sphere, iters=10)
>>> best_score, best_options = g.search()
>>> best_score
0.498641604188
>>> best_options['c1']
1
>>> best_options['c2']
1
"""

# Import from __future__
from __future__ import absolute_import, print_function, with_statement

# Import standard library
import itertools

# Import from pyswarms
# Import from package
from pyswarms.utils.search.base_search import SearchBase


class GridSearch(SearchBase):
    """Exhaustive search of optimal performance on selected objective function
    over all combinations of specified hyperparameter values."""

    def __init__(
        self,
        optimizer,
        n_particles,
        dimensions,
        options,
        objective_func,
        iters,
        bounds=None,
        velocity_clamp=(0, 1),
    ):
        """Initialize the Search"""

        # Assign attributes
        super(GridSearch, self).__init__(
            optimizer,
            n_particles,
            dimensions,
            options,
            objective_func,
            iters,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
        )
        # invoke assertions
        self.assertions()

    def generate_grid(self):
        """Generate the grid of all hyperparameter value combinations"""

        # Extract keys and values from options dictionary
        params = self.options.keys()
        items = [
            x if type(x) == list else [x]
            for x in list(zip(*self.options.items()))[1]
        ]

        # Create list of cartesian products of hyperparameter values
        # from options
        list_of_products = list(itertools.product(*items))

        # Return list of dicts for all hyperparameter value combinations
        return [dict(zip(*[params, list(x)])) for x in list_of_products]
