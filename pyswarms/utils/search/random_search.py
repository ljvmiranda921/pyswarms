# -*- coding: utf-8 -*-
"""
Hyperparameter random search.

Compares the relative performance of combinations of randomly generated
hyperparameter values in optimizing a specified objective function.

User provides lists of bounds for the uniform random value generation of
'c1', 'c2', and 'w', and the random integer value generation of 'k'.
Combinations of values are generated for the number of iterations specified,
and the generated grid of combinations is used in the search method to find
the optimal parameters for the objective function. The search method default
returns the minimum objective function score and hyperparameters that yield
the minimum score, yet maximum score can also be evaluated.


>>> options = {'c1': [1, 5],
               'c2': [6, 10],
               'w' : [2, 5],
               'k' : [11, 15],
               'p' : 1}
>>> g = RandomSearch(LocalBestPSO, n_particles=40, dimensions=20,
                   options=options, objective_func=sphere, iters=10)
>>> best_score, best_options = g.search()
>>> best_score
1.41978545901
>>> best_options['c1']
1.543556887693
>>> best_options['c2']
9.504769054771
"""

# Import from __future__
from __future__ import absolute_import, print_function, with_statement

# Import modules
import numpy as np
from past.builtins import xrange

# Import from pyswarms
# Import from package
from pyswarms.utils.search.base_search import SearchBase


class RandomSearch(SearchBase):
    """Search of optimal performance on selected objective function
    over combinations of randomly selected hyperparameter values
    within specified bounds for specified number of selection iterations."""

    def assertions(self):
        """Assertion method to check :code:`n_selection_iters` input

        Raises
        ------
        TypeError
            When :code:`n_selection_iters` is not of type int
        """
        super(RandomSearch, self).assertions()

        # Check type of n_selection_iters parameter
        if not isinstance(self.n_selection_iters, int):
            raise TypeError(
                "Parameter `n_selection_iters` must be of " "type int"
            )

    def __init__(
        self,
        optimizer,
        n_particles,
        dimensions,
        options,
        objective_func,
        iters,
        n_selection_iters,
        bounds=None,
        velocity_clamp=(0, 1),
    ):
        """Initialize the Search

        Attributes
        ----------
        n_selection_iters: int
            number of iterations of random parameter selection
        """

        # Assign n_selection_iters as attribute
        self.n_selection_iters = n_selection_iters
        # Assign attributes
        super(RandomSearch, self).__init__(
            optimizer,
            n_particles,
            dimensions,
            options,
            objective_func,
            iters,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
        )
        # Invoke assertions
        self.assertions()

    def generate_grid(self):
        """Generate the grid of hyperparameter value combinations"""

        options = dict(self.options)
        params = {}

        # Remove 'p' to hold as a constant in the paramater combinations
        p = options.pop("p")
        params["p"] = [p for _ in xrange(self.n_selection_iters)]

        # Assign generators based on parameter type
        param_generators = {
            "c1": np.random.uniform,
            "c2": np.random.uniform,
            "w": np.random.uniform,
            "k": np.random.randint,
        }

        # Generate random values for hyperparameters 'c1', 'c2', 'w', and 'k'
        for idx, bounds in options.items():
            params[idx] = param_generators[idx](
                *bounds, size=self.n_selection_iters
            )

        # Return list of dicts of hyperparameter combinations
        return [
            {
                "c1": params["c1"][i],
                "c2": params["c2"][i],
                "w": params["w"][i],
                "k": params["k"][i],
                "p": params["p"][i],
            }
            for i in xrange(self.n_selection_iters)
        ]
