# -*- coding: utf-8 -*-
"""
Hyperparameter grid search.

Compares the relative performance of hyperparameter value combinations in
optimizing a specified objective function.

For each hyperparameter, user can provide either a single value or a list
of possible values. A list of topologies to check can also be supplied. The
cartesian products of these hyperparameters are taken to produce a grid of
all possible combinations. These combinations are then tested to produce a
list of objective function scores. The search method default returns the
minimum objective function score and hyperparameters that yield the minimum
score, yet maximum score can also be evaluated.

>>> options_grid = {'c1': [1, 2, 3],
               'c2': [1, 2, 3],
               'w' : [2, 3, 5]}
>>> topologies = (Ring(p=1, k=4),
                  Ring(p=2, k=5),
                  Random(k=6))
The optimizer requires an options argument itself which is a shortcoming of
this architecture, to be solved in the future.
>>> optimizer = LocalBestPSO(
    n_particles=40,
    dimensions=20,
    options={'c1': 1, 'c2': 1, 'w': 1},
)
>>> g = GridSearch(
    optimizer,
    objective_func=sphere,
    iters=10,
    options_grid=options_grid,
    topologies=topologies,
)
>>> best_score, best_options = g.search()
>>> best_score
0.498641604188
>>> best_options['c1']
1
>>> best_options['c2']
1
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy.typing as npt

from pyswarms.backend.topology.base import Topology
from pyswarms.optimizers.base import BaseSwarmOptimizer
from pyswarms.utils.search.base_search import SearchBase
from pyswarms.utils.types import SwarmOption, SwarmOptions

OptionsGrid = Dict[SwarmOption, float | List[float]]


class GridSearch(SearchBase):
    """Exhaustive search of optimal performance on selected objective function
    over all combinations of specified hyperparameter values.
    """

    topologies: Tuple[Topology, ...]

    def __init__(
        self,
        optimizer: BaseSwarmOptimizer,
        objective_func: Callable[..., npt.NDArray[Any]],
        iters: int,
        options_grid: OptionsGrid,
        topologies: Optional[Tuple[Topology, ...]] = None,
    ):
        """Initialize the Search

        Attributes
        ----------
        optimizer : pyswarms.single
            either LocalBestPSO or GlobalBestPSO
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        options_grid : OptionsGrid
            A float or list of floats for each of the options c1, c2, w
        topologies : Tuple[Topology, ...], optional
            A list of topologies to test out
        """
        super(GridSearch, self).__init__(
            optimizer,
            objective_func,
            iters,
        )

        self.options_grid = {k: v if isinstance(v, list) else [v] for k, v in options_grid.items()}
        self.topologies = topologies if topologies is not None else (self.optimizer.topology,)

    def generate_grid(self):
        """Generate the grid of all hyperparameter value combinations"""
        params = list(self.options_grid.keys())
        list_of_products = itertools.product(*self.options_grid.values())
        return (
            (SwarmOptions(dict(zip(params, x))), topology)  # type: ignore
            for x in list_of_products
            for topology in self.topologies
        )
