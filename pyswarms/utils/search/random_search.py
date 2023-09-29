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

from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
from pyswarms.backend.topology.base import Topology

from pyswarms.optimizers.base import BaseSwarmOptimizer
from pyswarms.utils.search.base_search import SearchBase
from pyswarms.utils.types import SwarmOption, SwarmOptions

OptionRanges = Dict[SwarmOption, float | Tuple[float, float]]

T = TypeVar("T")

class RandomSearch(SearchBase):
    """Search of optimal performance on selected objective function
    over combinations of randomly selected hyperparameter values
    within specified bounds for specified number of selection iterations.
    """
    topologies: Tuple[Topology, ...]

    def __init__(
        self,
        optimizer: BaseSwarmOptimizer,
        objective_func: Callable[..., npt.NDArray[Any]],
        iters: int,
        n_selection_iters: int,
        option_ranges: OptionRanges,
        topologies: Optional[Tuple[Topology, ...]] = None
    ):
        """Initialize the Search

        Attributes
        ----------
        n_selection_iters: int
            number of iterations of random parameter selection
        """
        super().__init__(
            optimizer,
            objective_func,
            iters,
        )

        self.n_selection_iters = n_selection_iters
        self.option_ranges = {k: v if isinstance(v, tuple) else (v, v) for k, v in option_ranges.items()}
        self.topologies = topologies if topologies is not None else (self.optimizer.topology,)

    def generate_grid(self):
        """Generate the grid of hyperparameter value combinations"""

        return [
            (
                SwarmOptions(
                    {
                        "c1": np.random.uniform(*self.option_ranges["c1"]),
                        "c2": np.random.uniform(*self.option_ranges["c2"]),
                        "w": np.random.uniform(*self.option_ranges["w"]),
                    }
                ), self._pick_item(self.topologies)
            )
            for _ in range(self.n_selection_iters)
        ]

    # Wrap randint because of typing
    def _pick_item(self, items: Tuple[T,...]) -> T:
        i: int = np.random.randint(len(items))  # type: ignore
        return items[i]
