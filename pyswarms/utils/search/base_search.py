# -*- coding: utf-8 -*-
"""Base class for hyperparameter optimization search functions"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import numpy as np
import numpy.typing as npt

from pyswarms.optimizers.base import BaseSwarmOptimizer
from pyswarms.utils.types import SwarmOptions


class SearchBase(ABC):
    best_score: float
    best_options: SwarmOptions

    def __init__(
        self,
        optimizer: BaseSwarmOptimizer,
        objective_func: Callable[..., npt.NDArray[Any]],
        iters: int,
    ):
        """Initialize the Search

        Attributes
        ----------
        optimizer: BaseSwarmOptimizer
            optimizer instance to tune
        objective_func: function
            objective function to be evaluated
        iters: int
            number of iterations
        topologies: Tuple[Topology], optional
            Additional topologies to evaluate
        """
        # Assign attributes
        self.optimizer = optimizer
        self.objective_func = objective_func
        self.iters = iters

    def generate_score(self, options: SwarmOptions):
        """Generate score for optimizer's performance on objective function

        Parameters
        ----------

        options: dict
            a dict with the following keys: {'c1', 'c2', 'w'}
        """
        # Reset the optimizer and update the options
        self.optimizer.velocity_updater.init_options(options)
        self.optimizer.reset()

        # Return score
        return self.optimizer.optimize(self.objective_func, self.iters)[0]

    def search(self, maximum: bool = False):
        """Compare optimizer's objective function performance scores
        for all combinations of provided parameters

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
        self.best_score = np.inf
        for options in grid:
            score = self.generate_score(options) * (-1 if maximum else 1)
            if score < self.best_score:
                self.best_options = options
                self.best_score = score

        return self.best_score * (-1 if maximum else 1), self.best_options

    @abstractmethod
    def generate_grid(self) -> Iterable[SwarmOptions]:
        ...
