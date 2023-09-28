# -*- coding: utf-8 -*-
"""Base class for hyperparameter optimization search functions"""

import operator as op
from abc import ABC, abstractmethod
from typing import Any, Callable, List

import numpy.typing as npt
from pyswarms.optimizers.base import BaseSwarmOptimizer

from pyswarms.utils.types import SwarmOptions


class SearchBase(ABC):
    def __init__(
        self,
        optimizer: BaseSwarmOptimizer,
        objective_func: Callable[..., npt.NDArray[Any]],
        iters: int,
    ):
        """Initialize the Search

        Attributes
        ----------
        optimizer: pyswarms.single
            either LocalBestPSO or GlobalBestPSO
        objective_func: function
            objective function to be evaluated
        iters: int
            number of iterations
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
            a dict with the following keys: {'c1', 'c2', 'w', 'k', 'p'}
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
        scores = [self.generate_score(i) for i in grid]

        # Default behavior
        idx, self.best_score = min(enumerate(scores), key=op.itemgetter(1))

        # Catches the maximum bool flag
        if maximum:
            idx, self.best_score = max(enumerate(scores), key=op.itemgetter(1))

        # Return optimum hyperparameter value property from grid using index
        self.best_options = op.itemgetter(idx)(grid)
        return self.best_score, self.best_options

    @abstractmethod
    def generate_grid(self) -> List[SwarmOptions]:
        ...
