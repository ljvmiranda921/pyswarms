#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.optimizers import LocalBestPSO
from pyswarms.optimizers.binary import BinaryPSO
from pyswarms.utils.functions.single_obj import sphere
from pyswarms.utils.types import SwarmOptions

from .abc_test_optimizer import ABCTestOptimizer


class TestLocalBestOptimizer(ABCTestOptimizer):
    @pytest.fixture
    def optimizer(self, options: SwarmOptions): # type: ignore
        return LocalBestPSO(10, 2, options, p=2, k=2)

    def test_local_correct_pos(self, optimizer: BinaryPSO):
        """Test to check local optimiser returns the correct position corresponding to the best cost"""
        _, pos = optimizer.optimize(sphere, iters=5)

        # find best pos from history
        min_cost_idx = np.argmin(optimizer.cost_history)
        min_pos_idx = np.argmin(sphere(optimizer.pos_history[min_cost_idx]))
        assert np.array_equal(optimizer.pos_history[min_cost_idx][min_pos_idx], pos)
