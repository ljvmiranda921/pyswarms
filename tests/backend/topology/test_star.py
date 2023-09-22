#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
from typing import Any, Dict, Type
import numpy as np
import pytest
from pyswarms.backend.swarms import Swarm

# Import from pyswarms
from pyswarms.backend.topology import Star

from .abc_test_topology import ABCTestTopology

np.random.seed(4135157)


class TestStarTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Star

    @pytest.fixture
    def options(self) -> Dict[str, Any]:
        return {}

    def test_compute_gbest_return_values(self, swarm: Swarm, options: Dict[str, Any], topology: Type[Star]):
        """Test if compute_gbest() gives the expected return values"""
        topo = topology()
        expected_cost = 1.0002528364353296
        expected_pos = np.array([9.90438476e-01, 2.50379538e-03, 1.87405987e-05])
        pos, cost = topo.compute_gbest(swarm, **options)
        assert cost == pytest.approx(expected_cost) # type: ignore
        assert pos == pytest.approx(expected_pos) # type: ignore
