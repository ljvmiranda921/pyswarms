#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Literal, Type

import numpy as np
import pytest

from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology import VonNeumann

from .abc_test_topology import ABCTestTopology

np.random.seed(4135157)


class TestVonNeumannTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return VonNeumann

    @pytest.fixture
    def options(self) -> Dict[str, Any]:
        return {"p": 1, "r": 1, "dimensions": 3}

    @pytest.mark.parametrize("r", [0, 1])
    @pytest.mark.parametrize("p", [1, 2])
    def test_update_gbest_neighborhood(self, swarm: Swarm, topology: Type[VonNeumann], p: Literal[1, 2], r: int):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology(swarm.dimensions, p=p, r=r)
        pos, cost = topo.compute_gbest(swarm)
        expected_pos = np.array([9.90438476e-01, 2.50379538e-03, 1.87405987e-05])
        expected_pos_2 = np.array([9.98033031e-01, 4.97392619e-03, 3.07726256e-03])
        expected_cost = 1.0002528364353296
        assert cost == pytest.approx(expected_cost)  # type: ignore
        assert (pos[np.argmin(cost)] == pytest.approx(expected_pos)) or (  # type: ignore
            pos[np.argmin(cost)] == pytest.approx(expected_pos_2)  # type: ignore
        )

    @pytest.mark.parametrize("m", [i for i in range(3)])
    @pytest.mark.parametrize("n", [i for i in range(3)])
    def test_delannoy_numbers(self, m: int, n: int):
        expected_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])
        assert VonNeumann.delannoy(m, n) in expected_values
