#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.backend.topology import Ring

from .abc_test_topology import ABCTestTopology


class TestRingTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Ring

    @pytest.fixture(params=[(1, 2), (2, 3)])
    def options(self, request):
        p, k = request.param
        return {"p": p, "k": k}

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [i for i in range(1, 10)])
    @pytest.mark.parametrize("p", [1, 2])
    def test_compute_gbest_return_values(self, swarm, topology, p, k, static):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology(static=static)
        pos, cost = topo.compute_gbest(swarm, p=p, k=k)
        expected_pos = np.array(
            [9.90438476e-01, 2.50379538e-03, 1.87405987e-05]
        )
        expected_cost = 1.0002528364353296
        assert cost == pytest.approx(expected_cost)
        assert pos == pytest.approx(expected_pos)
