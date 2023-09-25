#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple, Type

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology import Ring

from .abc_test_topology import ABCTestTopology

np.random.seed(4135157)


if TYPE_CHECKING:

    class FixtureRequest:
        param: Tuple[int, int]

else:
    FixtureRequest = Any


class TestRingTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Ring

    @pytest.fixture(params=[(1, 2), (2, 3)])
    def options(self, request: FixtureRequest) -> Dict[str, Any]:  # type: ignore
        p, k = request.param
        return {"p": p, "k": k}

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [i for i in range(1, 10)])
    @pytest.mark.parametrize("p", [1, 2])
    def test_compute_gbest_return_values(
        self, swarm: Swarm, topology: Type[Ring], p: Literal[1, 2], k: int, static: bool
    ):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology(static=static, p=p, k=k)
        pos, cost = topo.compute_gbest(swarm)
        expected_cost = 1.0002528364353296
        expected_pos = np.array([9.90438476e-01, 2.50379538e-03, 1.87405987e-05])
        expected_pos_2 = np.array([9.98033031e-01, 4.97392619e-03, 3.07726256e-03])
        assert cost == pytest.approx(expected_cost)  # type: ignore
        assert (pos[np.argmin(cost)] == pytest.approx(expected_pos)) or (  # type: ignore
            pos[np.argmin(cost)] == pytest.approx(expected_pos_2)  # type: ignore
        )
