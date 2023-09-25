#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
from typing import TYPE_CHECKING, Any, Dict, Type

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology import Random

from .abc_test_topology import ABCTestTopology

np.random.seed(4135157)


if TYPE_CHECKING:

    class FixtureRequest:
        param: int

else:
    FixtureRequest = Any


class TestRandomTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return Random

    @pytest.fixture(params=[1, 2, 3])
    def options(self, request: FixtureRequest) -> Dict[str, Any]:  # type: ignore
        return {"k": request.param}

    @pytest.mark.parametrize("static", [True, False])
    def test_compute_gbest_return_values(
        self, swarm: Swarm, options: Dict[str, Any], topology: Type[Random], static: bool
    ):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology(static=static, **options)
        expected_cost = 1.0002528364353296
        expected_pos = np.array([9.90438476e-01, 2.50379538e-03, 1.87405987e-05])
        expected_pos_2 = np.array([9.98033031e-01, 4.97392619e-03, 3.07726256e-03])
        pos, cost = topo.compute_gbest(swarm, **options)
        assert cost == pytest.approx(expected_cost)  # type: ignore
        assert (pos[np.argmin(cost)] == pytest.approx(expected_pos)) or (  # type: ignore
            pos[np.argmin(cost)] == pytest.approx(expected_pos_2)  # type: ignore
        )

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [1, 2])
    def test_compute_neighbors_return_values(self, swarm: Swarm, topology: Type[Random], k: int, static: bool):
        """Test if __compute_neighbors() gives the expected shape and symmetry"""
        topo = topology(static=static, k=k)
        adj_matrix = topo._compute_neighbors(swarm)
        assert adj_matrix.shape == (swarm.n_particles, swarm.n_particles)
        assert np.allclose(adj_matrix, adj_matrix.T, atol=1e-8)  # Symmetry test

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("k", [1])
    def test_compute_neighbors_adjacency_matrix(self, swarm: Swarm, topology: Type[Random], k: int, static: bool):
        """Test if __compute_neighbors() gives the expected matrix"""
        np.random.seed(1)
        topo = topology(static=static, k=k)
        adj_matrix = topo._compute_neighbors(swarm)
        comparison_matrix = np.array(
            [
                [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            ]
        )
        assert np.allclose(adj_matrix, comparison_matrix, atol=1e-8)
