#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from typing import Any, Dict, Optional, Type

import pytest
from pyswarms.backend.position import PositionUpdater

from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.utils.types import Bounds


class ABCTestTopology(abc.ABC):
    """Abstract class that defines various tests for topologies

    Whenever a topology inherits from ABCTestTopology,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.fixture
    @abc.abstractmethod
    def topology(self) -> Type[Topology]:
        """Return an instance of the topology"""
        raise NotImplementedError("NotImplementedError::topology")

    @pytest.fixture
    @abc.abstractmethod
    def options(self) -> Dict[str, Any]:
        """Return a dictionary of options"""
        raise NotImplementedError("NotImplementedError::options")

    # @pytest.mark.parametrize("static", [True, False])
    # @pytest.mark.parametrize("clamp", [None, (0, 1), (-1, 1)])
    # def test_compute_velocity_return_values(
    #     self, topology: Type[Topology], options: Dict[str, Any], swarm: Swarm, clamp: Optional[Clamp], static: bool
    # ):
    #     """Test if compute_velocity() gives the expected shape and range"""
    #     topo = topology(static=static, **options)
    #     v = topo.compute_velocity(swarm, clamp)
    #     assert v.shape == swarm.position.shape
    #     if clamp is not None:
    #         assert (clamp[0] <= v).all() and (clamp[1] >= v).all()

    @pytest.mark.parametrize(
        "bounds",
        [None, ([-5, -5, -5], [5, 5, 5]), ([-10, -10, -10], [10, 10, 10])],
    )
    def test_compute_position_return_values(self, swarm: Swarm, bounds: Optional[Bounds]):
        """Test if compute_position() gives the expected shape and range"""
        position_updater = PositionUpdater(bounds)
        p = position_updater.compute(swarm)
        assert p.shape == swarm.velocity.shape
        if bounds is not None:
            assert (bounds[0] <= p).all() and (bounds[1] >= p).all()

    @pytest.mark.parametrize("static", [True, False])
    def test_neighbor_idx(self, topology: Type[Topology], options: Dict[str, Any], swarm: Swarm, static: bool):
        """Test if the neighbor_idx attribute is assigned"""
        topo = topology(static=static, **options)
        topo.compute_gbest(swarm)
        assert topo.neighbor_idx is not None  # type: ignore

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    def test_input_swarm(self, topology: Type[Topology], options: Dict[str, Any], swarm: Swarm, static: bool):
        """Test if AttributeError is raised when passed with a non-Swarm instance"""
        with pytest.raises(AttributeError):
            topo = topology(static=static, **options)
            topo.compute_gbest(swarm)
