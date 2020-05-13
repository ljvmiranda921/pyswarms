#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
import abc

# Import modules
import pytest


class ABCTestTopology(abc.ABC):
    """Abstract class that defines various tests for topologies

    Whenever a topology inherits from ABCTestTopology,
    you don't need to write down all tests anymore. Instead, you can just
    specify all required fixtures in the test suite.
    """

    @pytest.fixture
    def topology(self):
        """Return an instance of the topology"""
        raise NotImplementedError("NotImplementedError::topology")

    @pytest.fixture
    def options(self):
        """Return a dictionary of options"""
        raise NotImplementedError("NotImplementedError::options")

    @pytest.mark.parametrize("static", [True, False])
    def test_neighbor_idx(self, topology, options, swarm, static):
        """Test if the neighbor_idx attribute is assigned"""
        topo = topology(static=static)
        topo.compute_gbest(swarm, **options)
        assert topo.neighbor_idx is not None

    @pytest.mark.parametrize("static", [True, False])
    @pytest.mark.parametrize("swarm", [0, (1, 2, 3)])
    def test_input_swarm(self, topology, static, swarm, options):
        """Test if AttributeError is raised when passed with a non-Swarm instance"""
        with pytest.raises(AttributeError):
            topo = topology(static=static)
            topo.compute_gbest(swarm, **options)
