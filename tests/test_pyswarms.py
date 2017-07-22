#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyswarms` package."""

# Import modules
import unittest
import numpy as np

# Import from package
from pyswarms.utils.functions import single_obj as fx

class TestSingleObj(unittest.TestCase):
    """Tests all single-objective functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Test swarm with size=3, dimensions=2
        self.input = np.zeros([3,2])
        # Target value
        self.target = np.zeros(3)

    def test_sphere(self):
        """Test sphere function."""
        assert fx.sphere_func(self.input).all() == self.target.all()

    def test_rastrigin(self):
        """Test rastrigin function."""
        assert fx.rastrigin_func(self.input).all() == self.target.all()

    def test_ackley(self):
        """Test ackley function."""
        assert np.isclose(fx.ackley_func(self.input), self.target).all()

    def test_rosenbrock(self):
        """Test rosenbrock function."""
        assert fx.rosenbrock_func(np.ones([3,2])).all() == np.array([0,0,0]).all()

if __name__ == '__main__':
    unittest.main()