#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyswarms` package."""

# Import modules
import unittest
import numpy as np

# Import from package
from pyswarms.utils.functions import single_obj as fx


class TestSingleObj(unittest.TestCase):
    """Base class for testing single-objective functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Test swarm with size=3, dimensions=2
        self.input = np.zeros([3,2])
        # Target value
        self.target = np.zeros(3)

class ExpectedMinima(TestSingleObj):
    """Test if a function outputs a minima if fed with expected argmin."""

    def test_sphere(self):
        """Tests sphere function."""
        self.assertEqual(fx.sphere_func(self.input).all(), self.target.all())

    def test_rastrigin(self):
        """Tests rastrigin function."""
        self.assertEqual(fx.rastrigin_func(self.input).all(), self.target.all())

    def test_ackley(self):
        """Tests ackley function."""
        assert np.isclose(fx.ackley_func(self.input), self.target).all()

    def test_rosenbrock(self):
        """Tests rosenbrock function."""
        self.assertEqual(fx.rosenbrock_func(np.ones([3,2])).all(), np.array([0,0,0]).all())




if __name__ == '__main__':
    unittest.main()