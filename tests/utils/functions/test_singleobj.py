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
        # Target output size
        self.target_size = (self.input.shape[0], )

class ExpectedOutput(TestSingleObj):
    """Tests if a function outputs a minima if fed with expected argmin."""

    def test_sphere_output(self):
        """Tests sphere function output."""
        self.assertEqual(fx.sphere_func(self.input).all(), self.target.all())

    def test_rastrigin_output(self):
        """Tests rastrigin function output."""
        self.assertEqual(fx.rastrigin_func(self.input).all(), self.target.all())

    def test_ackley_output(self):
        """Tests ackley function output."""
        assert np.isclose(fx.ackley_func(self.input), self.target).all()

    def test_rosenbrock_output(self):
        """Tests rosenbrock function output."""
        self.assertEqual(fx.rosenbrock_func(np.ones([3,2])).all(),np.zeros(3).all())

class OutputSize(TestSingleObj):
    """Tests if the output of the function is the same as no. of particles"""

    def test_sphere_output_size(self):
        """Tests sphere output size."""
        self.assertEqual(fx.sphere_func(self.input).shape, self.target_size)

    def test_rastrigin_output_size(self):
        """Tests rastrigin output size."""
        self.assertEqual(fx.rastrigin_func(self.input).shape, self.target_size)

    def test_ackley_output_size(self):
        """Tests ackley output size."""
        self.assertEqual(fx.ackley_func(self.input).shape, self.target_size)

    def test_rosenbrock_output_size(self):
        """Tests rosenbrock output size."""
        self.assertEqual(fx.rosenbrock_func(self.input).shape, self.target_size)

class BoundFail(TestSingleObj):
    """Tests exception throws when fed with erroneous input."""

    def test_rastrigin_bound_fail(self):
        """Test rastrigin bound exception"""
        x = - np.random.randint(low=6,high=100,size=(3,2))
        x_ = np.random.randint(low=6,high=100,size=(3,2))
        with self.assertRaises(AssertionError):
            fx.rastrigin_func(x)
            fx.rastrigin_func(x_)

    def test_ackley_bound_fail(self):
        """Test ackley bound exception"""
        x = - np.random.randint(low=32,high=100,size=(3,2))
        x_ = np.random.randint(low=32,high=100,size=(3,2))
        with self.assertRaises(AssertionError):
            fx.ackley_func(x)
            fx.ackley_func(x_)

if __name__ == '__main__':
    unittest.main()