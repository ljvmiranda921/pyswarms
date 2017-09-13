#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyswarms` package."""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import unittest
import numpy as np

# Import from package
from pyswarms.utils.functions import single_obj as fx

class Base(unittest.TestCase):
    """Base class for testing single-objective functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Test swarm with size=3, dimensions=2
        self.input = np.zeros([3,2])
        self.input2 = np.ones([3,2])
        # Erroneous input - dimension
        self.bad_input = np.zeros([3,3])
        # Target value
        self.target = np.zeros(3)
        # Target output size
        self.target_size = (self.input.shape[0], )

class InstantiationBound(Base):
    """Tests exception throws when fed with erroneous input."""

    def test_rastrigin_bound_fail(self):
        """Test rastrigin bound exception"""
        x = - np.random.uniform(low=6,high=100,size=(3,2))
        x_ = np.random.uniform(low=6,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.rastrigin_func(x)

        with self.assertRaises(ValueError):
            fx.rastrigin_func(x_)

    def test_ackley_bound_fail(self):
        """Test ackley bound exception"""
        x = - np.random.uniform(low=32,high=100,size=(3,2))
        x_ = np.random.uniform(low=32,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.ackley_func(x)
        with self.assertRaises(ValueError):
            fx.ackley_func(x_)

    def test_beale_bound_fail(self):
        """Test beale bound exception"""
        x = - np.random.uniform(low=4.6666,high=100,size=(3,2))
        x_ = np.random.uniform(low=4.6666,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.beale_func(x)
        with self.assertRaises(ValueError):
            fx.beale_func(x_)

    def test_goldstein_bound_fail(self):
        """Test goldstein bound exception"""
        x = - np.random.uniform(low=2.00001,high=100,size=(3,2))
        x_ = np.random.uniform(low=2.00001,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.goldstein_func(x)
        with self.assertRaises(ValueError):
            fx.goldstein_func(x_)

    def test_booth_bound_fail(self):
        """Test booth bound exception"""
        x = - np.random.uniform(low=11.00001,high=100,size=(3,2))
        x_ = np.random.uniform(low=11.00001,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.booth_func(x)
        with self.assertRaises(ValueError):
            fx.booth_func(x_)

    def test_bukin6_bound_fail(self):
        """Test bukin6 bound exception"""
        x = - np.random.uniform(low=15.001,high=100,size=(3,2))
        x_ =  np.random.uniform(low=-5.001,high=-3.001,size=(3,2))
        x_1 =  np.random.uniform(low=-3.001,high=-100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.bukin6_func(x)
        with self.assertRaises(ValueError):
            fx.bukin6_func(x_)
        with self.assertRaises(ValueError):
            fx.bukin6_func(x_1)

    def test_matyas_bound_fail(self):
        """Test matyas bound exception"""
        x = - np.random.uniform(low=10.001,high=100,size=(3,2))
        x_ = np.random.uniform(low=10.001,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.matyas_func(x)
        with self.assertRaises(ValueError):
            fx.matyas_func(x_)

    def test_levi_bound_fail(self):
        """Test levi bound exception"""
        x = - np.random.uniform(low=10.001,high=100,size=(3,2))
        x_ = np.random.uniform(low=10.001,high=100,size=(3,2))

        with self.assertRaises(ValueError):
            fx.levi_func(x)
        with self.assertRaises(ValueError):
            fx.levi_func(x_)

    def test_schaffer2_bound_fail(self):
        """Test schaffer2 bound exception"""
        x = - np.random.uniform(low=100.001,high=1000,size=(3,2))
        x_ = np.random.uniform(low=100.001,high=1000,size=(3,2))

        with self.assertRaises(ValueError):
            fx.schaffer2_func(x)
        with self.assertRaises(ValueError):
            fx.schaffer2_func(x_)

class InstantiationDimension(Base):
    """Tests exception throws when fed with erroneous dimension."""

    def test_beale_dim_fail(self):
        """Test beale dim exception"""
        with self.assertRaises(IndexError):
            fx.beale_func(self.bad_input)

    def test_goldstein_dim_fail(self):
        """Test golstein dim exception"""
        with self.assertRaises(IndexError):
            fx.goldstein_func(self.bad_input)

    def test_booth_dim_fail(self):
        """Test booth dim exception"""
        with self.assertRaises(IndexError):
            fx.booth_func(self.bad_input)

    def test_bukin6_dim_fail(self):
        """Test bukin6 dim exception"""
        with self.assertRaises(IndexError):
            fx.bukin6_func(self.bad_input)

    def test_matyas_dim_fail(self):
        """Test matyas dim exception"""
        with self.assertRaises(IndexError):
            fx.matyas_func(self.bad_input)

    def test_levi_dim_fail(self):
        """Test levi dim exception"""
        with self.assertRaises(IndexError):
            fx.levi_func(self.bad_input)

    def test_schaffer2_dim_fail(self):
        """Test schaffer2 dim exception"""
        with self.assertRaises(IndexError):
            fx.schaffer2_func(self.bad_input)

class MethodsReturnValue(Base):
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
        self.assertEqual(fx.rosenbrock_func(self.input2).all(),np.zeros(3).all())

    def test_beale_output(self):
        """Tests beale function output."""
        assert np.isclose(fx.beale_func([3, 0.5] * self.input2),
            self.target).all()

    def test_goldstein_output(self):
        """Tests goldstein-price function output."""
        assert np.isclose(fx.goldstein_func([0, -1] * self.input2),
            (3 * np.ones(3))).all()

    def test_booth_output(self):
        """Test booth function output."""
        assert np.isclose(fx.booth_func([1, 3] * self.input2),
            self.target).all()

    def test_bukin6_output(self):
        """Test bukin function output."""
        assert np.isclose(fx.bukin6_func([-10, 1] * self.input2),
            self.target).all()

    def test_bukin6_output(self):
        """Test bukin function output."""
        assert np.isclose(fx.matyas_func(self.input),self.target).all()

    def test_levi_output(self):
        """Test levi function output."""
        assert np.isclose(fx.levi_func(self.input2), self.target).all()

    def test_schaffer2_output(self):
        """Test schaffer2 function output."""
        assert np.isclose(fx.schaffer2_func(self.input), self.target).all()

class MethodsReturnSize(Base):
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

    def test_beale_output_size(self):
        """Tests beale output size."""
        self.assertEqual(fx.beale_func(self.input).shape, self.target_size)

    def test_goldstein_output_size(self):
        """Test goldstein output size."""
        self.assertEqual(fx.goldstein_func(self.input).shape, self.target_size)

    def test_booth_output_size(self):
        """Test booth output size."""
        self.assertEqual(fx.booth_func(self.input).shape, self.target_size)

    def test_bukin6_output_size(self):
        """Test bukin6 output size."""
        self.assertEqual(fx.bukin6_func([-10,0] * self.input2).shape, self.target_size)

    def test_levi_output_size(self):
        """Test levi output size."""
        self.assertEqual(fx.levi_func(self.input).shape, self.target_size)

    def test_schaffer2_output_size(self):
        """Test schaffer2 output size."""
        self.assertEqual(fx.schaffer2_func(self.input).shape, self.target_size)

if __name__ == '__main__':
    unittest.main()