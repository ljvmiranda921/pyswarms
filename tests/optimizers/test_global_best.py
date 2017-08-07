#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit testing for pyswarms.single.GlobalBestPSO"""

# Import modules
import unittest
import numpy as np

# Import from package
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

class Base(unittest.TestCase):
    """Base class for tests

    This class defines a common `setUp` method that defines attributes
    which are used in the various tests.
    """
    def setUp(self):
        """Set up test fixtures"""
        self.options = {'c1':0.5, 'c2':0.7, 'w':0.5}
        self.safe_bounds = (np.array([-5,-5]), np.array([5,5]))

class Instantiation(Base):
    """Tests all aspects of instantiation

    Tests include: instantiation with args of wrong type, instantiation
    with input values outside constraints, etc.
    """
    def test_keyword_check_fail(self):
        """Tests if exceptions are thrown when keywords are missing"""
        check_c1 = {'c2':0.7, 'w':0.5}
        check_c2 = {'c1':0.5, 'w':0.5}
        check_m = {'c1':0.5, 'c2':0.7}
        with self.assertRaises(KeyError):
            optimizer = GlobalBestPSO(5,2,options=check_c1)
        with self.assertRaises(KeyError):
            optimizer = GlobalBestPSO(5,2,options=check_c2)
        with self.assertRaises(KeyError):
            optimizer = GlobalBestPSO(5,2,options=check_m)

    def test_bound_size_fail(self):
        """Tests if exception is thrown when bound length is not 2"""
        bounds = tuple(np.array([-5,-5]))
        with self.assertRaises(IndexError):
            optimizer = GlobalBestPSO(5,2,options=self.options,bounds=bounds)

    def test_bound_type_fail(self):
        """Tests if exception is thrown when bound type is not tuple"""
        bounds = [np.array([-5,-5]), np.array([5,5])]
        with self.assertRaises(TypeError):
            optimizer = GlobalBestPSO(5,2,bounds=bounds,options=self.options)

    def test_bound_maxmin_fail(self):
        """Tests if exception is thrown when min max of the bound is
        wrong."""
        bounds_1 = (np.array([5,5]), np.array([-5,-5]))
        bounds_2 = (np.array([5,-5]), np.array([-5,5]))
        with self.assertRaises(ValueError):
            optimizer = GlobalBestPSO(5,2,bounds=bounds_1,options=self.options)
        with self.assertRaises(ValueError):
            optimizer = GlobalBestPSO(5,2,bounds=bounds_2,options=self.options)

    def test_bound_shapes_fail(self):
        """Tests if exception is thrown when bounds are of unequal 
        shapes."""
        bounds = (np.array([-5,-5,-5]), np.array([5,5]))
        with self.assertRaises(IndexError):
            optimizer = GlobalBestPSO(5,2,bounds=bounds,options=self.options)

    def test_bound_shape_dims_fail(self):
        """Tests if exception is thrown when bound shape is not equal
        to dimensions."""
        bounds = (np.array([-5,-5,-5]), np.array([5,5,5]))
        with self.assertRaises(IndexError):
            optimizer = GlobalBestPSO(5,2,bounds=bounds,options=self.options)

    def test_vclamp_type_fail(self):
        """Tests if exception is thrown when velocity_clamp is not a tuple."""
        velocity_clamp = [1,3]
        with self.assertRaises(TypeError):
            optimizer = GlobalBestPSO(5,2,velocity_clamp=velocity_clamp,options=self.options)

    def test_vclamp_shape_fail(self):
        """Tests if exception is thrown when velocity_clamp is not equal to 2"""
        velocity_clamp = (1,1,1)
        with self.assertRaises(IndexError):
            optimizer = GlobalBestPSO(5,2,velocity_clamp=velocity_clamp,options=self.options)

    def test_vclamp_minmax_fail(self):
        """Tests if exception is thrown when velocity_clamp's minmax is wrong"""
        velocity_clamp = (3,2)
        with self.assertRaises(ValueError):
            optimizer = GlobalBestPSO(5,2,velocity_clamp=velocity_clamp,options=self.options)

class Methods(Base):
    """Tests all aspects of the class methods

    Tests include: wrong inputs of methods, wrong return types,
    unexpected attribute setting, and etc.
    """

    def test_reset(self):
        """Tests if the reset method resets the attributes required"""
        # Perform a simple optimization
        optimizer = GlobalBestPSO(5,2,options=self.options)
        optimizer.optimize(sphere_func, 100, verbose=0)
        # Reset the attributes
        optimizer.reset()
        # Perform testing
        self.assertEqual(optimizer.best_cost, np.inf)
        self.assertIsNone(optimizer.best_pos)

class Run(Base):
    """Perform a single run of the algorithm to see if something breaks."""

    def test_run(self):
        """Perform a single run."""
        optimizer = GlobalBestPSO(10,2,options=self.options)
        try:
            optimizer.optimize(sphere_func, 1000, verbose=0)
            trigger = True
        except:
            print('Execution failed.')
            trigger = False

        self.assertTrue(trigger)

if __name__ == '__main__':
    unittest.main()