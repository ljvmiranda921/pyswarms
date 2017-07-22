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
        # Test swarm
        self.x = np.array([[0,0],[0,0],[0,0]])

    def test_sphere(self):
        """Test sphere function."""
        assert fx.sphere_func(self.x).all() == np.array([0,0,0]).all()

if __name__ == '__main__':
    unittest.main()