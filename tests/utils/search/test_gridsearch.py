#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit testing for pyswarms.grid_search"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import unittest
import numpy as np

# Import from package
from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single import LocalBestPSO
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

class Base(unittest.TestCase):

    def setUp(self):
        """Sets up test fixtures"""
        self.optimizer = LocalBestPSO
        self.n_particles = 40
        self.dimensions = 20
        self.options = {'c1': [1, 2, 3],
                        'c2': [1, 2, 3],
                        'k' : [5, 10, 15],
                        'w' : [0.9, 0.7, 0.4],
                        'p' : [1]}
        self.mini_options = {'c1': [1,2],
                   'c2': 6,
                   'k': 5,
                   'w': 0.9,
                   'p': 0}
        self.bounds = (np.array([-5,-5]), np.array([5,5]))
        self.iters = 10
        self.objective_func = sphere_func
        self.g = GridSearch(self.optimizer, self.n_particles, self.dimensions,
                       self.options, self.objective_func, self.iters,
                       bounds=None, velocity_clamp=None)
        self.g_mini = GridSearch(self.optimizer, self.n_particles, self.dimensions,
                       self.mini_options, self.objective_func, self.iters,
                       bounds=None, velocity_clamp=None)

class MethodReturnType(Base):

    def test_search_min_best_options_return_type(self):
        """Tests if best options returns a dictionary"""
        minimum_best_score, minimum_best_options = self.g.search()
        self.assertIsInstance(minimum_best_options, dict)

    def test_search_max_best_options_return_type(self):
        """Tests if max best options returns a dictionary"""
        maximum_best_score, maximum_best_options = self.g.search(maximum=True)
        self.assertIsInstance(maximum_best_options, dict)

class MethodReturnValues(Base):

    def test_search_greater_values(self):
        """Tests if max is greater than min in sample use-case"""
        minimum_best_score, minimum_best_options = self.g.search()
        maximum_best_score, maximum_best_options = self.g.search(maximum=True)
        self.assertGreater(maximum_best_score, minimum_best_score)

    def test_generate_grid(self):
        """Tests if generate_grid function returns expected value."""
        self.assertEqual(self.g_mini.generate_grid(),
                         [{'c1': 1, 'c2': 6, 'k': 5, 'w': 0.9, 'p': 0},
                          {'c1': 2, 'c2': 6, 'k': 5, 'w': 0.9, 'p': 0}])

class Instantiation(Base):

    def test_optimizer_type_fail(self):
        """Tests that :code:`optimizer` of type :code:`string` raises
        :code:`TypeError`"""
        bad_optimizer = 'LocalBestPSO'  # a string instead of a class object
        with self.assertRaises(TypeError):
            g = GridSearch(bad_optimizer, self.n_particles, self.dimensions,
                           self.options, self.objective_func, self.iters,
                           bounds=None, velocity_clamp=None)
