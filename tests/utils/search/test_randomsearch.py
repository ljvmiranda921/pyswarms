#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit testing for pyswarms.random_search"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import unittest
import numpy as np

# Import from package
from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.single import LocalBestPSO
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

class Base(unittest.TestCase):

    def setUp(self):
        """Sets up test fixtures"""
        self.optimizer = LocalBestPSO
        self.n_particles = 40
        self.dimensions = 20
        self.options = {'c1': [1, 5],
                        'c2': [6, 10],
                        'k' : [11, 15],
                        'w' : [0.4, 0.9],
                        'p' : 1}
        self.bounds = (np.array([-5,-5]), np.array([5,5]))
        self.iters = 10
        self.n_selection_iters = 100
        self.objective_func = sphere_func
        self.g = RandomSearch(self.optimizer, self.n_particles,
                self.dimensions, self.options, self.objective_func,
                self.iters, self.n_selection_iters, self.bounds,
                velocity_clamp=None)
        self.g_unbound = RandomSearch(self.optimizer, self.n_particles,
                self.dimensions, self.options, self.objective_func,
                self.iters, self.n_selection_iters,bounds=None,
                velocity_clamp=None)

class MethodReturnType(Base):

    def test_search_min_best_options_return_type(self):
        """Tests if best options returns a dictionary"""
        minimum_best_score, minimum_best_options = self.g_unbound.search()
        self.assertIsInstance(minimum_best_options, dict)

    def test_search_max_best_options_return_type(self):
        """Tests if max best options returns a dictionary"""
        maximum_best_score, maximum_best_options = self.g_unbound.search(maximum=True)
        self.assertIsInstance(maximum_best_options, dict)

class MethodReturnValues(Base):

    def test_search_greater_values(self):
        """Tests if max is greater than min in sample use-case"""
        minimum_best_score, minimum_best_options = self.g_unbound.search()
        maximum_best_score, maximum_best_options = self.g_unbound.search(maximum=True)
        self.assertGreater(maximum_best_score, minimum_best_score)

    def test_generate_grid_combinations(self):
        """Test that the number of combinations in grid equals
        the number parameter selection iterations specficied"""
        grid = self.g.generate_grid()
        self.assertEqual(len(grid), self.n_selection_iters)

    def test_generate_grid_param_mapping(self):
        """Test that generated values are correctly mapped to each parameter
        and are within the specified bounds
        """
        grid = self.g.generate_grid()
        for i in ['c1','c2','k','w']:
            values = [x[i] for x in grid]
            for j in values:
                self.assertGreaterEqual(j, self.options[i][0])
                self.assertLessEqual(j, self.options[i][1])

class Instantiation(Base):

    def test_optimizer_type_fail(self):
        """Test that :code:`optimizer` of type :code:`string` raises
        :code:`TypeError`"""
        bad_optimizer = 'LocalBestPSO'  # a string instead of a class object
        with self.assertRaises(TypeError):
            g = RandomSearch(bad_optimizer, self.n_particles, self.dimensions,
                             self.options, self.objective_func, self.iters,
                             self.n_selection_iters, bounds=None,
                             velocity_clamp=None)

    def test_n_selection_iters_type_fail(self):
        """Test that :code:`n_selection_iters` of type :code:`float` raises
        :code:`TypeError`"""
        bad_n_selection_iters = 100.0  # should be an int
        with self.assertRaises(TypeError):
            g = RandomSearch(self.optimizer, self.n_particles, self.dimensions,
                             self.options, self.objective_func, self.iters,
                             bad_n_selection_iters, self.bounds,
                             velocity_clamp=None)
