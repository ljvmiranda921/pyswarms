#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit testing for pyswarms.grid_search"""

# Import modules
import unittest
import numpy as np

from pyswarms.utils.search.grid_search import GridSearch
from pyswarms.single import LocalBestPSO
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

class TestGridSearch(unittest.TestCase):

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
        self.bounds = (np.array([-5,-5]), np.array([5,5]))
        self.iters = 10
        self.objective_func = sphere_func

    def test_optimize(self):
        """Tests if the optimize method returns expected values."""
        g = GridSearch(self.optimizer, self.n_particles, self.dimensions,
                       self.options, self.objective_func, self.iters,
                       bounds=None, velocity_clamp=None)

        minimum_best_score, minimum_best_options = g.search()
        maximum_best_score, maximum_best_options = g.search(maximum=True)

        # Test method returns a dict
        self.assertEqual(type(minimum_best_options), dict)
        self.assertEqual(type(maximum_best_options), dict)

        # The scores could be equal, but for our test case the
        # max score is greater than the min.
        self.assertGreater(maximum_best_score, minimum_best_score)

    def test_generate_grid(self):
        """Tests if generate_grid function returns expected value."""
        options = {'c1': [1,2],
                   'c2': 6,
                   'k': 5,
                   'w': 0.9,
                   'p': 0}
        g = GridSearch(self.optimizer, self.n_particles, self.dimensions,
                       options, self.objective_func, self.iters,
                       bounds=None, velocity_clamp=None)
        self.assertEqual(g.generate_grid(),
                         [{'c1': 1, 'c2': 6, 'k': 5, 'w': 0.9, 'p': 0},
                          {'c1': 2, 'c2': 6, 'k': 5, 'w': 0.9, 'p': 0}])

if __name__ == '__main__':
    unittest.main()
