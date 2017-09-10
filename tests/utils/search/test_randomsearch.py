#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit testing for pyswarms.random_search"""

# Import modules
import unittest
import numpy as np

from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.single import LocalBestPSO
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

class TestRandomSearch(unittest.TestCase):

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

    def test_generate_grid(self):
        """Tests if generate_grid function returns expected values."""
        g = RandomSearch(self.optimizer, self.n_particles, self.dimensions,
                       self.options, self.objective_func, self.iters,
                       self.n_selection_iters,
                       self.bounds, velocity_clamp=None)

        #Test that the number of combinations in grid equals
        #the number parameter selection iterations specficied
        grid = g.generate_grid()
        self.assertEqual(len(grid), self.n_selection_iters)

        #Test that generated values are correctly mapped to each parameter
        #and are within the specified bounds
        for i in ['c1','c2','k','w']:
            values = [x[i] for x in grid]
            for j in values:
                self.assertGreaterEqual(j, self.options[i][0])
                self.assertLessEqual(j, self.options[i][1])

    def test_search(self):
        """Tests if the search method returns expected values."""
        g = RandomSearch(self.optimizer, self.n_particles, self.dimensions,
                         self.options, self.objective_func, self.iters,
                         self.n_selection_iters,
                         bounds=None, velocity_clamp=None)

        minimum_best_score, minimum_best_options = g.search()
        maximum_best_score, maximum_best_options = g.search(maximum=True)

        # Test method returns a dict
        self.assertEqual(type(minimum_best_options), dict)
        self.assertEqual(type(maximum_best_options), dict)

        # The scores could be equal, but for our test case the
        # max score is greater than the min.
        self.assertGreater(maximum_best_score, minimum_best_score)

if __name__ == '__main__':
    unittest.main()
