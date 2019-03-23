

#######################################
## TEST: BinaryPSO
"""
print("Testing BinaryPSO...\n")

def f_per_particle(m):
    return sum(m)

def f(x):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i]) for i in range(n_particles)]
    return np.array(j)

options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}
optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=15, options=options)
cost, pos = optimizer.optimize(f, iters=5)

assert f_per_particle(pos) == cost

######################
## TEST: GlobalBestPOS
print("\nTesting GlobalBestPSO...\n")
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
cost, pos = optimizer.optimize(fx.sphere, iters=5)

assert sum(pos ** 2) == cost

#####################
## TEST: LocalBestPOS
print("\nTesting LocalBestPSO...\n")
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)
cost, pos = optimizer.optimize(fx.sphere, iters=5)

assert sum(pos ** 2) == cost


#######################################
## TEST: GeneralBestPOS
# No test implemented
"""

import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.backend.topology import Pyramid
import unittest

class TestOptimizer(unittest.TestCase):
    def test_binary(self):
        dim = 10
        x = np.random.rand(dim,dim)
        def f_per_particle(m):
            # Get the subset of the features from the binary mask
            if np.count_nonzero(m) == 0:
                return sum(x)
            return sum(x[:, m==1]).mean()
        def f(x):
            n_particles = x.shape[0]
            j = [f_per_particle(x[i]) for i in range(n_particles)]
            return np.array(j)
        options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 2, 'p':2}
        optimizer = ps.discrete.BinaryPSO(n_particles=5, dimensions=dim, options=options)
        cost, pos = optimizer.optimize(f, iters=5)
        self.assertTrue(f_per_particle(pos) == cost)

    def test_global(self):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
        cost, pos = optimizer.optimize(fx.sphere, iters=5)
        self.assertTrue(sum(pos ** 2) == cost)

    def test_local(self):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
        optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)
        cost, pos = optimizer.optimize(fx.sphere, iters=5)
        self.assertTrue(sum(pos ** 2) == cost)

    def test_general(self):
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        my_topology = Pyramid(static=False)
        optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                            options=options, topology=my_topology)
        cost, pos = optimizer.optimize(fx.sphere, iters=5)
        print("\n\n", pos)
        print(sum(pos ** 2), '\n\n')
        self.assertTrue(sum(pos ** 2) == cost)

if __name__ == '__main__':
    unittest.main()
