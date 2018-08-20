# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.utils.decorators import cost


@pytest.mark.parametrize(
    "objective_func",
    [np.sum, np.prod]
)
def test_cost_decorator(objective_func, particles):
    n_particles = particles.shape[0]

    def cost_func_without_decorator(x):
        n_particles_in_func = x.shape[0]
        cost = np.array([objective_func(x[i]) for i in range(n_particles_in_func)])
        return cost

    @cost
    def cost_func_with_decorator(x):
        cost = objective_func(x)
        return cost

    undecorated = cost_func_without_decorator(particles)
    decorated = cost_func_with_decorator(particles)

    assert np.array_equal(decorated, undecorated)
    assert decorated.shape == (n_particles, )
