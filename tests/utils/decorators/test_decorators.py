from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

from pyswarms.utils.decorators import cost


@pytest.mark.parametrize("objective_func", [np.sum, np.prod])
def test_cost_decorator(objective_func: Callable[[npt.NDArray[Any]], float], particles: npt.NDArray[Any]):
    """Test if cost decorator returns the same shape and value as undecorated function"""
    n_particles = particles.shape[0]

    def cost_func_without_decorator(x: npt.NDArray[Any]):
        n_particles_in_func = x.shape[0]
        cost = np.array([objective_func(x[i]) for i in range(n_particles_in_func)])
        return cost

    @cost
    def cost_func_with_decorator(x: npt.NDArray[Any]):
        cost = objective_func(x)
        return cost

    undecorated = cost_func_without_decorator(particles)
    decorated = cost_func_with_decorator(particles)

    assert np.array_equal(decorated, undecorated)
    assert decorated.shape == (n_particles,)


def test_decorator_invalid_cost_func(particles: npt.NDArray[Any]):
    """Test if ValueError is raised whenever an invalid cost function is passed"""

    def objective_func(x: npt.NDArray[Any]):
        """Returns a numpy.ndarray instead of int or float"""
        return np.array([1, 3])

    @cost  # type: ignore
    def cost_func_with_wrong_output_shape_decorated(x: npt.NDArray[Any]):
        cost = objective_func(x)
        return cost

    with pytest.raises(ValueError):
        cost_func_with_wrong_output_shape_decorated(particles)
