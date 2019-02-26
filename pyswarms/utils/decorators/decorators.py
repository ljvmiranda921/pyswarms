# Import modules
import numpy as np
from functools import wraps


def cost(cost_func):
    """A decorator for the cost function

    This decorator allows the creation of much simpler cost functions. Instead
    of writing a cost function that returns a shape of :code:`(n_particles, 0)`
    it enables the usage of shorter and simpler cost functions that directly
    return the cost.  A simple example might be:

     .. code-block:: python
         import pyswarms
         import numpy as np

         @pyswarms.cost
         def cost_func(x):
            cost = np.abs(np.sum(x))
            return cost

    The decorator expects your cost function to use a d-dimensional array
    (where d is the number of dimensions for the optimization) as and argument.

    .. note::
        Some :code:`numpy` functions return a :code:`np.ndarray` with single
        values in it.  Be aware of the fact that without unpacking the value
        the optimizer will raise an exception.

    Parameters
    ----------
    cost_func : callable
        A callable object that can be used as cost function in the optimization
        (must return a :code:`float` or an :code:`int`).

    Returns
    -------
    callable
        The vectorized output for all particles as defined by :code:`cost_func`
    """

    @wraps(cost_func)
    def cost_dec(particles, **kwargs):
        n_particles = particles.shape[0]
        vector = np.array(
            [cost_func(particles[i], **kwargs) for i in range(n_particles)]
        )
        if vector.shape != (n_particles,):
            msg = "Cost function must return int or float. You passed: {}"
            cost_return_type = type(cost_func(particles[0], **kwargs))
            raise ValueError(msg.format(cost_return_type))
        return vector

    return cost_dec
