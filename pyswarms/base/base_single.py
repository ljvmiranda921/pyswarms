# -*- coding: utf-8 -*-

r"""
Base class for single-objective Particle Swarm Optimization
implementations.

All methods here are abstract and raises a :code:`NotImplementedError`
when not used. When defining your own swarm implementation,
create another class,

    >>> class MySwarm(SwarmBase):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

and define all the necessary methods needed.

As a guide, check the global best and local best implementations in this
package.

.. note:: Regarding :code:`options`, it is highly recommended to
    include parameters used in position and velocity updates as
    keyword arguments. For parameters that affect the topology of
    the swarm, it may be much better to have them as positional
    arguments.

See Also
--------
:mod:`pyswarms.single.global_best`: global-best PSO implementation
:mod:`pyswarms.single.local_best`: local-best PSO implementation
"""

import os
import yaml
import logging
import numpy as np
import logging.config
from collections import namedtuple

# Import from package
from ..backend import create_swarm


class SwarmOptimizer(object):
    def assertions(self):
        """Assertion method to check various inputs.

        Raises
        ------
        TypeError
            When the :code:`bounds` is not of type tuple
        IndexError
            When the :code:`bounds` is not of size 2.
            When the arrays in :code:`bounds` is not of equal size.
            When the shape of :code:`bounds` is not the same as :code:`dimensions`.
        ValueError
            When the value of :code:`bounds[1]` is less than
            :code:`bounds[0]`.
        """

        # Check setting of bounds
        if self.bounds is not None:
            if not isinstance(self.bounds, tuple):
                raise TypeError("Parameter `bound` must be a tuple.")
            if not len(self.bounds) == 2:
                raise IndexError("Parameter `bound` must be of size 2.")
            if not self.bounds[0].shape == self.bounds[1].shape:
                raise IndexError("Arrays in `bound` must be of equal shapes")
            if (
                not self.bounds[0].shape[0]
                == self.bounds[1].shape[0]
                == self.dimensions
            ):
                raise IndexError(
                    "Parameter `bound` must be the same shape "
                    "as dimensions."
                )
            if not (self.bounds[1] > self.bounds[0]).all():
                raise ValueError(
                    "Values of `bounds[1]` must be greater than "
                    "`bounds[0]`."
                )

        # Check clamp settings
        if hasattr(self.velocity_clamp, "__iter__"):
            if not len(self.velocity_clamp) == 2:
                raise IndexError(
                    "Parameter `velocity_clamp` must be of " "size 2"
                )
            if not self.velocity_clamp[0] < self.velocity_clamp[1]:
                raise ValueError(
                    "Make sure that velocity_clamp is in the "
                    "form (min, max)"
                )

        # Check setting of center
        if isinstance(self.center, (list, np.ndarray)):
            if not len(self.center) == self.dimensions:
                raise IndexError(
                    "Parameter `center` must be the same shape "
                    "as dimensions."
                )
            if isinstance(self.center, np.ndarray) and self.center.ndim != 1:
                raise ValueError("Parameter `center` must have a 1d array")

        # Required keys in options argument
        if not all(key in self.options for key in ("c1", "c2", "w")):
            raise KeyError("Missing either c1, c2, or w in options")

    def setup_logging(
        self,
        default_path="./config/logging.yaml",
        default_level=logging.INFO,
        env_key="LOG_CFG",
    ):
        """Setup logging configuration

        Parameters
        ----------
        default_path : str (default is `./config/logging.yaml`)
            the path where the logging configuration is stored
        default_level: logging.LEVEL (default is `logging.INFO`)
            the default logging level
        env_key : str
            the environment key for accessing the setup
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, "rt") as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        velocity_clamp=None,
        center=1.0,
        ftol=-np.inf,
        init_pos=None,
    ):
        """Initializes the swarm.

        Creates a Swarm class depending on the values initialized

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        bounds : tuple of :code:`np.ndarray` (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for convergence
        """
        self.setup_logging()
        # Initialize primary swarm attributes
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.velocity_clamp = velocity_clamp
        self.swarm_size = (n_particles, dimensions)
        self.options = options
        self.center = center
        self.ftol = ftol
        self.init_pos = init_pos
        # Initialize named tuple for populating the history list
        self.ToHistory = namedtuple(
            "ToHistory",
            [
                "best_cost",
                "mean_pbest_cost",
                "mean_neighbor_cost",
                "position",
                "velocity",
            ],
        )
        # Invoke assertions
        self.assertions()
        # Initialize resettable attributes
        self.reset()

    def _populate_history(self, hist):
        """Populates all history lists

        The :code:`cost_history`, :code:`mean_pbest_history`, and
        :code:`neighborhood_best` is expected to have a shape of
        :code:`(iters,)`,on the other hand, the :code:`pos_history`
        and :code:`velocity_history` are expected to have a shape of
        :code:`(iters, n_particles, dimensions)`

        Parameters
        ----------
        hist : namedtuple
            Must be of the same type as self.ToHistory
        """
        self.cost_history.append(hist.best_cost)
        self.mean_pbest_history.append(hist.mean_pbest_cost)
        self.mean_neighbor_history.append(hist.mean_neighbor_cost)
        self.pos_history.append(hist.position)
        self.velocity_history.append(hist.velocity)

    def optimize(self, objective_func, iters, *args, print_step=1, verbose=1, **kwargs):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function :code:`objective_func` for a number of iterations
        :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        print_step : int (the default is 1)
            amount of steps for printing into console.
        verbose : int (the default is 1)
            verbosity setting.

        Raises
        ------
        NotImplementedError
            When this method is not implemented.
        """
        raise NotImplementedError("SwarmOptimizer::optimize()")

    def reset(self):
        """Resets the attributes of the optimizer.

        All variables/atributes that will be re-initialized when this
        method is defined here. Note that this method
        can be called twice: (1) during initialization, and (2) when
        this is called from an instance.

        It is good practice to keep the number of resettable
        attributes at a minimum. This is to prevent spamming the same
        object instance with various swarm definitions.

        Normally, swarm definitions are as atomic as possible, where
        each type of swarm is contained in its own instance. Thus, the
        following attributes are the only ones recommended to be
        resettable:

        * Swarm position matrix (self.pos)
        * Velocity matrix (self.pos)
        * Best scores and positions (gbest_cost, gbest_pos, etc.)

        Otherwise, consider using positional arguments.
        """
        # Initialize history lists
        self.cost_history = []
        self.mean_pbest_history = []
        self.mean_neighbor_history = []
        self.pos_history = []
        self.velocity_history = []

        # Initialize the swarm
        self.swarm = create_swarm(
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            bounds=self.bounds,
            center=self.center,
            init_pos=self.init_pos,
            clamp=self.velocity_clamp,
            options=self.options,
        )
