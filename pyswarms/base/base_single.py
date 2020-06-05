# -*- coding: utf-8 -*-

r"""
Base class for single-objective Particle Swarm Optimization
implementations.

All methods here are abstract and raise a :code:`NotImplementedError`
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
:mod:`pyswarms.single.general_optimizer`: a more general PSO implementation with a custom topology
"""

# Import standard library
import abc
from collections import namedtuple

# Import modules
import numpy as np

from ..backend import create_swarm


class SwarmOptimizer(abc.ABC):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        velocity_clamp=None,
        center=1.0,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
    ):
        """Initialize the swarm

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
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        center : list, optional
            an array of size :code:`dimensions`
        ftol : float, optional
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`.
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        """
        # Initialize primary swarm attributes
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.velocity_clamp = velocity_clamp
        self.swarm_size = (n_particles, dimensions)
        self.options = options
        self.center = center
        self.ftol = ftol

        try:
            assert ftol_iter > 0 and isinstance(ftol_iter, int)
        except AssertionError:
            raise AssertionError(
                "ftol_iter expects an integer value greater than 0"
            )

        self.ftol_iter = ftol_iter
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
        # Initialize resettable attributes
        self.reset()

    def _populate_history(self, hist):
        """Populate all history lists

        The :code:`cost_history`, :code:`mean_pbest_history`, and
        :code:`neighborhood_best` is expected to have a shape of
        :code:`(iters,)`,on the other hand, the :code:`pos_history`
        and :code:`velocity_history` are expected to have a shape of
        :code:`(iters, n_particles, dimensions)`

        Parameters
        ----------
        hist : collections.namedtuple
            Must be of the same type as self.ToHistory
        """
        self.cost_history.append(hist.best_cost)
        self.mean_pbest_history.append(hist.mean_pbest_cost)
        self.mean_neighbor_history.append(hist.mean_neighbor_cost)
        self.pos_history.append(hist.position)
        self.velocity_history.append(hist.velocity)

    @abc.abstractmethod
    def optimize(self, objective_func, iters, n_processes=None, **kwargs):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`objective_func` for a number of iterations
        :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation
            Default is None with no parallelization
        kwargs : dict
            arguments for objective function

        Raises
        ------
        NotImplementedError
            When this method is not implemented.
        """
        raise NotImplementedError("SwarmOptimizer::optimize()")

    def reset(self):
        """Reset the attributes of the optimizer

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
