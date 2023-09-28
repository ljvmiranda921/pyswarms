# -*- coding: utf-8 -*-

r"""
Base class for single-objective discrete Particle Swarm Optimization
implementations.

All methods here are abstract and raises a :code:`NotImplementedError`
when not used. When defining your own swarm implementation,
create another class,

    >>> class MySwarm(DiscreteSwarmOptimizer):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

and define all the necessary methods needed.

As a guide, check the discrete PSO implementations in this package.

.. note:: Regarding :code:`options`, it is highly recommended to
    include parameters used in position and velocity updates as
    keyword arguments. For parameters that affect the topology of
    the swarm, it may be much better to have them as positional
    arguments.

See Also
--------
:mod:`pyswarms.discrete.binary`: binary PSO implementation

"""

import abc
import multiprocessing as mp
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from loguru import logger
from tqdm import trange

from pyswarms.backend.operators import compute_objective_function
from pyswarms.backend.position import PositionUpdater
from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.base import Topology
from pyswarms.backend.velocity import VelocityUpdater
from pyswarms.utils.types import Position, Velocity


class Options(TypedDict):
    c1: float
    c2: float
    w: float


class BaseSwarmOptimizer(abc.ABC):
    # Initialize history lists
    cost_history: List[float] = []
    mean_pbest_history: List[float] = []
    mean_neighbor_history: List[float] = []
    pos_history: List[Position] = []
    velocity_history: List[Velocity] = []

    # Will be set by the _init_swarm method
    swarm: Swarm

    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        topology: Topology,
        velocity_updater: VelocityUpdater,
        position_updater: PositionUpdater,
        init_pos: Optional[Position] = None,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
    ):
        """Initialize the swarm.

        Creates a :code:`numpy.ndarray` of positions depending on the
        number of particles needed and the number of dimensions.
        The initial positions of the particles depends on the argument
        :code:`binary`, which governs if a binary matrix will be produced.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        velocity_updater : VelocityUpdater
            Class for updating the velocity matrix.
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        ftol : float
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
        self.top = topology
        self.velocity_updater = velocity_updater
        self.position_updater = position_updater
        self.swarm_size = (n_particles, dimensions)
        self.init_pos = init_pos
        self.ftol = ftol

        assert ftol_iter > 0 and isinstance(ftol_iter, int), "ftol_iter expects an integer value greater than 0"

        self.ftol_iter = ftol_iter
        self.name = __name__

        # Initialize resettable attributes
        self.reset()

    @abc.abstractmethod
    def _init_swarm(self) -> None:
        """Initialise a new swarm object"""
        ...

    def _populate_history(self):
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
        self.cost_history.append(self.swarm.best_cost)
        self.mean_pbest_history.append(float(np.mean(self.swarm.pbest_cost)))
        self.mean_neighbor_history.append(self.swarm.best_cost)
        self.pos_history.append(self.swarm.position)
        self.velocity_history.append(self.swarm.velocity)

    def optimize(
        self,
        objective_func: Callable[..., npt.NDArray[Any]],
        iters: int,
        n_processes: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any
    ) -> Tuple[float, Position]:
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        self._setup(n_processes, verbose)
        logger.debug("Obj. func. args: {}".format(kwargs))

        self.pbar = trange(iters, desc=self.name, disable=not verbose)
        for i in self.pbar:
            if not self._step(i, objective_func, iters, **kwargs):
                break

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()

        # Write report in log and return final cost and position
        logger.log(
            self.log_level,
            "Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos),
        )

        self._teardown()

        return (final_best_cost, final_best_pos)

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

        self._init_swarm()

    def _setup(self, n_processes: Optional[int], verbose: bool):
        self.log_level = "DEBUG" if verbose else "TRACE"

        # Setup Pool of processes for parallel evaluation
        self.pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        self.ftol_history: Deque[bool] = deque(maxlen=self.ftol_iter)

    def _teardown(self):
        if self.pool is not None:
            self.pool.close()

    def _step(self, i: int, objective_func: Callable[..., npt.NDArray[Any]], iters: int, **kwargs: Any) -> bool:
        # Compute cost for current position and personal best
        self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=self.pool, **kwargs)
        self.swarm.compute_pbest()
        best_cost_yet_found = self.swarm.best_cost

        # Update swarm
        self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)

        self._populate_history()

        # Verify stop criteria based on the relative acceptable cost ftol
        relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
        delta = np.abs(self.swarm.best_cost - best_cost_yet_found) < relative_measure
        self.ftol_history.append(delta)
        if i >= self.ftol_iter and all(self.ftol_history):
            return False

        # Print to console
        self.pbar.set_postfix(best_cost=self.swarm.best_cost)  # type: ignore

        # Perform velocity and position updates
        self.swarm.velocity = self._compute_velocity(i, iters)
        self.swarm.position = self._compute_position(i, iters)

        return True

    def _compute_position(self, iter_cur: int, iter_max: int):
        """Update the position matrix of the swarm"""
        return self.position_updater.compute(self.swarm)

    def _compute_velocity(self, iter_cur: int, iter_max: int):
        """Update the velocity matrix of the swarm"""
        return self.velocity_updater.compute(self.swarm, iter_cur, iter_max)
