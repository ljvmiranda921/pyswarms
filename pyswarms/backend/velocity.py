from typing import Callable, Dict, Optional, Tuple

import numpy as np

from pyswarms.backend.handlers import OptionsHandler, VelocityHandler, VelocityStrategy
from pyswarms.backend.swarms import Swarm
from pyswarms.utils.types import Bounds, Clamp, OptionsStrategy, SwarmOption, SwarmOptions, Velocity


class VelocityUpdater:
    """Class for updating particle velocities

    Attributes
    ----------
    options : SwarmOptions
        Dictionary containing the 3 parameters for evolution:
            * c1 : float|Tuple[OptionsStrategy, float]|OptionsHandler
                cognitive parameter
            * c2 : float|Tuple[OptionsStrategy, float]|OptionsHandler
                social parameter
            * w : float|Tuple[OptionsStrategy, float]|OptionsHandler
                inertia parameter
    clamp : tuple of floats, optional
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.
    vh : pyswarms.backend.handlers.VelocityHandler
        a VelocityHandler object with a specified handling strategy.
        For further information see :mod:`pyswarms.backend.handlers`.
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    """
    options: Dict[SwarmOption, Callable[[int, int], float]]

    def __init__(
        self,
        options: SwarmOptions,
        clamp: Optional[Clamp],
        vh: VelocityStrategy | VelocityHandler = "unmodified",
        bounds: Optional[Bounds] = None,
    ):
        self.clamp = clamp
        self.bounds = bounds
        self.init_options(options)

        if isinstance(vh, str):
            self.vh = VelocityHandler.factory(vh, self.clamp, self.bounds)
        else:
            self.vh = vh

    def init_options(self, options: SwarmOptions):
        self.options = {
            "c1": self.init_option("c1", options["c1"]),
            "c2": self.init_option("c2", options["c2"]),
            "w": self.init_option("w", options["w"]),
        }

    def init_option(self, option: SwarmOption, value: float|Tuple[OptionsStrategy, float]|OptionsHandler) -> Callable[[int, int], float]:
        if isinstance(value, float|int):
            def get_option(*_: int):
                return value
            return get_option
        elif isinstance(value, tuple):
            strategy, start_value = value
            return OptionsHandler.factory(strategy, option, start_value)
        elif isinstance(value, OptionsHandler):
            return value
        
        raise ValueError(f"Option value should be float, Tuple[OptionsStrategy, float] or OptionsHandler, received {type(value)}")
    
    def get_options(self, iter_cur: int, iter_max: int):
        return (
            self.options["c1"](iter_cur, iter_max),
            self.options["c2"](iter_cur, iter_max),
            self.options["w"](iter_cur, iter_max),
        )

    def compute(self, swarm: Swarm, iter_cur: int, iter_max: int):
        """Update the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm. The velocity is handled
        by a :code:`VelocityHandler`.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.swarms.backend import Swarm, VelocityHandler

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_vh = VelocityHandler(strategy="invert")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = compute_velocity(my_swarm, clamp, my_vh, bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        # Prepare parameters
        swarm_size = swarm.position.shape

        # Perform options update
        c1, c2, w = self.get_options(iter_cur, iter_max)

        # Compute for cognitive and social terms
        cognitive = c1 * np.random.uniform(0, 1, swarm_size) * (swarm.pbest_pos - swarm.position)
        social = c2 * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position)

        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * swarm.velocity) + cognitive + social
        updated_velocity = self.vh(temp_velocity, swarm.position)

        return updated_velocity

    def generate_velocity(self, n_particles: int, dimensions: int) -> Velocity:
        """Initialize a velocity vector

        Parameters
        ----------
        n_particles : int
            number of particles to be generated in the swarm.
        dimensions: int
            number of dimensions to be generated in the swarm.
        clamp : tuple of floats, optional
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping. Default is :code:`None`

        Returns
        -------
        numpy.ndarray
            velocity matrix of shape (n_particles, dimensions)
        """
        min_velocity, max_velocity = (0, 1) if self.clamp is None else np.array(self.clamp)

        velocity = (max_velocity - min_velocity) * np.random.random_sample(
            size=(n_particles, dimensions)
        ) + min_velocity

        return velocity
