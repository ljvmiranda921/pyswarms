from typing import Dict, Optional

import numpy as np

from pyswarms.backend.handlers import OptionsHandler, OptionsStrategy, VelocityHandler, VelocityStrategy
from pyswarms.backend.swarms import Swarm
from pyswarms.utils.types import Bounds, Clamp, SwarmOptions, Velocity


class VelocityUpdater:
    """Class for updating particle velocities

    Attributes
    ----------
    options : SwarmOptions
        Dictionary containing the 3 parameters for evolution:
            * c1 : float
                cognitive parameter
            * c2 : float
                social parameter
            * w : float
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

    def __init__(
        self,
        options: SwarmOptions,
        clamp: Optional[Clamp],
        vh: VelocityStrategy | VelocityHandler = "unmodified",
        bounds: Optional[Bounds] = None,
        oh_strategy: Optional[Dict[str, OptionsStrategy]] = None,
    ):
        self.options = options
        self.clamp = clamp
        self.bounds = bounds

        if isinstance(vh, str):
            self.vh = VelocityHandler.factory(vh, self.clamp, self.bounds)
        else:
            self.vh = vh

        if oh_strategy is None:
            oh_strategy = {}
        self.oh = OptionsHandler(oh_strategy)

        self.iterations = 0

    def compute(self, swarm: Swarm, itermax: int = 0):
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
        options = self.oh(self.options, iternow=self.iterations, itermax=itermax)

        # Compute for cognitive and social terms
        cognitive = options["c1"] * np.random.uniform(0, 1, swarm_size) * (swarm.pbest_pos - swarm.position)
        social = options["c2"] * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position)

        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (options["w"] * swarm.velocity) + cognitive + social
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
