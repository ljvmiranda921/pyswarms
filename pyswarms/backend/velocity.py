import numpy as np


from typing import Optional, TypedDict
from pyswarms.backend.handlers import VelocityHandler
from pyswarms.backend.swarms import Swarm
from pyswarms.utils.types import Bounds, Clamp


class SwarmOptions(TypedDict):
    c1: float
    c2: float
    w: float


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

    def __init__(self, options: SwarmOptions, clamp: Optional[Clamp], vh: VelocityHandler, bounds: Optional[Bounds] = None):
        self.options = options
        self.clamp = clamp
        self.vh = vh
        self.bounds = bounds

    def compute(self, swarm: Swarm):
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
        
        # Compute for cognitive and social terms
        cognitive = self.c1 * np.random.uniform(0, 1, swarm_size) * (swarm.pbest_pos - swarm.position)
        social = self.c2 * np.random.uniform(0, 1, swarm_size) * (swarm.best_pos - swarm.position)
        
        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (self.w * swarm.velocity) + cognitive + social
        updated_velocity = self.vh(temp_velocity, self.clamp, position=swarm.position, bounds=self.bounds)

        return updated_velocity
    
    @property
    def c1(self):
        return self.options["c1"]
    
    @property
    def c2(self):
        return self.options["c2"]
    
    @property
    def w(self):
        return self.options["w"]