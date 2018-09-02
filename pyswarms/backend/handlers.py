"""
Handler Backend

This module provides Handler classes for the position as well as
the velocity of particles. This is necessary when boundary conditions
are imposed on the PSO algorithm. Particles that do not stay inside
these boundary conditions have to be handled by either adjusting their
position after they left the bounded search space or adjusting their
velocity when it would position them outside the search space.
"""

import logging

import numpy as np

from ..utils.reporter import Reporter

rep = Reporter(logger=logging.getLogger(__name__))


class BoundaryHandler:
    def __init__(self, strategy):
        """ A BoundaryHandler class

        This class offers a way to handle boundary conditions. It contains
        methods to avoid having particles outside of the defined boundaries.
        It repairs the position of particles that would leave the boundares
        in the next optimization step by using one of the follwing methods:

        * Nearest:
            Reposition the particle to the nearest bound.
        * Random:
            Reposition the particle randomly in between the bounds.
        * Shrink:
            Shrink the velocity of the particle such that it lands on the
            bounds.
        * Reflective:
            Mirror the particle position from outside the bounds to inside the
            bounds.
        * Intermediate:
            Reposition the particle to the midpoint between its current
            position on the bound surpassing axis and the bound itself.
            This only adjusts the axes that surpass the boundaries.
        * Resample:
            Redraw the velocity until the next position is inside the bounds.

        Attributes
        ----------
        strategy : str
            The strategy to be used.
            The following are available:
                * "nearest"

                * "random"

                * "shrink"

                * "reflective"

                * "intermediate"

                * "resample"

            For a description of these see above.
        """
        self.strategy = strategy

    def __call__(self, position, bounds, *args, **kwargs):
        """Make class callable

        The BoundaryHandler can be called as a function to use the strategy
        that is passed at initialization to repair boundary issues. An example
        for the usage:

        .. code-block :: python

            from pyswarms.backend import operators as op
            from pyswarms.backend.handlers import BoundaryHandler

            bh = BoundaryHandler(strategy="reflective")
            ops.compute_position(swarm, bounds, handler=bh)


        Parameters
        ----------
        position : np.ndarray
            The swarm position to be handled
        bounds : tuple of :code:`np.ndarray` or list
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`
        *args : tuple
        **kwargs : dict

        Returns
        -------
        numpy.ndarray
            the adjusted positions of the swarm
        """
        # Assign new attributes
        self.position = position
        self.lower_bound, self.upper_bound = bounds
        self.__out_of_bounds()

        if self.strategy == "nearest":
            new_position = self.nearest()
        elif self.strategy == "reflective":
            new_position = self.reflective()
        elif self.strategy == "shrink":
            new_position = self.shrink()
        elif self.strategy == "random":
            new_position = self.random()
        elif self.strategy == "intermediate":
            new_position = self.random()
        elif self.strategy == "resample":
            new_position = self.resample()

        return self.position

    def __out_of_bounds(self):
        """
        Helper method to find indices

        This helper methods finds the indices of the positions that do
        transgress the imposed bounds and stores them in class attributes
        """
        self.greater_than_bound = np.nonzero(self.position > self.upper_bound)
        self.lower_than_bound = np.nonzero(self.position < self.lower_bound)

    def nearest(self):
        """
        Set position to nearest bound

        This method resets particles that exceed the bounds to the nearest
        available bound. For every axis on which the coordiantes of the particle
        surpasses the boundary conditions the coordinate is set to the respective
        bound that it surpasses.
        """
        bool_greater = self.position > self.upper_bound
        bool_lower = self.position < self.lower_bound
        self.position = np.where(bool_greater, self.upper_bound, self.position)
        self.position = np.where(bool_lower, self.lower_bound, self.position)

    def reflective(self):
        pass

    def shrink(self):
        pass

    def random(self):
        """
        Set position to random location

        This method resets particles that exeed the bounds to a random position
        inside the boundary conditions.
        """
        sample = np.random.sample((self.position.shape[0],))
        self.position[self.greater_than_bound[0]] = np.array(
            [
                (self.upper_bound[i] - self.lower_bound[i]) * sample[i]
                + self.lower_bound[i]
                for i in range(sample.size)
            ]
        )

    def intermediate(self):
        pass

    def resample(self):
        pass
