"""Handlers

This module provides Handler classes for the position as well as the velocity
of particles. This is necessary when boundary conditions are imposed on the PSO
algorithm. Particles that do not stay inside these boundary conditions have to
be handled by either adjusting their position after they left the bounded
search space or adjusting their velocity when it would position them outside
the search space.

"""

import inspect
import logging

import numpy as np

from ..utils.reporter import Reporter


class BoundaryHandler(object):
    def __init__(self, strategy):
        """ A BoundaryHandler class

        This class offers a way to handle boundary conditions. It contains
        methods to avoid having particles outside of the defined boundaries.
        It repairs the position of particles that would leave the boundares in
        the next optimization step by using one of the follwing methods:

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
            position on the bound surpassing axis and the bound itself.  This
            only adjusts the axes that surpass the boundaries.
        * Resample:
            Redraw the velocity until the next position is inside the bounds.

        The BoundaryHandler can be called as a function to use the strategy
        that is passed at initialization to repair boundary issues. An example
        for the usage:

        .. code-block :: python

            from pyswarms.backend import operators as op
            from pyswarms.backend.handlers import BoundaryHandler

            bh = BoundaryHandler(strategy="reflective")
            ops.compute_position(swarm, bounds, handler=bh)

        Attributes
        ----------
        strategy : str
            The strategy to use. To see all available strategies,
            call :code:`BoundaryHandler.strategies`
        """
        self.strategy = strategy
        self.strategies = self.__get_all_strategies()
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def __call__(self, position, bounds, **kwargs):
        """Apply the selected strategy to the position-matrix given the bounds

        Parameters
        ----------
        position : np.ndarray
            The swarm position to be handled
        bounds : tuple of :code:`np.ndarray` or list
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`
        kwargs : dict

        Returns
        -------
        numpy.ndarray
            the adjusted positions of the swarm
        """
        # Combine `position` and `bounds` with extra keyword args
        kwargs_ = {**{"position": position, "bounds": bounds}, **kwargs}

        try:
            new_position = self.strategies[self.strategy](**kwargs_)
        except KeyError:
            message = "Unrecognized strategy: {}. Choose one among: " + str(
                [strat for strat in self.strategies.keys()]
            )
            self.rep.log.exception(message.format(self.strategy))
            raise
        else:
            return new_position

    def __out_of_bounds(self, position, bounds):
        """Helper method to find indices of out-of-bound positions"""
        lb, ub = bounds
        greater_than_bound = np.nonzero(position > ub)
        lower_than_bound = np.nonzero(position < lb)
        return (lower_than_bound, greater_than_bound)

    def __get_all_strategies(self):
        """Helper method to automatically generate a dict of strategies"""
        return {
            k: v
            for k, v in inspect.getmembers(self, predicate=inspect.isroutine)
            if not k.startswith(("__", "_"))
        }

    def nearest(self, **k):
        """Set position to nearest bound

        This method resets particles that exceed the bounds to the nearest
        available bound. For every axis on which the coordiantes of the particle
        surpasses the boundary conditions the coordinate is set to the respective
        bound that it surpasses.
        """
        try:
            lb, ub = k["bounds"]
            bool_greater = k["position"] > ub
            bool_lower = k["position"] < lb
            new_pos = np.where(bool_greater, ub, k["position"]).where(
                bool_lower, lb, k["position"]
            )
        except KeyError:
            raise
        else:
            return new_pos

    def reflective(self, **k):
        pass

    def shrink(self, **k):
        pass

    def random(self, **k):
        """Set position to random location

        This method resets particles that exeed the bounds to a random position
        inside the boundary conditions.
        """
        lb, ub = k["bounds"]
        sample = np.random.sample((k["position"].shape[0],))
        lower_than_bound, greater_than_bound = self.__out_of_bounds(
            k["position"], k["bounds"]
        )
        # Set indices that are greater than bounds
        k["position"][greater_than_bound[0]] = np.array(
            [(ub[i] - lb[i]) * sample[i] + lb[i] for i in range(sample.size)]
        )

    def intermediate(self, **k):
        pass

    def resample(self, **k):
        pass
