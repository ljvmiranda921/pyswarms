"""
Handlers

This module provides Handler classes for the position as well as the velocity
of particles. This is necessary when boundary conditions are imposed on the PSO
algorithm. Particles that do not stay inside these boundary conditions have to
be handled by either adjusting their position after they left the bounded
search space or adjusting their velocity when it would position them outside
the search space.
For the follwing documentation let :math:`x_{i, t, d}` be the :math:`d` th
coordinate of the particle :math:`i` 's position vector at the time :math:`t`,
:math:`lb` the vector of the lower boundaries and :math:`ub` the vector of the
upper boundaries.
"""

import inspect
import logging

import numpy as np

from ..utils.reporter import Reporter
from .operators import compute_velocity


class BoundaryHandler(object):
    def __init__(self, strategy):
        """ A BoundaryHandler class

        This class offers a way to handle boundary conditions. It contains
        methods to repair particle positions outside of the defined boundaries.
        Following strategies are available for the handling:

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

        The BoundaryHandler can be called as a function to use the strategy
        that is passed at initialization to repair boundary issues. An example
        for the usage:

        .. code-block :: python

            from pyswarms.backend import operators as op
            from pyswarms.backend.handlers import BoundaryHandler

            bh = BoundaryHandler(strategy="reflective")
            ops.compute_position(swarm, bounds, handler=bh)

        By passing the handler, the :func:`compute_position()` functions now has
        the ability to reset the particles by calling the :code:`BoundaryHandler`
        inside.

        Attributes
        ----------
        strategy : str
            The strategy to use. To see all available strategies,
            call :code:`BoundaryHandler.strategies`
        """
        self.strategy = strategy
        self.strategies = self.__get_all_strategies()
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.memory = None

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
        kwargs_ = self.__merge_dicts(
            {"position": position, "bounds": bounds}, kwargs
        )

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

    def __merge_dicts(self, *dict_args):
        """Backward-compatible helper method to combine two dicts"""
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def __out_of_bounds(self, position, bounds, velocity=None):
        """Helper method to find indices of out-of-bound positions

        This method finds the indices of the particles that are out-of-bound
        if a velocity is specified it returns the indices of the particles that
        will be out-of-bounds after the velocity is applied.
        """
        if velocity is not None:
            position += velocity
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
        r"""Set position to nearest bound

        This method resets particles that exceed the bounds to the nearest
        available bound. For every axis on which the coordiantes of the particle
        surpasses the boundary conditions the coordinate is set to the respective
        bound that it surpasses.
        The following equation describes this strategy:

        .. math::

            x_{i, t, d} = \begin{cases}
                                lb_d & \quad \text{if }x_{i, t, d} < lb_d \\
                                ub_d & \quad \text{if }x_{i, t, d} > ub_d \\
                                x_{i, t, d} & \quad \text{otherwise}
                          \end{cases}

        """
        try:
            lb, ub = k["bounds"]
            bool_greater = k["position"] > ub
            bool_lower = k["position"] < lb
            new_pos = np.where(bool_lower, lb, k["position"])
            new_pos = np.where(bool_greater, ub, new_pos)
        except KeyError:
            self.rep.log.exception("Keyword 'bounds'  or 'position' missing")
            raise
        else:
            return new_pos

    def reflective(self, **k):
        pass

    def shrink(self, **k):
        r"""Set the particle to the boundary

        This methods resets particles that exceed the bounds to the intersection
        of its previous velocity and the bound. This can be imagined as shrinking
        the previous velocity until the particle is back in the valid search space.
        Let :math:`\sigma_{i, t, d}` be the :math:`d` th shrinking value of the
        :math:`i` th particle at the time :math:`t` and :math:`v_{i, t}` the velocity
        of the :math:`i` th particle at the time :math:`t`. Then the new position
        computed by the follwing equation:

        .. math::
            :nowrap:

            \begin{gather*}
            \mathbf{x}_{i, t} = \mathbf{x}_{i, t-1} + \sigma_{i, t} \mathbf{v}_{i, t} \\
            \\
            \text{with} \\
            \\
            \sigma_{i, t, d} = \begin{cases}
                                \frac{lb_d-x_{i, t-1, d}}{v_{i, t, d}} & \quad \text{if } x_{i, t, d} < lb_d \\
                                \frac{ub_d-x_{i, t-1, d}}{v_{i, t, d}} & \quad \text{if } x_{i, t, d} > ub_d \\
                                1 & \quad \text{otherwise}
                          \end{cases} \\
            \\
            \text{and} \\
            \\
            \sigma_{i, t} = \min_{d=1...n} \sigma_{i, t, d}
            \\
            \end{gather*}

        """
        try:
            if self.memory is None:
                new_pos = k["position"]
                self.memory = new_pos
            else:
                lb, ub = k["bounds"]
                lower_than_bound, greater_than_bound = self.__out_of_bounds(
                    k["position"], k["bounds"]
                )
                velocity = k["position"] - self.memory
                # Create a coefficient matrix
                sigma = np.tile(1, k["position"].shape)
                sigma[lower_than_bound] = (
                    lb[lower_than_bound[1]] - self.memory[lower_than_bound]
                ) / velocity[lower_than_bound]
                min_sigma = np.amin(sigma, axis=1)
                new_pos = k["position"]
                new_pos[lower_than_bound[0]] = (
                    self.memory[lower_than_bound[0]]
                    + min_sigma[lower_than_bound[0]]
                    * velocity[lower_than_bound[0]]
                )
                self.memory = new_pos
        except KeyError:
            self.rep.log.exception("Keyword 'bounds' or 'position' missing")
            raise
        else:
            return new_pos

    def random(self, **k):
        """Set position to random location

        This method resets particles that exeed the bounds to a random position
        inside the boundary conditions.
        """
        try:
            lb, ub = k["bounds"]
            lower_than_bound, greater_than_bound = self.__out_of_bounds(
                k["position"], k["bounds"]
            )
            # Set indices that are greater than bounds
            new_pos = k["position"]
            new_pos[greater_than_bound[0]] = np.array(
                [
                    (ub - lb)
                    * np.random.random_sample((k["position"].shape[1],))
                    + lb
                ]
            )
            new_pos[lower_than_bound[0]] = np.array(
                [
                    (ub - lb)
                    * np.random.random_sample((k["position"].shape[1],))
                    + lb
                ]
            )
        except KeyError:
            self.rep.log.exception("Keyword 'bounds' or 'position' missing")
            raise
        else:
            return new_pos

    def intermediate(self, **k):
        r"""Set the particle to an intermediate position

        This method resets particles that exceed the bounds to an intermediate
        position between the bound and their earlier position. Namely, it changes
        the coordinate of the out-of-bounds axis to the middle value between the
        previous position and the boundary of the axis.
        The follwing equation describes this strategy:

        .. math::

            x_{i, t, d} = \begin{cases}
                                \frac{1}{2} \left (x_{i, t-1, d} + lb_d \right) & \quad \text{if }x_{i, t, d} < lb_d \\
                                \frac{1}{2} \left (x_{i, t-1, d} + ub_d \right) & \quad \text{if }x_{i, t, d} > ub_d \\
                                x_{i, t, d} & \quad \text{otherwise}
                          \end{cases}

        """
        try:
            if self.memory is None:
                new_pos = k["position"]
                self.memory = new_pos
            else:
                lb, ub = k["bounds"]
                lower_than_bound, greater_than_bound = self.__out_of_bounds(
                    k["position"], k["bounds"]
                )
                new_pos = k["position"]
                new_pos[lower_than_bound] = 0.5 * (
                    self.memory[lower_than_bound] + lb[lower_than_bound[1]]
                )
                new_pos[greater_than_bound] = 0.5 * (
                    self.memory[greater_than_bound] + ub[greater_than_bound[1]]
                )
                self.memory = new_pos
        except KeyError:
            self.rep.log.exception("Keyword 'bound' or 'position' missing")
            raise
        else:
            return new_pos

    def periodic(self, **k):
        r"""Sets the particles a periodic fashion

        This method resets the particles that exeed the bounds by using the
        modulo function to cut down the position. This creates a virtual,
        periodic plane which is tiled with the search space.
        The follwing equation describtes this strategy:

        .. math::
            :nowrap:

            \begin{gather*}
            x_{i, t, d} = \begin{cases}
                                ub_d - (lb_d - x_{i, t, d}) \mod s_d & \quad \text{if }x_{i, t, d} < lb_d \\
                                lb_d + (x_{i, t, d} - ub_d) \mod s_d & \quad \text{if }x_{i, t, d} > ub_d \\
                                x_{i, t, d} & \quad \text{otherwise}
                          \end{cases}\\
            \\
            \text{with}\\
            \\
            s_d = |ub_d - lb_d|
            \end{gather*}

        """
        try:
            lb, ub = k["bounds"]
            lower_than_bound, greater_than_bound = self.__out_of_bounds(
                k["position"], k["bounds"]
            )
            bound_d = np.abs(ub - lb)
            new_pos = k["position"]
            new_pos[lower_than_bound[0]] = np.remainder(
                (ub - lb + new_pos[lower_than_bound[0]]), bound_d
            )
            new_pos[greater_than_bound[0]] = np.remainder(
                (lb + (new_pos[greater_than_bound[0]] - ub)), bound_d
            )
        except KeyError:
            self.rep.log.exception("Keyword 'bound' or 'position' missing")
            raise
        else:
            return new_pos


# TODO Finish it
class VelocityHandler(object):
    def __init__(self, strategy):
        """ A VelocityHandler class

        This class offers a way to handle velocities. It contains
        methods to repair the velocities of particles that exceeded the
        defined boundaries. Following strategies are available for the handling:

        * Resample:
            Redraw the velocity until the next position is inside the bounds.
        """
        self.strategy = strategy
        self.strategies = self.__get_all_strategies()
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.memory = None

    def __call__(self, velocity, clamp, **kwargs):
        """Apply the selected strategy to the velocity-matrix given the bounds

        Parameters
        ----------
        velocity : np.ndarray
            The swarm position to be handled
        clamp : tuple of :code:`np.ndarray` or list
            a tuple of size 2 where the first entry is the minimum clamp while
            the second entry is the maximum clamp. Each array must be of shape
            :code:`(dimensions,)`
        kwargs : dict

        Returns
        -------
        numpy.ndarray
            the adjusted positions of the swarm
        """
        # Combine `position` and `bounds` with extra keyword args
        kwargs_ = self.__merge_dicts(
            {"position": position, "bounds": bounds}, kwargs
        )

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

    def resample(self, **k):
        """Redraw velocity until the particle is feasible

        This method redraws the particle velocity if it would cause a particle to
        go out-of-bounds in the next optimization step.
        """
        try:
            lb, ub = k["bounds"]
            new_vel = k["velocity"]
            while True:
                lower_than_bound, greater_than_bound = self.__out_of_bounds(
                    k["position"], k["bounds"], k["velocity"]
                )

                if not lower_than_bound and not greater_than_bound:
                    break

                # TODO Create a more efficient method to redraw the velocity
                # One possibility would be to force it to the middle of the
                # boundaries by using a dummy swarm with all pbests and gbests
                # in the middle. Another one is to reduce the clamp every time it
                # unsuccessfully redraws the velocity.
                masking_vel = compute_velocity(k["swarm"], k["clamp"])
                new_vel[lower_than_bound[0]] = masking_vel[lower_than_bound[0]]
                new_vel[greate_than_bound[0]] = masking_vel[
                    greater_than_bound[0]
                ]
        except KeyError:
            self.rep.log.exception("Keyword 'bound' or 'position' missing")
            raise
        else:
            return new_vel
