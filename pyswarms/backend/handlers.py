"""
Handlers

This module provides Handler classes for the position as well as the velocity
of particles. This is necessary when boundary conditions are imposed on the PSO
algorithm. Particles that do not stay inside these boundary conditions have to
be handled by either adjusting their position after they left the bounded
search space or adjusting their velocity when it would position them outside
the search space. In particular, this approach is important if the optimium of
a function is near the boundaries.
For the following documentation let :math:`x_{i, t, d}` be the :math:`d` th
coordinate of the particle :math:`i` 's position vector at the time :math:`t`,
:math:`lb` the vector of the lower boundaries and :math:`ub` the vector of the
upper boundaries.
The algorithms in this module are adapted from [SH2010]

[SH2010] Sabine Helwig, "Particle Swarms for Constrained Optimization",
PhD thesis, Friedrich-Alexander Universität Erlangen-Nürnberg, 2010.
"""

import inspect
import logging

import numpy as np
import math
from copy import copy

from ..utils.reporter import Reporter


class HandlerMixin(object):
    """A HandlerMixing class

    This class offers some basic functionality for the Handlers.
    """

    def _merge_dicts(self, *dict_args):
        """Backward-compatible helper method to combine two dicts"""
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def _out_of_bounds(self, position, bounds):
        """Helper method to find indices of out-of-bound positions

        This method finds the indices of the particles that are out-of-bound.
        """
        lb, ub = bounds
        greater_than_bound = np.nonzero(position > ub)
        lower_than_bound = np.nonzero(position < lb)
        return (lower_than_bound, greater_than_bound)

    def _get_all_strategies(self):
        """Helper method to automatically generate a dict of strategies"""
        return {
            k: v
            for k, v in inspect.getmembers(self, predicate=inspect.isroutine)
            if not k.startswith(("__", "_"))
        }


class BoundaryHandler(HandlerMixin):
    def __init__(self, strategy):
        """A BoundaryHandler class

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

        By passing the handler, the :func:`compute_position()` function now has
        the ability to reset the particles by calling the :code:`BoundaryHandler`
        inside.

        Attributes
        ----------
        strategy : str
            The strategy to use. To see all available strategies,
            call :code:`BoundaryHandler.strategies`
        """
        self.strategy = strategy
        self.strategies = self._get_all_strategies()
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.memory = None

    def __call__(self, position, bounds, **kwargs):
        """Apply the selected strategy to the position-matrix given the bounds

        Parameters
        ----------
        position : numpy.ndarray
            The swarm position to be handled
        bounds : tuple of numpy.ndarray or list
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`
        kwargs : dict

        Returns
        -------
        numpy.ndarray
            the adjusted positions of the swarm
        """
        try:
            new_position = self.strategies[self.strategy](
                position, bounds, **kwargs
            )
        except KeyError:
            message = "Unrecognized strategy: {}. Choose one among: " + str(
                [strat for strat in self.strategies.keys()]
            )
            self.rep.logger.exception(message.format(self.strategy))
            raise
        else:
            return new_position

    def nearest(self, position, bounds, **kwargs):
        r"""Set position to nearest bound

        This method resets particles that exceed the bounds to the nearest
        available boundary. For every axis on which the coordiantes of the particle
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
        lb, ub = bounds
        bool_greater = position > ub
        bool_lower = position < lb
        new_pos = np.where(bool_lower, lb, position)
        new_pos = np.where(bool_greater, ub, new_pos)
        return new_pos

    def reflective(self, position, bounds, **kwargs):
        r"""Reflect the particle at the boundary

        This method reflects the particles that exceed the bounds at the
        respective boundary. This means that the amount that the component
        which is orthogonal to the exceeds the boundary is mirrored at the
        boundary. The reflection is repeated until the position of the particle
        is within the boundaries. The following algorithm describes the
        behaviour of this strategy:

        .. math::
            :nowrap:

            \begin{gather*}
                \text{while } x_{i, t, d} \not\in \left[lb_d,\,ub_d\right] \\
                \text{ do the following:}\\
                \\
                x_{i, t, d} =   \begin{cases}
                                    2\cdot lb_d - x_{i, t, d} & \quad \text{if } x_{i,
                                    t, d} < lb_d \\
                                    2\cdot ub_d - x_{i, t, d} & \quad \text{if } x_{i,
                                    t, d} > ub_d \\
                                    x_{i, t, d} & \quad \text{otherwise}
                                \end{cases}
            \end{gather*}
        """
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(
            position, bounds
        )
        new_pos = position
        while lower_than_bound[0].size != 0 or greater_than_bound[0].size != 0:
            if lower_than_bound[0].size > 0:
                new_pos[lower_than_bound] = (
                    2 * lb[lower_than_bound[1]] - new_pos[lower_than_bound]
                )
            if greater_than_bound[0].size > 0:
                new_pos[greater_than_bound] = (
                    2 * ub[greater_than_bound[1]] - new_pos[greater_than_bound]
                )
            lower_than_bound, greater_than_bound = self._out_of_bounds(
                new_pos, bounds
            )

        return new_pos

    def shrink(self, position, bounds, **kwargs):
        r"""Set the particle to the boundary

        This method resets particles that exceed the bounds to the intersection
        of its previous velocity and the boundary. This can be imagined as shrinking
        the previous velocity until the particle is back in the valid search space.
        Let :math:`\sigma_{i, t, d}` be the :math:`d` th shrinking value of the
        :math:`i` th particle at the time :math:`t` and :math:`v_{i, t}` the velocity
        of the :math:`i` th particle at the time :math:`t`. Then the new position is
        computed by the following equation:

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
        if self.memory is None:
            new_pos = position
            self.memory = new_pos
        else:
            lb, ub = bounds
            lower_than_bound, greater_than_bound = self._out_of_bounds(
                position, bounds
            )
            velocity = position - self.memory
            # Create a coefficient matrix
            sigma = np.tile(1.0, position.shape)
            sigma[lower_than_bound] = (
                lb[lower_than_bound[1]] - self.memory[lower_than_bound]
            ) / velocity[lower_than_bound]
            sigma[greater_than_bound] = (
                ub[greater_than_bound[1]] - self.memory[greater_than_bound]
            ) / velocity[greater_than_bound]
            min_sigma = np.amin(sigma, axis=1)
            new_pos = position
            new_pos[lower_than_bound[0]] = (
                self.memory[lower_than_bound[0]]
                + np.multiply(
                    min_sigma[lower_than_bound[0]],
                    velocity[lower_than_bound[0]].T,
                ).T
            )
            new_pos[greater_than_bound[0]] = (
                self.memory[greater_than_bound[0]]
                + np.multiply(
                    min_sigma[greater_than_bound[0]],
                    velocity[greater_than_bound[0]].T,
                ).T
            )
            self.memory = new_pos
        return new_pos

    def random(self, position, bounds, **kwargs):
        """Set position to random location

        This method resets particles that exeed the bounds to a random position
        inside the boundary conditions.
        """
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(
            position, bounds
        )
        # Set indices that are greater than bounds
        new_pos = position
        new_pos[greater_than_bound[0]] = np.array(
            [
                np.array([u - l for u, l in zip(ub, lb)])
                * np.random.random_sample((position.shape[1],))
                + lb
            ]
        )
        new_pos[lower_than_bound[0]] = np.array(
            [
                np.array([u - l for u, l in zip(ub, lb)])
                * np.random.random_sample((position.shape[1],))
                + lb
            ]
        )
        return new_pos

    def intermediate(self, position, bounds, **kwargs):
        r"""Set the particle to an intermediate position

        This method resets particles that exceed the bounds to an intermediate
        position between the boundary and their earlier position. Namely, it changes
        the coordinate of the out-of-bounds axis to the middle value between the
        previous position and the boundary of the axis.
        The following equation describes this strategy:

        .. math::

            x_{i, t, d} = \begin{cases}
                                \frac{1}{2} \left (x_{i, t-1, d} + lb_d \right) & \quad \text{if }x_{i, t, d} < lb_d \\
                                \frac{1}{2} \left (x_{i, t-1, d} + ub_d \right) & \quad \text{if }x_{i, t, d} > ub_d \\
                                x_{i, t, d} & \quad \text{otherwise}
                          \end{cases}

        """
        if self.memory is None:
            new_pos = position
            self.memory = new_pos
        else:
            lb, ub = bounds
            lower_than_bound, greater_than_bound = self._out_of_bounds(
                position, bounds
            )
            new_pos = position
            new_pos[lower_than_bound] = 0.5 * (
                self.memory[lower_than_bound] + lb[lower_than_bound[1]]
            )
            new_pos[greater_than_bound] = 0.5 * (
                self.memory[greater_than_bound] + ub[greater_than_bound[1]]
            )
            self.memory = new_pos
        return new_pos

    def periodic(self, position, bounds, **kwargs):
        r"""Sets the particles a periodic fashion

        This method resets the particles that exeed the bounds by using the
        modulo function to cut down the position. This creates a virtual,
        periodic plane which is tiled with the search space.
        The following equation describtes this strategy:

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
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(
            position, bounds
        )
        bound_d = np.tile(
            np.abs(np.array(ub) - np.array(lb)), (position.shape[0], 1)
        )
        ub = np.tile(ub, (position.shape[0], 1))
        lb = np.tile(lb, (position.shape[0], 1))
        new_pos = position
        if lower_than_bound[0].size != 0 and lower_than_bound[1].size != 0:
            new_pos[lower_than_bound] = ub[lower_than_bound] - np.mod(
                (lb[lower_than_bound] - new_pos[lower_than_bound]),
                bound_d[lower_than_bound],
            )
        if greater_than_bound[0].size != 0 and greater_than_bound[1].size != 0:
            new_pos[greater_than_bound] = lb[greater_than_bound] + np.mod(
                (new_pos[greater_than_bound] - ub[greater_than_bound]),
                bound_d[greater_than_bound],
            )
        return new_pos


class VelocityHandler(HandlerMixin):
    def __init__(self, strategy):
        """A VelocityHandler class

        This class offers a way to handle velocities. It contains
        methods to repair the velocities of particles that exceeded the
        defined boundaries. Following strategies are available for the handling:

        * Unmodified:
            Returns the unmodified velocites.
        * Adjust
            Returns the velocity that is adjusted to be the distance between the current
            and the previous position.
        * Invert
            Inverts and shrinks the velocity by the factor :code:`-z`.
        * Zero
            Sets the velocity of out-of-bounds particles to zero.

        """
        self.strategy = strategy
        self.strategies = self._get_all_strategies()
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.memory = None

    def __call__(self, velocity, clamp, **kwargs):
        """Apply the selected strategy to the velocity-matrix given the bounds

        Parameters
        ----------
        velocity : numpy.ndarray
            The swarm position to be handled
        clamp : tuple of numpy.ndarray or list
            a tuple of size 2 where the first entry is the minimum clamp while
            the second entry is the maximum clamp. Each array must be of shape
            :code:`(dimensions,)`
        kwargs : dict

        Returns
        -------
        numpy.ndarray
            the adjusted positions of the swarm
        """
        try:
            new_position = self.strategies[self.strategy](
                velocity, clamp, **kwargs
            )
        except KeyError:
            message = "Unrecognized strategy: {}. Choose one among: " + str(
                [strat for strat in self.strategies.keys()]
            )
            self.rep.logger.exception(message.format(self.strategy))
            raise
        else:
            return new_position

    def __apply_clamp(self, velocity, clamp):
        """Helper method to apply a clamp to a velocity vector"""
        clamped_vel = velocity
        min_velocity, max_velocity = clamp
        lower_than_clamp = clamped_vel <= min_velocity
        greater_than_clamp = clamped_vel >= max_velocity
        clamped_vel = np.where(lower_than_clamp, min_velocity, clamped_vel)
        clamped_vel = np.where(greater_than_clamp, max_velocity, clamped_vel)
        return clamped_vel

    def unmodified(self, velocity, clamp=None, **kwargs):
        """Leaves the velocity unchanged"""
        if clamp is None:
            new_vel = velocity
        else:
            if clamp is not None:
                new_vel = self.__apply_clamp(velocity, clamp)
        return new_vel

    def adjust(self, velocity, clamp=None, **kwargs):
        r"""Adjust the velocity to the new position

        The velocity is adjusted such that the following equation holds:

        .. math::

                \mathbf{v_{i,t}} = \mathbf{x_{i,t}} - \mathbf{x_{i,t-1}}

        .. note::
            This method should only be used in combination with a position handling
            operation.

        """
        try:
            if self.memory is None:
                new_vel = velocity
                self.memory = kwargs["position"]
            else:
                new_vel = kwargs["position"] - self.memory
                self.memory = kwargs["position"]
                if clamp is not None:
                    new_vel = self.__apply_clamp(new_vel, clamp)
        except KeyError:
            self.rep.logger.exception("Keyword 'position' missing")
            raise
        else:
            return new_vel

    def invert(self, velocity, clamp=None, **kwargs):
        r"""Invert the velocity if the particle is out of bounds

        The velocity is inverted and shrinked. The shrinking is determined by the
        kwarg :code:`z`. The default shrinking factor is :code:`0.5`. For all
        velocities whose particles are out of bounds the following equation is
        applied:

        .. math::

            \mathbf{v_{i,t}} = -z\mathbf{v_{i,t}}
        """
        try:
            # Default for the shrinking factor
            if "z" not in kwargs:
                z = 0.5
            else:
                z = kwargs["z"]
            lower_than_bound, greater_than_bound = self._out_of_bounds(
                kwargs["position"], kwargs["bounds"]
            )
            new_vel = velocity
            new_vel[lower_than_bound[0]] = (-z) * new_vel[lower_than_bound[0]]
            new_vel[greater_than_bound[0]] = (-z) * new_vel[
                greater_than_bound[0]
            ]
            if clamp is not None:
                new_vel = self.__apply_clamp(new_vel, clamp)
        except KeyError:
            self.rep.logger.exception("Keyword 'position' or 'bounds' missing")
            raise
        else:
            return new_vel

    def zero(self, velocity, clamp=None, **kwargs):
        """Set velocity to zero if the particle is out of bounds"""
        try:
            lower_than_bound, greater_than_bound = self._out_of_bounds(
                kwargs["position"], kwargs["bounds"]
            )
            new_vel = velocity
            new_vel[lower_than_bound[0]] = np.zeros(velocity.shape[1])
            new_vel[greater_than_bound[0]] = np.zeros(velocity.shape[1])
        except KeyError:
            self.rep.logger.exception("Keyword 'position' or 'bounds' missing")
            raise
        else:
            return new_vel


class OptionsHandler(HandlerMixin):
    def __init__(self, strategy):
        """An OptionsHandler class

        This class offers a way to handle options. It contains
        methods to vary the options at runtime.
        Following strategies are available for the handling:

        * exp_decay:
            Decreases the parameter exponentially between limits.

        * lin_variation:
            Decreases/increases the parameter linearly between limits.

        * random:
            takes a uniform random value between (0.5,1)

        * nonlin_mod:
            Decreases/increases the parameter between limits according to a nonlinear modulation index .

        The OptionsHandler can be called as a function to use the strategy
        that is passed at initialization to account for time-varying coefficients. An example
        for the usage:

        .. code-block :: python

            from pyswarms.backend import operators as op
            from pyswarms.backend.handlers import OptionsHandler


            oh = OptionsHandler(strategy={ "w":"exp_decay", "c1":"nonlin_mod","c2":"lin_variation"})

            for i in range(iters):
                # some initial stuff
                new_options = oh(default_options, iternow=i, itermax=iters, end_opts={"c1":0.5, "c2":2.5, "w":0.4})
                # more updates using new_options

        By passing the handler, the :func:`compute_position()` function now has
        the ability to reset the particles by calling the :code:`BoundaryHandler`
        inside.

        Attributes
        ----------
        strategy : str
            The strategy to use. To see all available strategies,
            call :code:`OptionsHandler.strategies`
        """
        self.strategy = strategy
        self.strategies = self._get_all_strategies()
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def __call__(self, start_opts, **kwargs):
        try:
            if not self.strategy:
                return start_opts
            return_opts = copy(start_opts)
            for opt in start_opts:
                if opt in self.strategy:
                    return_opts[opt] = self.strategies[self.strategy[opt]](
                        start_opts, opt, **kwargs
                    )
        except KeyError:
            message = "Unrecognized strategy: {}. Choose one among: " + str(
                [strat for strat in self.strategies.keys()]
            )
            self.rep.logger.exception(message.format(self.strategy))
            raise
        else:
            return return_opts

    def exp_decay(self, start_opts, opt, **kwargs):
        """Exponentially decreasing between start and end

        Ref: Li, H.-R., & Gao, Y.-L. (2009). Particle Swarm Optimization Algorithm with Exponent
        Decreasing Inertia Weight and Stochastic Mutation. 2009 Second International Conference
        on Information and Computing Science. doi:10.1109/icic.2009.24
        """

        try:
            # default values from reference paper
            if "d1" not in kwargs:
                d1 = 0.2
            else:
                d1 = kwargs["d1"]
            if "d2" not in kwargs:
                d2 = 7
            else:
                d2 = kwargs["d2"]

            end_opts = {
                "w": 0.4,
                "c1": 0.8 * start_opts["c1"],
                "c2": 1 * start_opts["c2"],
            }
            if "end_opts" in kwargs:
                if opt in kwargs["end_opts"]:
                    end_opts[opt] = kwargs["end_opts"][opt]
            start = start_opts[opt]
            end = end_opts[opt]
            new_val = (start - end - d1) * math.exp(
                1 / (1 + d2 * kwargs["iternow"] / kwargs["itermax"])
            )
        except KeyError:
            self.rep.logger.exception("Keyword 'itermax' or 'iternow' missing")
            raise
        else:
            return new_val

    def lin_variation(self, start_opts, opt, **kwargs):
        """
        Linearly decreasing/increasing between start and end

        Ref: Shi Y, Eberhart R.: Empirical study of particle swarm optimization Proc
        of Congress on Computational Intelligence,Washington DC ,USA
        1999, pp.1945 - 1950.
        """

        try:
            end_opts = {
                "w": 0.4,
                "c1": 0.8 * start_opts["c1"],
                "c2": 1 * start_opts["c2"],
            }
            if "end_opts" in kwargs:
                if opt in kwargs["end_opts"]:
                    end_opts[opt] = kwargs["end_opts"][opt]
            start = start_opts[opt]
            end = end_opts[opt]
            new_val = (
                end
                + (start - end)
                * (kwargs["itermax"] - kwargs["iternow"])
                / kwargs["itermax"]
            )
        except KeyError:
            self.rep.logger.exception("Keyword 'itermax' or 'iternow' missing")
            raise
        else:
            return new_val

    def random(self, start_opts, opt, **kwargs):
        """Random value between start option and end option

        Reference: ] R.C. Eberhart, Y.H. Shi, Tracking and optimizing dynamic systems with particle
        swarms, in: Congress on Evolutionary Computation, Korea, 2001
        """
        start = start_opts[opt]
        if opt in kwargs["end_opts"]:
            end = kwargs["end_opts"][opt]
        else:
            end = start + 1
        return start + (end - start) * np.random.rand()

    def nonlin_mod(self, start_opts, opt, **kwargs):
        """Non linear decreasing/increasing with modulation index

        Reference:  A. Chatterjee, P. Siarry, Nonlinear inertia weight variation for dynamic adaption
        in particle swarm optimization, Computer and Operations Research 33 (2006)
        859–871, March 2006
        """

        try:
            if "n" not in kwargs:
                n = 1.2
            else:
                n = kwargs["n"]

            end_opts = {
                "w": 0.4,
                "c1": 0.8 * start_opts["c1"],
                "c2": 1 * start_opts["c2"],
            }
            if "end_opts" in kwargs:
                if opt in kwargs["end_opts"]:
                    end_opts[opt] = kwargs["end_opts"][opt]

            start = start_opts[opt]
            end = end_opts[opt]
            new_val = end + (start - end) * (
                (kwargs["itermax"] - kwargs["iternow"]) ** n
                / kwargs["itermax"] ** n
            )
        except KeyError:
            self.rep.logger.exception("Keyword 'itermax' or 'iternow' missing")
            raise
        else:
            return new_val
