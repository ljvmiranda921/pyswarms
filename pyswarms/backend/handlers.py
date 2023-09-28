r"""
Handlers

This module provides Handler classes for the position, velocity and time varying acceleration coefficients
of particles. Particles that do not stay inside these boundary conditions have to
be handled by either adjusting their position after they left the bounded
search space or adjusting their velocity when it would position them outside
the search space. In particular, this approach is important if the optimium of
a function is near the boundaries.
For the following documentation let :math:`x_{i, t, d} \ ` be the :math:`d` th
coordinate of the particle :math:`i` 's position vector at the time :math:`t`,
:math:`lb` the vector of the lower boundaries and :math:`ub` the vector of the
upper boundaries.
The :class:`OptionsHandler` class provide methods which allow faster and better convergence by varying
the options :math:`w, c_{1}, c_{2}` with various strategies.

The algorithms in the :class:`BoundaryHandler` and :class:`VelocityHandler` classes are adapted from [SH2010]

[SH2010] Sabine Helwig, "Particle Swarms for Constrained Optimization",
PhD thesis, Friedrich-Alexander Universität Erlangen-Nürnberg, 2010.
"""

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pyswarms.utils.types import (
    BoundaryStrategy,
    Bounds,
    Clamp,
    OptionsStrategy,
    Position,
    SwarmOption,
    Velocity,
    VelocityStrategy,
)


class HandlerMixin(object):
    """A HandlerMixing class

    This class offers some basic functionality for the Handlers.
    """

    def _out_of_bounds(self, position: npt.NDArray[Any], bounds: Bounds):
        """Helper method to find indices of out-of-bound positions

        This method finds the indices of the particles that are out-of-bound.
        """
        lb, ub = bounds
        greater_than_bound = np.nonzero(position > ub)
        lower_than_bound = np.nonzero(position < lb)

        return (lower_than_bound, greater_than_bound)

    def _get_all_strategies(self):
        """Helper method to automatically generate a dict of strategies"""
        return {k: v for k, v in inspect.getmembers(self, predicate=inspect.isroutine) if not k.startswith(("__", "_"))}


class BoundaryHandler(HandlerMixin, ABC):
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
    """

    memory: Optional[Position] = None

    @abstractmethod
    def __call__(self, position: Position, bounds: Bounds) -> Position:
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
        ...

    @staticmethod
    def factory(strategy: BoundaryStrategy):
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

            Parameters
            ----------
            strategy : BoundaryStrategy
        """
        if strategy == "intermediate":
            return IntermediateHandler()
        elif strategy == "nearest":
            return NearestHandler()
        elif strategy == "periodic":
            return PeriodicHandler()
        elif strategy == "random":
            return RandomBoundaryHandler()
        elif strategy == "reflective":
            return ReflectiveHandler()
        elif strategy == "shrink":
            return ShrinkHandler()

        raise ValueError(
            f"""Strategy {strategy} does not match any of
            ["nearest", "random", "shrink", "reflective", "intermediate", "periodic"]"""
        )


class NearestHandler(BoundaryHandler):
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

    def __call__(self, position: Position, bounds: Bounds):
        lb, ub = bounds
        bool_greater = position > ub
        bool_lower = position < lb
        new_pos = np.where(bool_lower, lb, position)
        new_pos = np.where(bool_greater, ub, new_pos)
        return new_pos


class ReflectiveHandler(BoundaryHandler):
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

    def __call__(self, position: Position, bounds: Bounds):
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(position, bounds)
        new_pos = position
        while lower_than_bound[0].size != 0 or greater_than_bound[0].size != 0:
            if lower_than_bound[0].size > 0:
                new_pos[lower_than_bound] = 2 * lb[lower_than_bound[1]] - new_pos[lower_than_bound]
            if greater_than_bound[0].size > 0:
                new_pos[greater_than_bound] = 2 * ub[greater_than_bound[1]] - new_pos[greater_than_bound]
            lower_than_bound, greater_than_bound = self._out_of_bounds(new_pos, bounds)

        return new_pos


class ShrinkHandler(BoundaryHandler):
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

    def __call__(self, position: Position, bounds: Bounds):
        if self.memory is None:
            new_pos = position
            self.memory = new_pos
        else:
            lb, ub = bounds
            lower_than_bound, greater_than_bound = self._out_of_bounds(position, bounds)
            velocity = position - self.memory
            # Create a coefficient matrix
            sigma = np.tile(1.0, position.shape)
            sigma[lower_than_bound] = (lb[lower_than_bound[1]] - self.memory[lower_than_bound]) / velocity[
                lower_than_bound
            ]
            sigma[greater_than_bound] = (ub[greater_than_bound[1]] - self.memory[greater_than_bound]) / velocity[
                greater_than_bound
            ]
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


class RandomBoundaryHandler(BoundaryHandler):
    """Set position to random location

    This method resets particles that exeed the bounds to a random position
    inside the boundary conditions.
    """

    def __call__(self, position: Position, bounds: Bounds):
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(position, bounds)
        # Set indices that are greater than bounds
        new_pos = position
        new_pos[greater_than_bound[0]] = np.array(
            [np.array([u - l for u, l in zip(ub, lb)]) * np.random.random_sample((position.shape[1],)) + lb]
        )
        new_pos[lower_than_bound[0]] = np.array(
            [np.array([u - l for u, l in zip(ub, lb)]) * np.random.random_sample((position.shape[1],)) + lb]
        )
        return new_pos


class IntermediateHandler(BoundaryHandler):
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

    def __call__(self, position: Position, bounds: Bounds):
        if self.memory is None:
            new_pos = position
            self.memory = new_pos
        else:
            lb, ub = bounds
            lower_than_bound, greater_than_bound = self._out_of_bounds(position, bounds)
            new_pos = position
            new_pos[lower_than_bound] = 0.5 * (self.memory[lower_than_bound] + lb[lower_than_bound[1]])
            new_pos[greater_than_bound] = 0.5 * (self.memory[greater_than_bound] + ub[greater_than_bound[1]])
            self.memory = new_pos
        return new_pos


class PeriodicHandler(BoundaryHandler):
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

    def __call__(self, position: Position, bounds: Bounds):
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(position, bounds)
        bound_d = np.tile(np.abs(np.array(ub) - np.array(lb)), (position.shape[0], 1))
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


class VelocityHandler(HandlerMixin, ABC):
    memory: Optional[Position] = None

    def __init__(self, clamp: Optional[Clamp] = None, bounds: Optional[Bounds] = None):
        """Initialize the VelocityHandler

        Parameters
        ----------
        clamp : Optional[Clamp], optional
            Minimum and maximum velocity values, by default None
        bounds : Optional[Bounds], optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`, by default None
        """
        self.clamp = clamp
        self.bounds = bounds

    @abstractmethod
    def __call__(self, velocity: Velocity, position: Optional[Position]) -> Velocity:
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
        ...

    def _apply_clamp(self, velocity: Velocity):
        """Helper method to apply a clamp to a velocity vector"""
        if self.clamp is None:
            return velocity
        min_velocity, max_velocity = self.clamp
        return np.clip(velocity, min_velocity, max_velocity)

    @staticmethod
    def factory(strategy: VelocityStrategy, clamp: Optional[Clamp] = None, bounds: Optional[Bounds] = None):
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

        Parameters
        ----------
        strategy : VelocityStrategy
        clamp : Optional[Clamp], optional
        bounds : Optional[Bounds], optional
        """
        if strategy == "unmodified":
            return UnmodifiedVelocityHandler(clamp, bounds)
        elif strategy == "adjust":
            return AdjustVelocityHandler(clamp, bounds)
        elif strategy == "invert":
            return InvertVelocityHandler(clamp, bounds)
        elif strategy == "zero":
            return ZeroVelocityHandler(clamp, bounds)

        raise ValueError(f'Strategy {strategy} does not match any of ["unmodified", "adjust", "invert", "zero"]')


class UnmodifiedVelocityHandler(VelocityHandler):
    def __call__(self, velocity: Velocity, position: Optional[Position]) -> Velocity:
        """Leaves the velocity unchanged"""
        return self._apply_clamp(velocity)


class AdjustVelocityHandler(VelocityHandler):
    def __call__(self, velocity: Velocity, position: Optional[Position]):
        r"""Adjust the velocity to the new position

        The velocity is adjusted such that the following equation holds:

        .. math::

                \mathbf{v_{i,t}} = \mathbf{x_{i,t}} - \mathbf{x_{i,t-1}}

        .. note::
            This method should only be used in combination with a position handling
            operation.

        """
        if position is None:
            raise ValueError("Position must not be None")

        if self.memory is None:
            new_vel = velocity
            self.memory = position
        else:
            new_vel = position - self.memory
            self.memory = position
            new_vel = self._apply_clamp(new_vel)

        return new_vel


class InvertVelocityHandler(VelocityHandler):
    def __init__(self, clamp: Optional[Clamp] = None, bounds: Optional[Bounds] = None, z: float = 0.5):
        super().__init__(clamp, bounds)
        self.z = z

    def __call__(self, velocity: Velocity, position: Optional[Position]) -> Velocity:
        r"""Invert the velocity if the particle is out of bounds

        The velocity is inverted and shrinked. The shrinking is determined by the
        kwarg :code:`z`. The default shrinking factor is :code:`0.5`. For all
        velocities whose particles are out of bounds the following equation is
        applied:

        .. math::

            \mathbf{v_{i,t}} = -z\mathbf{v_{i,t}}
        """
        if position is None:
            raise ValueError("Position must not be None")

        if self.bounds is None:
            raise ValueError("Bounds must not be None")

        lower_than_bound, greater_than_bound = self._out_of_bounds(position, self.bounds)
        new_vel = velocity
        new_vel[lower_than_bound[0]] = (-self.z) * new_vel[lower_than_bound[0]]
        new_vel[greater_than_bound[0]] = (-self.z) * new_vel[greater_than_bound[0]]

        new_vel = self._apply_clamp(new_vel)

        return new_vel


class ZeroVelocityHandler(VelocityHandler):
    def __call__(self, velocity: Velocity, position: Optional[Position]):
        """Set velocity to zero if the particle is out of bounds"""
        if position is None:
            raise ValueError("Position must not be None")

        if self.bounds is None:
            raise ValueError("Bounds must not be None")

        lower_than_bound, greater_than_bound = self._out_of_bounds(position, self.bounds)
        new_vel = velocity
        new_vel[lower_than_bound[0]] = np.zeros(velocity.shape[1])
        new_vel[greater_than_bound[0]] = np.zeros(velocity.shape[1])

        return new_vel


class OptionsHandler(ABC):
    """An OptionsHandler class

    This class offers a way to handle options. It contains
    methods to vary the options at runtime.
    Following strategies are available for the handling:

    * exp_decay:
        Decreases the parameter exponentially between limits.

    * lin_variation:
        Decreases/increases the parameter linearly between limits.

    * random:
        takes a uniform random value between limits

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
            # initial operations for global and local best positions
            new_options = oh(default_options, iternow=i, itermax=iters, end_opts={"c1":0.5, "c2":2.5, "w":0.4})
            # more updates using new_options

    .. note::
        As of pyswarms v1.3.0, you will need to create your own optimization loop to change the default ending
        options and other arguments for each strategy in all of the handlers on this page.

    A more comprehensive tutorial is also present `here`_ for interested users.

    .. _here: https://pyswarms.readthedocs.io/en/latest/examples/tutorials/options_handler.html



    Attributes
    ----------
    strategy : str
        The strategy to use. To see all available strategies,
        call :code:`OptionsHandler.strategies`
    """

    end_value: float

    def __init__(self, option: SwarmOption, start_value: float, end_value: Optional[float] = None):
        """Initialise the OptionsHandler

        Parameters
        ----------
        option : SwarmOption
            Which option this handler will manage.
        start_value : float
            Initial value for the option
        end_value : Optional[float], optional
            Final value for the option. If None, it will be computed automatically.
            Defaults:
                :math:`w^{end} = 0.4,
                c^{end}_{1} = 0.8 * c^{start}_{1},
                c^{end}_{2} = c^{start}_{2}`
        """
        self.option = option
        self.start_value = start_value
        self.set_end_option(end_value)

    def set_end_option(self, end_value: Optional[float]):
        if end_value is not None:
            self.end_value = end_value
            return

        if self.option == "c1":
            self.end_value = 0.8 * self.start_value
        elif self.option == "c2":
            self.end_value = self.start_value
        else:
            self.end_value = 0.4

    @abstractmethod
    def __call__(self, iter: int, iter_max: int) -> float:
        """
        Parameters
        ----------
        iter : int
            Current iteration.
        iter_max : int
            Total number of iterations.

        Returns
        -------
        float
            Value of the option at the current iteration.
        """
        ...

    @staticmethod
    def factory(
        strategy: OptionsStrategy,
        option: SwarmOption,
        start_value: float,
        end_value: Optional[float] = None,
        **kwargs: Any,
    ):
        if strategy == "exp_decay":
            return ExpDecayHandler(option, start_value, end_value, **kwargs)
        elif strategy == "lin_variation":
            return LinVariationHandler(option, start_value, end_value, **kwargs)
        elif strategy == "nonlin_mod":
            return NonlinModHandler(option, start_value, end_value, **kwargs)
        elif strategy == "random":
            return RandomHandler(option, start_value, end_value, **kwargs)

        raise ValueError(
            f'Strategy {strategy} does not match any of ["exp_decay", "lin_variation", "nonlin_mod", "random"]'
        )


class ExpDecayHandler(OptionsHandler):
    """Exponentially decreasing between :math:`w_{start}` and :math:`w_{end}`
    The velocity is adjusted such that the following equation holds:

    Defaults: :math:`
        d_{1}=0.2,
        d_{2}=7,
        w^{end} = 0.4,
        c^{end}_{1} = 0.8 * c^{start}_{1},
        c^{end}_{2} = c^{start}_{2}`

    .. math::
            w = (w^{start}-w^{end}-d_{1})exp(\\frac{1}{1+ \\frac{d_{2} * iter}{iter_{max}}})

    Ref: Li, H.-R., & Gao, Y.-L. (2009). Particle Swarm Optimization Algorithm with Exponent
    Decreasing Inertia Weight and Stochastic Mutation. 2009 Second International Conference
    on Information and Computing Science. doi:10.1109/icic.2009.24
    """

    def __init__(
        self, option: SwarmOption, start_value: float, end_value: Optional[float] = None, d1: float = 0.2, d2: float = 7
    ):
        """Initialise the ExpDecayHandler

        Parameters
        ----------
        option : SwarmOption
            Which option this handler will manage.
        start_value : float
            Initial value for the option
        end_value : Optional[float], optional
            Final value for the option. If None, it will be computed automatically.
        d1 : float, optional
            By default 0.2
        d2 : float, optional
            By default 7
        """
        super().__init__(option, start_value, end_value)
        self.d1 = d1
        self.d2 = d2

    def __call__(self, iter_cur: int, iter_max: int):
        return (self.start_value - self.end_value - self.d1) * math.exp(1 / (1 + self.d2 * iter_cur / iter_max))


class LinVariationHandler(OptionsHandler):
    """
    Linearly decreasing/increasing between :math:`w_{start}` and :math:`w_{end}`

    Defaults: :math:`w^{end} = 0.4, c^{end}_{1} = 0.8 * c^{start}_{1}, c^{end}_{2} = c^{start}_{2}`

    .. math::
            w = w^{end}+(w^{start}-w^{end}) \\frac{iter^{max}-iter}{iter^{max}}

    Ref: Xin, Jianbin, Guimin Chen, and Yubao Hai. "A particle swarm optimizer with
    multi-stage linearly-decreasing inertia weight." 2009 International joint conference
    on computational sciences and optimization. Vol. 1. IEEE, 2009.
    """

    def __call__(self, iter_cur: int, iter_max: int):
        return self.start_value + (self.end_value - self.start_value) * iter_cur / iter_max


class RandomHandler(OptionsHandler):
    """Random value between :math:`w^{start}` and :math:`w^{end}`

    .. math::
            w = start + (end-start)*rand(0,1)

    Ref: R.C. Eberhart, Y.H. Shi, Tracking and optimizing dynamic systems with particle
    swarms, in: Congress on Evolutionary Computation, Korea, 2001
    """

    def set_end_option(self, end_value: Optional[float]):
        if end_value is not None:
            self.end_value = end_value
            return

        self.end_value = self.start_value + 1

    def __call__(self, iter_cur: int, iter_max: int):
        return self.start_value + (self.end_value - self.start_value) * np.random.rand()


class NonlinModHandler(OptionsHandler):
    """Non linear decreasing/increasing with modulation index(n).
    The linear strategy can be made to converge faster without compromising
    on exploration with the use of this index which makes the equation non-linear.

    Defaults: :math:`n=1.2`

    .. math::
            w = w^{end}+(w^{start}-w^{end}) \\frac{(iter^{max}-iter)^{n}}{(iter^{max})^{n}}

    Ref:  A. Chatterjee, P. Siarry, Nonlinear inertia weight variation for dynamic adaption
    in particle swarm optimization, Computer and Operations Research 33 (2006)
    859–871, March 2006
    """

    def __init__(self, option: SwarmOption, start_value: float, end_value: Optional[float] = None, n: float = 1.2):
        """Initialise the NonlinModHandler

        Parameters
        ----------
        option : SwarmOption
            Which option this handler will manage.
        start_value : float
            Initial value for the option
        end_value : Optional[float], optional
            Final value for the option. If None, it will be computed automatically.
        n : float > 0, optional
            Larger values make it converge to end_value quicker, by default 1.2
        """
        super().__init__(option, start_value, end_value)
        self.n = n
        assert self.n > 0, "n must be larger than 0"

    def __call__(self, iter_cur: int, iter_max: int) -> float:
        return self.end_value + (self.start_value - self.end_value) * ((iter_max - iter_cur) / iter_max) ** self.n
