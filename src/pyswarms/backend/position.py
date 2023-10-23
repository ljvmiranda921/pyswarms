"""
Class for handling positions during optimization. This involves
    - Generating initial swarm positions
    - Updating swarm positions
    - Calling the boundary handler
"""

from typing import Optional

import numpy as np

from pyswarms.backend.handlers import BoundaryHandler, BoundaryStrategy
from pyswarms.backend.swarms import Swarm
from pyswarms.utils.types import Bound, Bounds, Position


class PositionUpdater:
    bh: Optional[BoundaryHandler] = None

    def __init__(self, bounds: Optional[Bounds] = None, bh: BoundaryHandler | BoundaryStrategy = "periodic"):
        """
        Parameters
        ----------
        bounds : Optional[Bounds], optional
            Position bounds, by default None
        bh : BoundaryHandler | BoundaryStrategy, optional
            A BoundaryHandler instance or a strategy, by default "periodic"
        """
        self.bounds = bounds

        if self.bounds is None:
            return

        if isinstance(bh, str):
            self.bh = BoundaryHandler.factory(bh, self.bounds)
        else:
            self.bh = bh
            bh.set_bounds(self.bounds)


    def compute(self, swarm: Swarm):
        """Update the position matrix

        This method updates the position matrix given the current position and the
        velocity. If bounded, the positions are handled by a
        :code:`BoundaryHandler` instance

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.swarms.backend import Swarm, VelocityHandler

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_bh = BoundaryHandler(strategy="intermediate")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.position = compute_position(my_swarm, bounds, my_bh)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of numpy.ndarray or list, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler object with a specified handling strategy
            For further information see :mod:`pyswarms.backend.handlers`.

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        temp_position: Position = swarm.position.copy() + swarm.velocity

        if self.bh is not None:
            temp_position = self.bh(temp_position)

        position = temp_position

        return position

    def generate_position(
        self,
        n_particles: int,
        dimensions: int,
        center: float | Position = 1.00,
        init_pos: Optional[Position] = None,
    ) -> Position:
        """Generate a swarm

        Parameters
        ----------
        n_particles : int
            number of particles to be generated in the swarm.
        dimensions: int
            number of dimensions to be generated in the swarm
        center : numpy.ndarray or float, optional
            controls the mean or center whenever the swarm is generated randomly.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
            Default is :code:`None`.

        Returns
        -------
        numpy.ndarray
            swarm matrix of shape (n_particles, n_dimensions)

        Raises
        ------
        ValueError
            When the shapes and values of bounds, dimensions, and init_pos
            are inconsistent.
        TypeError
            When the argument passed to bounds is not an iterable.
        """
        if init_pos is not None:
            if self.bounds is not None:
                if not (np.all(self.bounds[0] <= init_pos) and np.all(init_pos <= self.bounds[1])):
                    raise ValueError("User-defined init_pos is out of bounds.")
            pos = self._check_init_pos(init_pos, n_particles, dimensions)
        else:
            if self.bounds is not None:
                lb, ub = self.bounds
                min_bounds = self._check_bound(lb, dimensions)
                max_bounds = self._check_bound(ub, dimensions)
                pos = center * np.random.uniform(low=min_bounds, high=max_bounds, size=(n_particles, dimensions))
            else:
                pos = center * np.random.uniform(low=0.0, high=1.0, size=(n_particles, dimensions))

        return pos

    def _check_bound(self, bound: Bound, dim: int):
        """Check if a bound is vector or scalar. In case of a vector, check dimensions and 
        add a new a axis to support further manipulations.
        """
        if isinstance(bound, tuple | list | np.ndarray):
            bound = np.array(bound)
            assert bound.ndim == 1, f"Bound must be shape ({dim},) but got {bound.shape}"
            assert bound.shape[0] == dim, f"Bound must be shape ({dim},) but got {bound.shape}"
            bound = bound[None, :]

        return bound

    def generate_discrete_position(
        self, n_particles: int, dimensions: int, binary: bool = False, init_pos: Optional[Position] = None
    ) -> Position:
        """Generate a discrete swarm

        Parameters
        ----------
        n_particles : int
            number of particles to be generated in the swarm.
        dimensions: int
            number of dimensions to be generated in the swarm.
        binary : bool
            generate a binary matrix. Default is :code:`False`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
            Default is :code:`None`

        Returns
        -------
        numpy.ndarray
            swarm matrix of shape (n_particles, n_dimensions)

        Raises
        ------
        ValueError
            When init_pos during binary=True does not contain two unique values.
            When init_pos has the incorrect number of dimensions
        AssertionError
            When init_pos has an incorrect shape
        """
        if init_pos is not None:
            pos = self._check_init_pos(init_pos, n_particles, dimensions, binary)
        else:
            if binary:
                pos = np.random.randint(2, size=(n_particles, dimensions))  # type: ignore
            else:
                pos = np.random.random_sample(size=(n_particles, dimensions)).argsort(axis=1)

        return pos

    def _check_init_pos(self, init_pos: Position, n_particles: int, dimensions: int, binary: bool = False):
        """Checks the shape and contents of the initial position.
        """
        if binary and len(np.unique(init_pos)) > 2:
            raise ValueError("User-defined init_pos is not binary!")

        if init_pos.ndim == 1:
            assert init_pos.shape[0] == dimensions
            pos = np.repeat([init_pos], n_particles, axis=0)
        elif init_pos.ndim == 2:
            assert init_pos.shape[0] == n_particles
            assert init_pos.shape[1] == dimensions
            pos = init_pos
        else:
            raise ValueError("init_pos must be 1D or 2D")

        return pos
