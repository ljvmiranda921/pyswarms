# -*- coding: utf-8 -*-

r"""
SPSO-2011 optimizer.

This module contains an implementation skeleton for Standard Particle
Swarm Optimisation 2011.
"""

import logging
import multiprocessing as mp

from collections import deque

import numpy as np

from ..backend.operators import compute_objective_function, compute_pbest
from ..base import SwarmOptimizer
from ..utils.reporter import Reporter

LOGGER = logging.getLogger(__name__)


class SPSO2011(SwarmOptimizer):
    """Standard Particle Swarm Optimisation 2011.

    This optimizer is intended to implement SPSO-2011 with adaptive
    random informant topology, rotationally invariant hyperspherical
    sampling, and absorbing-wall boundary handling.
    """

    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        velocity_clamp=None,
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
        seed=None,
    ):
        super(SPSO2011, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )

        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._validate_options()
        self.rep = Reporter(logger=logging.getLogger(__name__))
        self.name = __name__
        self.informants = None

    def _validate_options(self):
        """Validate SPSO-2011-specific options."""
        required_options = ("w", "c1", "c2", "k")

        for option in required_options:
            if option not in self.options:
                raise KeyError("Missing SPSO-2011 option: '{}'".format(option))

        if self.options["k"] <= 0:
            raise ValueError("SPSO-2011 option 'k' must be positive")

    def _initialize_spso2011_velocity(self):
        """Initialize velocity according to the SPSO-2011 rule.

        SPSO-2011 uses V0 = (U(bounds) - X0) / 2. This requires explicit
        bounds. If bounds are unavailable, the existing PySwarms-generated
        velocity is retained.
        """
        if self.bounds is None:
            return

        lower_bound, upper_bound = self.bounds
        random_position = self.rng.uniform(
            low=lower_bound,
            high=upper_bound,
            size=self.swarm.position.shape,
        )
        self.swarm.velocity = (random_position - self.swarm.position) / 2.0

    def _generate_informants(self):
        """Generate the SPSO-2011 adaptive random informant topology.

        The convention is: informants[i, j] is True when particle j informs
        particle i (the row-oriented interpretation of the rule is convenient for computing).
        Each particle informs itself, and each particle randomly informs k particles.
        Duplicate choices are allowed by the standard,
        but the boolean matrix naturally stores each link once.
        """
        k = self.options["k"]
        informants = np.eye(self.n_particles, dtype=bool)

        for informer in range(self.n_particles):
            informed_particles = self.rng.integers(
                low=0,
                high=self.n_particles,
                size=k,
            )
            informants[informed_particles, informer] = True

        self.informants = informants

    def _compute_neighbor_best(self):
        """Compute each particle's best informant position and cost."""
        if self.informants is None:
            self._generate_informants()

        best_pos = np.empty_like(self.swarm.position)
        best_cost = np.empty(self.n_particles)

        for particle in range(self.n_particles):
            neighbor_idx = np.where(self.informants[particle])[0]
            best_neighbor = neighbor_idx[
                np.argmin(self.swarm.pbest_cost[neighbor_idx])
            ]
            best_pos[particle] = self.swarm.pbest_pos[best_neighbor]
            best_cost[particle] = self.swarm.pbest_cost[best_neighbor]

        return best_pos, best_cost

    def _sample_hypersphere(self, centers, radii):
        """Sample points uniformly inside hyperspheres.

        Parameters
        ----------
        centers : numpy.ndarray
            Hypersphere centers with shape (n_particles, dimensions).
        radii : numpy.ndarray
            Hypersphere radii with shape (n_particles,).
        """
        directions = self.rng.normal(size=centers.shape)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(
            directions,
            norms,
            out=np.zeros_like(directions),
            where=norms > 0,
        )

        scales = self.rng.random((self.n_particles, 1)) ** (
            1.0 / self.dimensions
        )
        return centers + directions * radii[:, np.newaxis] * scales

    def _compute_velocity(self, neighbor_best_pos):
        """Compute the SPSO-2011 rotationally invariant velocity update."""
        position = self.swarm.position
        pbest_pos = self.swarm.pbest_pos
        velocity = self.swarm.velocity

        c1 = self.swarm.options["c1"]
        c2 = self.swarm.options["c2"]
        w = self.swarm.options["w"]

        u1 = self.rng.random(position.shape)
        u2 = self.rng.random(position.shape)

        personal_attractor = position + c1 * u1 * (pbest_pos - position)
        neighbor_attractor = position + c2 * u2 * (neighbor_best_pos - position)
        gravity_center = (
            position + personal_attractor + neighbor_attractor
        ) / 3.0
        radius = np.linalg.norm(gravity_center - position, axis=1)
        hypersphere_point = self._sample_hypersphere(gravity_center, radius)

        return w * velocity + hypersphere_point - position

    def _apply_absorbing_bounds(self):
        """Apply SPSO-2011 absorbing-wall boundary handling."""
        if self.bounds is None:
            return

        lower_bound, upper_bound = self.bounds
        below_bounds = self.swarm.position < lower_bound
        above_bounds = self.swarm.position > upper_bound
        out_of_bounds = below_bounds | above_bounds

        self.swarm.position = np.minimum(
            np.maximum(self.swarm.position, lower_bound),
            upper_bound,
        )
        self.swarm.velocity[out_of_bounds] = 0.0

    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
    ):
        """Run SPSO-2011 optimization.

        Parameters
        ----------
        objective_func : callable
            Objective function to be evaluated.
        iters : int
            Number of iterations.
        n_processes : int, optional
            Number of processes to use for parallel particle evaluation.
        verbose : bool, optional
            Enable or disable logs and progress bar.
        kwargs : dict
            Additional keyword arguments passed to the objective function.

        Returns
        -------
        tuple
            Best cost and best position found by the swarm.
        """
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )

        pool = None if n_processes is None else mp.Pool(n_processes)

        self._initialize_spso2011_velocity()
        self._generate_informants()
        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)

        ftol_history = deque(maxlen=self.ftol_iter)

        try:
            for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
                self.swarm.current_cost = compute_objective_function(
                    self.swarm, objective_func, pool=pool, **kwargs
                )
                self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                    self.swarm
                )

                best_cost_yet_found = self.swarm.best_cost
                neighbor_best_pos, neighbor_best_cost = self._compute_neighbor_best()

                best_particle = self.swarm.pbest_cost.argmin()
                self.swarm.best_pos = self.swarm.pbest_pos[best_particle].copy()
                self.swarm.best_cost = self.swarm.pbest_cost[best_particle].copy()

                if self.swarm.best_cost >= best_cost_yet_found:
                    self._generate_informants()

                if verbose:
                    self.rep.hook(best_cost=self.swarm.best_cost)

                hist = self.ToHistory(
                    best_cost=self.swarm.best_cost,
                    mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                    mean_neighbor_cost=np.mean(neighbor_best_cost),
                    position=self.swarm.position,
                    velocity=self.swarm.velocity,
                )
                self._populate_history(hist)

                relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
                delta = (
                    np.abs(self.swarm.best_cost - best_cost_yet_found)
                    < relative_measure
                )
                if i < self.ftol_iter:
                    ftol_history.append(delta)
                else:
                    ftol_history.append(delta)
                    if all(ftol_history):
                        break

                self.swarm.velocity = self._compute_velocity(neighbor_best_pos)
                self.swarm.position = self.swarm.position + self.swarm.velocity
                self._apply_absorbing_bounds()
        finally:
            if n_processes is not None:
                pool.close()

        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()

        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )

        return final_best_cost, final_best_pos
