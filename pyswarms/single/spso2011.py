# -*- coding: utf-8 -*-

r"""
SPSO-2011 optimizer.

This module contains an implementation skeleton for Standard Particle
Swarm Optimisation 2011.
"""

import logging

import numpy as np

from ..base import SwarmOptimizer

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

    def _validate_options(self):
        """Validate SPSO-2011-specific options."""
        required_options = ("w", "c1", "c2", "k")

        for option in required_options:
            if option not in self.options:
                raise KeyError("Missing SPSO-2011 option: '{}'".format(option))

        if self.options["k"] <= 0:
            raise ValueError("SPSO-2011 option 'k' must be positive")

    def optimize(self, objective_func, iters, n_processes=None, verbose=True, **kwargs):
        """Run SPSO-2011 optimization.

        This method is not implemented yet.
        """
        raise NotImplementedError("SPSO-2011 optimize loop is not implemented yet.")
