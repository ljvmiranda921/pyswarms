# -*- coding: utf-8 -*-

r"""
A general Particle Swarm Optimization (general PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Uses a user specified
topology.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = m * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.backend.topology import Pyramid
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters and topology
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    my_topology = Pyramid(static=False)

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GeneralOptimizerPSO(n_particles=10, dimensions=2,
                                        options=options, topology=my_topology)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere_func, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""
# Import from stdlib
import logging

# Import modules
import numpy as np

# Import from package
from ..base import SwarmOptimizer
from ..backend.operators import compute_pbest
from ..backend.topology import Topology, Ring, Random, VonNeumann
from ..utils.console_utils import cli_print, end_report


class GeneralOptimizerPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        topology,
        bounds=None,
        velocity_clamp=None,
        center=1.00,
        ftol=-np.inf,
        init_pos=None,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}` or :code:`{'c1', 'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                if used with the :code:`Ring`, :code:`VonNeumann` or :code:`Random` topology the additional
                parameter k must be included
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                if used with the :code:`Ring` topology the additional
                parameters k and p must be included
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
                if used with the :code:`VonNeumann` topology the additional
                parameters p and r must be included
                * r: int
                    the range of the VonNeumann topology.
                    This is used to determine the number of
                    neighbours in the topology.
        topology : pyswarms.backend.topology.Topology
            a :code:`Topology` object that defines the topology to use
            in the optimization process. The currently available topologies
            are:
                * Star
                    All particles are connected
                * Ring (static and dynamic)
                    Particles are connected to the k nearest neighbours
                * VonNeumann
                    Particles are connected in a VonNeumann topology
                * Pyramid (static and dynamic)
                    Particles are connected in N-dimensional simplices
                * Random (static and dynamic)
                    Particles are connected to k random particles
                Static variants of the topologies remain with the same neighbours
                over the course of the optimization. Dynamic variants calculate
                new neighbours every time step.
        bounds : tuple of :code:`np.ndarray` (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dimensions,)`.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence
        """
        super(GeneralOptimizerPSO, self).__init__(
            n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            init_pos=init_pos,
        )

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()
        # Initialize the topology and check for type
        if not isinstance(topology, Topology):
            raise TypeError("Parameter `topology` must be a Topology object")
        else:
            self.top = topology

        # Case for the Ring topology
        if isinstance(topology, (Ring, VonNeumann)):
            # Assign p-value as attributes
            self.p = options["p"]
            # Exceptions for the p value
            if "p" not in self.options:
                raise KeyError("Missing p in options")
            if self.p not in [1, 2]:
                raise ValueError(
                    "p-value should either be 1 (for L1/Minkowski) "
                    "or 2 (for L2/Euclidean)."
                )

        # Case for Random, VonNeumann and Ring topologies
        if isinstance(topology, (Random, Ring, VonNeumann)):
            if not isinstance(topology, VonNeumann):
                self.k = options["k"]
                if not isinstance(self.k, int):
                    raise ValueError(
                        "No. of neighbors must be an integer between"
                        "0 and no. of particles."
                    )
                if not 0 <= self.k <= self.n_particles - 1:
                    raise ValueError(
                        "No. of neighbors must be between 0 and no. "
                        "of particles."
                    )
                if "k" not in self.options:
                    raise KeyError("Missing k in options")
            else:
                # Assign range r as attribute
                self.r = options["r"]
                if not isinstance(self.r, int):
                    raise ValueError("The range must be a positive integer")
                if (
                    self.r <= 0
                    or not 0
                           <= VonNeumann.delannoy(self.swarm.dimensions, self.r)
                           <= self.n_particles - 1
                ):
                    raise ValueError(
                        "The range must be set such that the computed"
                        "Delannoy number (number of neighbours) is"
                        "between 0 and the no. of particles."
                    )

    def optimize(self, objective_func, iters, print_step=1, verbose=1, **kwargs):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        print_step : int (default is 1)
            amount of steps for printing into console.
        verbose : int  (default is 1)
            verbosity setting.
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """

        cli_print("Arguments Passed to Objective Function: {}".format(kwargs),
                  verbose, 2, logger=self.logger)

        for i in range(iters):
            # Compute cost for current position and personal best
            self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
            self.swarm.pbest_cost = objective_func(self.swarm.pbest_pos, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm
            )
            best_cost_yet_found = self.swarm.best_cost
            # If the topology is a ring topology just use the local minimum
            if isinstance(self.top, Ring) and not isinstance(self.top, VonNeumann):
                # Update gbest from neighborhood
                self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                    self.swarm, self.p, self.k
                )
            # If the topology is a VonNeumann topology pass the neighbour and range attribute to compute_gbest()
            if isinstance(self.top, VonNeumann):
                # Update gbest from neighborhood
                self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                    self.swarm, self.p, self.r
                )
            # If the topology is a random topology pass the neighbor attribute to compute_gbest()
            elif isinstance(self.top, Random):
                # Get minima of pbest and check if it's less than gbest
                if np.min(self.swarm.pbest_cost) < self.swarm.best_cost:
                    self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                        self.swarm, self.k
                    )
            else:
                # Get minima of pbest and check if it's less than gbest
                if np.min(self.swarm.pbest_cost) < self.swarm.best_cost:
                    self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                        self.swarm
                    )
            # Print to console
            if i % print_step == 0:
                cli_print(
                    "Iteration {}/{}, cost: {}".format(i + 1, iters, self.swarm.best_cost),
                    verbose,
                    2,
                    logger=self.logger
                )
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            if (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            ):
                break
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds
            )
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.best_pos.copy()
        # Write report in log and return final cost and position
        end_report(
            final_best_cost, final_best_pos, verbose, logger=self.logger
        )
        return(final_best_cost, final_best_pos)
