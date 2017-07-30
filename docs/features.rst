========
Features
========



Single-Objective Optimizers
---------------------------

These are standard optimization techniques that aims to find the optima of a single objective function.

Continuous 
~~~~~~~~~~

Single-objective optimization where the search space is continuous.

* :mod:`pyswarms.single.gb` - global-best Particle Swarm Optimization algorithm with a star-topology. Every particle compares itself with the best-performing particle in the swarm.

* :mod:`pyswarms.single.lb` - local-best Particle Swarm Optimization algorithm with a ring-topology. Every particle compares itself only with its nearest-neighbours as computed by a distance metric.


Utilities
---------

Test Functions
~~~~~~~~~~~~~~

These functions can be used as benchmark tests for assessing the performance of the optimization
algorithm.

* :mod:`pyswarms.utils.functions.single_obj` - single-objective test functions
