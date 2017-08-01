========
Features
========

There are two ways in which optimizers are implemented in PySwarms. The first involves
quick-and-easy implementations of classic PSO algorithms. Here, the topologies (or the way
a swarm behaves) is hardcoded in the source code. This is useful for fast implementations that
doesn't need prior set-up.

The second involves a set of experimental classes where topology is not defined. Instead, one
should create an object that inherits from a :code:`Topology` class, and pass it as a parameter
in the experimental PSO classes. There are some topologies that are already implemented, but it's also possible
to define a custom-made one. This is perfect for researchers who wanted to try out various swarm
behaviours and movements.

Single-Objective Optimizers
---------------------------

These are standard optimization techniques that aims to find the optima of a single objective function.

Continuous 
~~~~~~~~~~

Single-objective optimization where the search-space is continuous. Perfect for optimizing various
functions.

* :mod:`pyswarms.single.gb` - classic global-best Particle Swarm Optimization algorithm with a star-topology. Every particle compares itself with the best-performing particle in the swarm.

* :mod:`pyswarms.single.lb` - classic local-best Particle Swarm Optimization algorithm with a ring-topology. Every particle compares itself only with its nearest-neighbours as computed by a distance metric.

* :mod:`pyswarms.single.exp` - experimental Particle Swarm Optimization algorithm. 

Discrete 
~~~~~~~~

Single-objective optimization where the search-space is discrete. Useful for job-scheduling, traveling
salesman, or any other sequence-based problems.

* :mod:`pyswarms.discrete.bn` - classic binary Particle Swarm Optimization algorithm without mutation. Uses a ring topology to choose its neighbours (but can be set to global).


Utilities
---------

Test Functions
~~~~~~~~~~~~~~

These functions can be used as benchmark tests for assessing the performance of the optimization
algorithm.

* :mod:`pyswarms.utils.functions.single_obj` - single-objective test functions
