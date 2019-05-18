========
Features
========


Single-Objective Optimizers
---------------------------

These are standard optimization techniques for finding the optima of a single
objective function.

Continuous 
~~~~~~~~~~

Single-objective optimization where the search-space is continuous. Perfect
for optimizing various common functions.

* :mod:`pyswarms.single.global_best` - classic global-best Particle Swarm Optimization algorithm with a star-topology. Every particle compares itself with the best-performing particle in the swarm.

* :mod:`pyswarms.single.local_best` - classic local-best Particle Swarm Optimization algorithm with a ring-topology. Every particle compares itself only with its nearest-neighbours as computed by a distance metric.

* :mod:`pyswarms.single.general_optimizer` - alterable but still classic Particle Swarm Optimization algorithm with a custom topology. Every topology in the :mod:`pyswarms.backend` module can be passed as an argument.

Discrete 
~~~~~~~~

Single-objective optimization where the search-space is discrete. Useful for
job-scheduling, traveling salesman, or any other sequence-based problems.

* :mod:`pyswarms.discrete.binary` - classic binary Particle Swarm Optimization algorithm without mutation. Uses a ring topology to choose its neighbours (but can be set to global).

Utilities
---------

Benchmark Functions
~~~~~~~~~~~~~~~~~~~

These functions can be used as benchmarks for assessing the performance of
the optimization algorithm.

* :mod:`pyswarms.utils.functions.single_obj` - single-objective test functions

Search
~~~~~~

These search methods can be used to compare the relative performance of
hyperparameter value combinations in reducing a specified objective function.

* :mod:`pyswarms.utils.search.grid_search` - exhaustive search of optimal performance on selected objective function over cartesian products of provided hyperparameter values

* :mod:`pyswarms.utils.search.random_search` - search for optimal performance on selected objective function over combinations of randomly selected hyperparameter values within specified bounds for specified number of selection iterations

Plotters
~~~~~~~~

A quick and easy to use tool for the visualization of optimizations. It allows you to easily create animations and
to visually check your optimization!

* :mod:`pyswarms.utils.plotters`

Environment
~~~~~~~~~~~~
.. deprecated:: 0.4.0
    Use :mod:`pyswarms.utils.plotters` instead!

Various environments that allow you to analyze your swarm performance and
make visualizations!

* :mod:`pyswarms.utils.environments.plot_environment` - an environment for plotting the cost history and animating particles in a 2D or 3D space.
