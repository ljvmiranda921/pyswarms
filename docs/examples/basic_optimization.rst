
Basic Optimization
==================

In this example, we’ll be performing a simple optimization of
single-objective functions using the global-best optimizer in
``pyswarms.single.GBestPSO`` and the local-best optimizer in
``pyswarms.single.LBestPSO``. This aims to demonstrate the basic
capabilities of the library when applied to benchmark problems.

.. code-block:: python

    import sys
    # Change directory to access the pyswarms module
    sys.path.append('../')


.. code-block:: python

    print('Running on Python version: {}'.format(sys.version))


.. parsed-literal::

    Running on Python version: 3.6.7 (default, Oct 22 2018, 11:32:17)
    [GCC 8.2.0]


.. code-block:: python

    # Import modules
    import numpy as np

    # Import PySwarms
    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Some more magic so that the notebook will reload external python modules;
    # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
    %load_ext autoreload
    %autoreload 2

Optimizing a function
---------------------

First, let’s start by optimizing the sphere function. Recall that the
minima of this function can be located at ``f(0,0..,0)`` with a value of
``0``. In case you don’t remember the characteristics of a given
function, simply call ``help(<function>)``.

For now let’s just set some arbitrary parameters in our optimizers.
There are, at minimum, three steps to perform optimization:

1. Set the hyperparameters to configure the swarm as a ``dict``.
2. Create an instance of the optimizer by passing the dictionary along
   with the necessary arguments.
3. Call the ``optimize()`` method and have it store the optimal cost and
   position in a variable.

The ``optimize()`` method returns a ``tuple`` of values, one of which
includes the optimal cost and position after optimization. You can store
it in a single variable and just index the values, or unpack it using
several variables at once.

.. code-block:: python

    %%time
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=1000)


.. parsed-literal::

    2019-01-30 04:23:31,846 - pyswarms.single.global_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pyswarms.single.global_best: 100%|██████████|1000/1000, best_cost=5.34e-43
    2019-01-30 04:23:46,631 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 5.3409429804817095e-43, best pos: [-4.84855366e-22 -5.46817677e-22]


.. parsed-literal::

    CPU times: user 5.63 s, sys: 916 ms, total: 6.55 s
    Wall time: 14.8 s


We can see that the optimizer was able to find a good minima as shown
above. You can control the verbosity of the output using the ``verbose``
argument, and the number of steps to be printed out using the
``print_step`` argument.

Now, let’s try this one using local-best PSO:

.. code-block:: python

    %%time
    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}

    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)

    # Perform optimization
    cost, pos = optimizer.optimize(fx.sphere, iters=1000)


.. parsed-literal::

    2019-01-30 04:23:46,672 - pyswarms.single.local_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    pyswarms.single.local_best: 100%|██████████|1000/1000, best_cost=1.19e-48
    2019-01-30 04:24:02,254 - pyswarms.single.local_best - INFO - Optimization finished | best cost: 1.1858559943008184e-48, best pos: [5.47013119e-24 7.95177208e-25]


.. parsed-literal::

    CPU times: user 6.63 s, sys: 1.04 s, total: 7.68 s
    Wall time: 15.6 s


Optimizing a function with bounds
---------------------------------

Another thing that we can do is to set some bounds into our solution, so
as to contain our candidate solutions within a specific range. We can do
this simply by passing a ``bounds`` parameter, of type ``tuple``, when
creating an instance of our swarm. Let’s try this using the global-best
PSO with the Rastrigin function (``rastrigin`` in
``pyswarms.utils.functions.single_obj``).

Recall that the Rastrigin function is bounded within ``[-5.12, 5.12]``.
If we pass an unbounded swarm into this function, then a ``ValueError``
might be raised. So what we’ll do is to create a bound within the
specified range. There are some things to remember when specifying a
bound:

-  A bound should be of type tuple with length 2.
-  It should contain two ``numpy.ndarrays`` so that we have a
   ``(min_bound, max_bound)``
-  Obviously, all values in the ``max_bound`` should always be greater
   than the ``min_bound``. Their shapes should match the dimensions of
   the swarm.

What we’ll do now is to create a 10-particle, 2-dimensional swarm. This
means that we have to set our maximum and minimum boundaries with the
shape of 2. In case we want to initialize an n-dimensional swarm, we
then have to set our bounds with the same shape n. A fast workaround for
this would be to use the ``numpy.ones`` function multiplied by a
constant.

.. code-block:: python

    # Create bounds
    max_bound = 5.12 * np.ones(2)
    min_bound = - max_bound
    bounds = (min_bound, max_bound)

.. code-block:: python

    %%time
    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO with bounds argument
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(fx.rastrigin, iters=1000)


.. parsed-literal::

    2019-01-30 04:24:02,463 - pyswarms.single.global_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pyswarms.single.global_best: 100%|██████████|1000/1000, best_cost=0
    2019-01-30 04:24:17,995 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 0.0, best pos: [1.99965504e-09 9.50602717e-10]



.. parsed-literal::

    CPU times: user 6.74 s, sys: 1.01 s, total: 7.75 s
    Wall time: 15.5 s


Basic Optimization with Arguments
---------------------------------

Here, we will run a basic optimization using an objective function that
needs parameterization. We will use the ``single.GBestPSO`` and a
version of the rosenbrock function to demonstrate

.. code-block:: python

    import sys
    # change directory to access pyswarms
    sys.path.append('../')

    print("Running Python {}".format(sys.version))


.. parsed-literal::

    Running Python 3.6.7 (default, Oct 22 2018, 11:32:17)
    [GCC 8.2.0]


.. code-block:: python

    # import modules
    import numpy as np

    # create a parameterized version of the classic Rosenbrock unconstrained optimzation function
    def rosenbrock_with_args(x, a, b, c=0):

        f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
        return f

Using Arguments
~~~~~~~~~~~~~~~

Arguments can either be passed in using a tuple or a dictionary, using
the ``kwargs={}`` paradigm. First lets optimize the Rosenbrock function
using keyword arguments. Note in the definition of the Rosenbrock
function above, there were two arguments that need to be passed other
than the design variables, and one optional keyword argument, ``a``,
``b``, and ``c``, respectively

.. code-block:: python

    from pyswarms.single.global_best import GlobalBestPSO

    # instantiate the optimizer
    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

    # now run the optimization, pass a=1 and b=100 as a tuple assigned to args

    cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, a=1, b=100, c=0)


.. parsed-literal::


    2019-01-30 04:24:18,385 - pyswarms.single.global_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pyswarms.single.global_best: 100%|██████████|1000/1000, best_cost=1.65e-18
    2019-01-30 04:24:33,873 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 1.6536536065757395e-18, best pos: [1. 1.]


It is also possible to pass a dictionary of key word arguments by using
``**`` decorator when passing the dict

.. code-block:: python

    kwargs={"a": 1.0, "b": 100.0, 'c':0}
    cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, **kwargs)


.. parsed-literal::

    2019-01-30 04:24:33,904 - pyswarms.single.global_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pyswarms.single.global_best: 100%|██████████|1000/1000, best_cost=9.13e-19
    2019-01-30 04:24:49,482 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 9.132114249459913e-19, best pos: [1. 1.]


Any key word arguments in the objective function can be left out as they
will be passed the default as defined in the prototype. Note here, ``c``
is not passed into the function.

.. code-block:: python

    cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, a=1, b=100)


.. parsed-literal::

    2019-01-30 04:24:49,518 - pyswarms.single.global_best - INFO - Optimize for 1000 iters with {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    pyswarms.single.global_best: 100%|██████████|1000/1000, best_cost=9.13e-19
    2019-01-30 04:25:05,071 - pyswarms.single.global_best - INFO - Optimization finished | best cost: 9.125748012380431e-19, best pos: [1. 1.]
