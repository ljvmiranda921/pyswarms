
Basic Optimization
==================

In this example, we'll be performing a simple optimization of
single-objective functions using the global-best optimizer in
``pyswarms.single.GBestPSO`` and the local-best optimizer in
``pyswarms.single.LBestPSO``. This aims to demonstrate the basic
capabilities of the library when applied to benchmark problems.

.. code:: ipython3

    import sys
    sys.path.append('../')

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

First, let's start by optimizing the sphere function. Recall that the
minima of this function can be located at ``f(0,0..,0)`` with a value of
``0``. In case you don't remember the characteristics of a given
function, simply call ``help(<function>)``.

For now let's just set some arbitrary parameters in our optimizers.
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

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    
    # Call instance of PSO
    gbest_pso = ps.single.GBestPSO(n_particles=10, dims=2, **options)
    
    # Perform optimization
    cost, pos = gbest_pso.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 0.0035824017918
    Iteration 101/1000, cost: 1.02538653288e-08
    Iteration 201/1000, cost: 9.95696087972e-13
    Iteration 301/1000, cost: 8.22034343822e-16
    Iteration 401/1000, cost: 3.7188438887e-19
    Iteration 501/1000, cost: 1.23935292549e-25
    Iteration 601/1000, cost: 6.03016193248e-28
    Iteration 701/1000, cost: 3.70755768681e-34
    Iteration 801/1000, cost: 2.64385328058e-37
    Iteration 901/1000, cost: 1.76488833461e-40
    ================================
    Optimization finished!
    Final cost: 0.000
    Best value: [-6.5732265560180066e-24, -7.4004230063696789e-22]
    
    

We can see that the optimizer was able to find a good minima as shown
above. You can control the verbosity of the output using the ``verbose``
argument, and the number of steps to be printed out using the
``print_step`` argument.

Now, let's try this one using local-best PSO:

.. code-block:: python

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 2}
    
    # Call instance of PSO
    lbest_pso = ps.single.LBestPSO(n_particles=10, dims=2, **options)
    
    # Perform optimization
    cost, pos = lbest_pso.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 0.190175474818
    Iteration 101/1000, cost: 1.14470953523e-06
    Iteration 201/1000, cost: 6.79485221069e-11
    Iteration 301/1000, cost: 1.00691597113e-14
    Iteration 401/1000, cost: 2.98301783945e-18
    Iteration 501/1000, cost: 2.13856158282e-20
    Iteration 601/1000, cost: 5.49351926815e-25
    Iteration 701/1000, cost: 1.7673389214e-29
    Iteration 801/1000, cost: 1.83082804473e-33
    Iteration 901/1000, cost: 1.75920918448e-36
    ================================
    Optimization finished!
    Final cost: 3.000
    Best value: [-8.2344756213578705e-21, -2.6563827831876976e-20]
    
    

Optimizing a function with bounds
---------------------------------

Another thing that we can do is to set some bounds into our solution, so
as to contain our candidate solutions within a specific range. We can do
this simply by passing a ``bounds`` parameter, of type ``tuple``, when
creating an instance of our swarm. Let's try this using the global-best
PSO with the Rastrigin function (``rastrigin_func`` in
``pyswarms.utils.functions.single_obj``).

Recall that the Rastrigin function is bounded within ``[-5.12, 5.12]``.
If we pass an unbounded swarm into this function, then a ``ValueError``
might be raised. So what we'll do is to create a bound within the
specified range. There are some things to remember when specifying a
bound:

-  A bound should be of type tuple with length 2.
-  It should contain two ``numpy.ndarrays`` so that we have a
   ``(min_bound, max_bound)``
-  Obviously, all values in the ``max_bound`` should always be greater
   than the ``min_bound``. Their shapes should match the dimensions of
   the swarm.

What we'll do now is to create a 10-particle, 2-dimensional swarm. This
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

    # Initialize swarm
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    
    # Call instance of PSO with bounds argument
    optimizer = ps.single.GBestPSO(n_particles=10, dims=2, bounds=bounds, **options)
    
    # Perform optimization
    cost, pos = optimizer.optimize(fx.rastrigin_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 10.3592595923
    Iteration 101/1000, cost: 0.00381030608321
    Iteration 201/1000, cost: 1.31982446305e-07
    Iteration 301/1000, cost: 1.16529008665e-11
    Iteration 401/1000, cost: 0.0
    Iteration 501/1000, cost: 0.0
    Iteration 601/1000, cost: 0.0
    Iteration 701/1000, cost: 0.0
    Iteration 801/1000, cost: 0.0
    Iteration 901/1000, cost: 0.0
    ================================
    Optimization finished!
    Final cost: 0.000
    Best value: [8.9869507154871327e-10, -2.7262405947023504e-09]

