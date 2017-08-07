
Basic Optimization
==================

In this example, we'll be performing a simple optimization of
single-objective functions using the global-best optimizer in
``pyswarms.single.GBestPSO`` and the local-best optimizer in
``pyswarms.single.LBestPSO``. This aims to demonstrate the basic
capabilities of the library when applied to benchmark problems.

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
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
    
    # Perform optimization
    cost, pos = optimizer.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 0.215476174296
    Iteration 101/1000, cost: 5.26998280059e-07
    Iteration 201/1000, cost: 1.31313801471e-11
    Iteration 301/1000, cost: 1.63948780036e-15
    Iteration 401/1000, cost: 2.72294062778e-19
    Iteration 501/1000, cost: 3.69002488955e-22
    Iteration 601/1000, cost: 3.13387805277e-27
    Iteration 701/1000, cost: 1.65106278625e-30
    Iteration 801/1000, cost: 6.95403958989e-35
    Iteration 901/1000, cost: 1.33520105208e-41
    ================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [9.4634973546019334e-23, 1.7011045174312443e-22]
    
    

We can see that the optimizer was able to find a good minima as shown
above. You can control the verbosity of the output using the ``verbose``
argument, and the number of steps to be printed out using the
``print_step`` argument.

Now, let's try this one using local-best PSO:

.. code-block:: python

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
    
    # Call instance of PSO
    optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=options)
    
    # Perform optimization
    cost, pos = optimizer.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 0.0573032190292
    Iteration 101/1000, cost: 8.92699853837e-07
    Iteration 201/1000, cost: 4.56513550671e-10
    Iteration 301/1000, cost: 2.35083665314e-16
    Iteration 401/1000, cost: 8.09981989467e-20
    Iteration 501/1000, cost: 2.58846774519e-22
    Iteration 601/1000, cost: 3.33919326611e-26
    Iteration 701/1000, cost: 2.15052800954e-30
    Iteration 801/1000, cost: 1.09638832057e-33
    Iteration 901/1000, cost: 3.92671836329e-38
    ================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [1.4149803165668767e-21, -9.9189063589743749e-24]
    
    

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
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
    
    # Perform optimization
    cost, pos = optimizer.optimize(fx.rastrigin_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    Iteration 1/1000, cost: 6.93571097813
    Iteration 101/1000, cost: 0.00614705911661
    Iteration 201/1000, cost: 7.22876336567e-09
    Iteration 301/1000, cost: 5.89750470681e-13
    Iteration 401/1000, cost: 0.0
    Iteration 501/1000, cost: 0.0
    Iteration 601/1000, cost: 0.0
    Iteration 701/1000, cost: 0.0
    Iteration 801/1000, cost: 0.0
    Iteration 901/1000, cost: 0.0
    ================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [-6.763954278218746e-11, 2.4565912679296225e-09]
    

