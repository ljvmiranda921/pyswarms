
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

    Running on Python version: 3.6.3 |Anaconda custom (64-bit)| (default, Oct 13 2017, 12:02:49) 
    [GCC 7.2.0]
    

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
    cost, pos = optimizer.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    INFO:pyswarms.single.global_best:Iteration 1/1000, cost: 0.11075768527574707
    INFO:pyswarms.single.global_best:Iteration 101/1000, cost: 7.521863508083004e-08
    INFO:pyswarms.single.global_best:Iteration 201/1000, cost: 2.8159915186067273e-11
    INFO:pyswarms.single.global_best:Iteration 301/1000, cost: 8.794923638889175e-17
    INFO:pyswarms.single.global_best:Iteration 401/1000, cost: 1.4699516547190895e-21
    INFO:pyswarms.single.global_best:Iteration 501/1000, cost: 5.111264897313781e-23
    INFO:pyswarms.single.global_best:Iteration 601/1000, cost: 8.329697430155943e-27
    INFO:pyswarms.single.global_best:Iteration 701/1000, cost: 1.662161785541961e-30
    INFO:pyswarms.single.global_best:Iteration 801/1000, cost: 6.140424420222279e-34
    INFO:pyswarms.single.global_best:Iteration 901/1000, cost: 2.0523902169204634e-39
    INFO:pyswarms.single.global_best:================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [-2.431421462417008e-22, -9.502018378214418e-23]
    
    

.. parsed-literal::

    CPU times: user 144 ms, sys: 14.8 ms, total: 159 ms
    Wall time: 151 ms
    

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
    cost, pos = optimizer.optimize(fx.sphere_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    INFO:pyswarms.single.local_best:Iteration 1/1000, cost: 0.01379181672220725
    INFO:pyswarms.single.local_best:Iteration 101/1000, cost: 2.084056061999154e-07
    INFO:pyswarms.single.local_best:Iteration 201/1000, cost: 9.44588224259351e-10
    INFO:pyswarms.single.local_best:Iteration 301/1000, cost: 1.5414149511766008e-13
    INFO:pyswarms.single.local_best:Iteration 401/1000, cost: 3.283944854760787e-16
    INFO:pyswarms.single.local_best:Iteration 501/1000, cost: 2.093366830537641e-20
    INFO:pyswarms.single.local_best:Iteration 601/1000, cost: 5.0279508047072096e-24
    INFO:pyswarms.single.local_best:Iteration 701/1000, cost: 1.0492646748670006e-27
    INFO:pyswarms.single.local_best:Iteration 801/1000, cost: 2.2616819643931453e-29
    INFO:pyswarms.single.local_best:Iteration 901/1000, cost: 8.48269618909152e-35
    INFO:pyswarms.single.local_best:================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [2.122881378865588e-18, -5.35447408455737e-19]
    
    

.. parsed-literal::

    CPU times: user 355 ms, sys: 4.36 ms, total: 359 ms
    Wall time: 353 ms
    

Optimizing a function with bounds
---------------------------------

Another thing that we can do is to set some bounds into our solution, so
as to contain our candidate solutions within a specific range. We can do
this simply by passing a ``bounds`` parameter, of type ``tuple``, when
creating an instance of our swarm. Let’s try this using the global-best
PSO with the Rastrigin function (``rastrigin_func`` in
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
    cost, pos = optimizer.optimize(fx.rastrigin_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    INFO:pyswarms.single.global_best:Iteration 1/1000, cost: 12.243865048066269
    INFO:pyswarms.single.global_best:Iteration 101/1000, cost: 1.1759164022634394
    INFO:pyswarms.single.global_best:Iteration 201/1000, cost: 0.9949603350768896
    INFO:pyswarms.single.global_best:Iteration 301/1000, cost: 0.9949590581556009
    INFO:pyswarms.single.global_best:Iteration 401/1000, cost: 0.9949590570934177
    INFO:pyswarms.single.global_best:Iteration 501/1000, cost: 0.9949590570932898
    INFO:pyswarms.single.global_best:Iteration 601/1000, cost: 0.9949590570932898
    INFO:pyswarms.single.global_best:Iteration 701/1000, cost: 0.9949590570932898
    INFO:pyswarms.single.global_best:Iteration 801/1000, cost: 0.9949590570932898
    INFO:pyswarms.single.global_best:Iteration 901/1000, cost: 0.9949590570932898
    INFO:pyswarms.single.global_best:================================
    Optimization finished!
    Final cost: 0.9950
    Best value: [3.5850411183743393e-09, -0.9949586379966202]
    
    

.. parsed-literal::

    CPU times: user 213 ms, sys: 7.55 ms, total: 221 ms
    Wall time: 210 ms
    

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

    Running Python 3.5.2 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:53:06) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
    

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
    
    # instatiate the optimizer
    x_max = 10 * np.ones(2)
    x_min = -1 * x_max
    bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)
    
    # now run the optimization, pass a=1 and b=100 as a tuple assigned to args
    
    cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, print_step=100, verbose=3, a=1, b=100, c=0)


.. parsed-literal::

    INFO:pyswarms.single.global_best:Arguments Passed to Objective Function: {'c': 0, 'b': 100, 'a': 1}
    INFO:pyswarms.single.global_best:Iteration 1/1000, cost: 1022.9667801907804
    INFO:pyswarms.single.global_best:Iteration 101/1000, cost: 0.0011172801146408992
    INFO:pyswarms.single.global_best:Iteration 201/1000, cost: 7.845605970774126e-07
    INFO:pyswarms.single.global_best:Iteration 301/1000, cost: 1.313503109901238e-09
    INFO:pyswarms.single.global_best:Iteration 401/1000, cost: 5.187079604907219e-10
    INFO:pyswarms.single.global_best:Iteration 501/1000, cost: 1.0115283486088853e-10
    INFO:pyswarms.single.global_best:Iteration 601/1000, cost: 2.329870757208421e-13
    INFO:pyswarms.single.global_best:Iteration 701/1000, cost: 4.826176894160183e-15
    INFO:pyswarms.single.global_best:Iteration 801/1000, cost: 3.125715456651088e-17
    INFO:pyswarms.single.global_best:Iteration 901/1000, cost: 1.4236768129666014e-19
    INFO:pyswarms.single.global_best:================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [0.99999999996210465, 0.9999999999218413]
    
    

It is also possible to pass a dictionary of key word arguments by using
``**`` decorator when passing the dict

.. code-block:: python

    kwargs={"a": 1.0, "b": 100.0, 'c':0}
    cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, print_step=100, verbose=3, **kwargs)


.. parsed-literal::

    INFO:pyswarms.single.global_best:Arguments Passed to Objective Function: {'c': 0, 'b': 100.0, 'a': 1.0}
    INFO:pyswarms.single.global_best:Iteration 1/1000, cost: 1.996797703363527e-21
    INFO:pyswarms.single.global_best:Iteration 101/1000, cost: 1.0061676299213387e-24
    INFO:pyswarms.single.global_best:Iteration 201/1000, cost: 4.8140236741112245e-28
    INFO:pyswarms.single.global_best:Iteration 301/1000, cost: 2.879342304056693e-29
    INFO:pyswarms.single.global_best:Iteration 401/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 501/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 601/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 701/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 801/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 901/1000, cost: 0.0
    INFO:pyswarms.single.global_best:================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [1.0, 1.0]
    
    

Any key word arguments in the objective function can be left out as they
will be passed the default as defined in the prototype. Note here, ``c``
is not passed into the function.

.. code-block:: python

    cost, pos = optimizer.optimize(rosenbrock_with_args, 1000, print_step=100, verbose=3, a=1, b=100)


.. parsed-literal::

    INFO:pyswarms.single.global_best:Arguments Passed to Objective Function: {'b': 100, 'a': 1}
    INFO:pyswarms.single.global_best:Iteration 1/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 101/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 201/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 301/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 401/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 501/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 601/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 701/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 801/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 901/1000, cost: 0.0
    INFO:pyswarms.single.global_best:================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [1.0, 1.0]
    
    
