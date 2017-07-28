=====
Usage
=====

To use PySwarms in a project::

    import pyswarms as ps


Basic Usage
------------
Suppose you want to find the minima of :math:`f(x) = x^2` using global best PSO, simply import the 
built-in sphere function, :code:`pyswarms.utils.functions.sphere_func()`, and the necessary optimizer:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import sphere_func

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'm':0.9}

    # Call instance of PSO
    optimizer = ps.GBestPSO(n_particles=10, dims=2, **options)

    # Perform optimization
    stats = optimizer.optimize(sphere_func, iters=100)
