.. image:: docs/pyswarms-header.png
        :alt: PySwarms Logo
        :align: center

------------

.. image:: https://badge.fury.io/py/pyswarms.svg
        :target: https://badge.fury.io/py/pyswarms
        :alt: PyPI Version

.. image:: https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=master
        :target: https://travis-ci.org/ljvmiranda921/pyswarms
        :alt: Build Status

.. image:: https://readthedocs.org/projects/pyswarms/badge/?version=latest
        :target: https://pyswarms.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg   
        :target: https://raw.githubusercontent.com/ljvmiranda921/pyswarms/master/LICENSE
        :alt: License

.. image:: http://joss.theoj.org/papers/10.21105/joss.00433/status.svg
        :target: https://doi.org/10.21105/joss.00433
        :alt: Citation

.. image:: https://badges.gitter.im/Join%20Chat.svg
        :target: https://gitter.im/pyswarms/Issues
        :alt: Gitter Chat

PySwarms is an extensible research toolkit for particle swarm optimization (PSO) in Python.

It is intended for swarm intelligence researchers, practitioners, and students who would like a high-level declarative interface of implementing PSO in their problems. PySwarms both allows basic optimization with PSO and interaction with swarm optimizations. Interaction is enabled due to object primitives provided by the package for optimization. This makes PySwarms useful for researchers or students.

* **Free software:** MIT license
* **Documentation:** https://pyswarms.readthedocs.io.
* **Python versions:** 2.7, 3.4, 3.5 and above

Features
--------

* High-level module for Particle Swarm Optimization. For a list of all optimizers, check this_ link.
* Built-in objective functions to test optimization algorithms.
* Plotting environment for cost histories and particle movement.
* Hyperparameter search tools to optimize swarm behaviour.
* (For Devs and Researchers): Highly-extensible API for implementing your own techniques.

.. _this: https://pyswarms.readthedocs.io/en/latest/features.html

Dependencies
-------------
* numpy >= 1.13.0
* scipy >= 0.17.0
* matplotlib >= 1.3.1

Installation
-------------
To install PySwarms, run this command in your terminal:

.. code-block:: console

    $ pip install pyswarms

This is the preferred method to install PySwarms, as it will always install the most recent stable release.

In case you want to install the bleeding-edge version, clone this repo:

.. code-block:: console

    $ git clone https://github.com/ljvmiranda921/pyswarms.git

and then run

.. code-block:: console

    $ python setup.py install

Basic Usage
------------

PySwarms provides a high-level implementation of various particle swarm optimization
algorithms. Thus, it aims to be very easy to use and customize. Moreover, supporting
modules can also be used to help you in your optimization problem.


Optimizing a sphere function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can import PySwarms as any other Python module,

.. code-block:: python

    import pyswarms as ps

Suppose we want to find the minima of :math:`f(x) = x^2` using global best PSO, simply import the 
built-in sphere function, :code:`pyswarms.utils.functions.sphere_func()`, and the necessary optimizer:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

    # Perform optimization
    best_cost, best_pos = optimizer.optimize(fx.sphere_func, iters=100, verbose=3, print_step=25)

.. code-block::

    >>> 2017-10-03 10:12:33,859 - pyswarms.single.global_best - INFO - Iteration 1/100, cost: 0.131244226714
    >>> 2017-10-03 10:12:33,878 - pyswarms.single.global_best - INFO - Iteration 26/100, cost: 1.60297958653e-05
    >>> 2017-10-03 10:12:33,893 - pyswarms.single.global_best - INFO - Iteration 51/100, cost: 1.60297958653e-05
    >>> 2017-10-03 10:12:33,906 - pyswarms.single.global_best - INFO - Iteration 76/100, cost: 2.12638727702e-06
    >>> 2017-10-03 10:12:33,921 - pyswarms.single.global_best - INFO - ================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [-0.0003521098028145481, -0.00045459382339127453]

This will run the optimizer for :code:`100` iterations, and will return the best cost and best
position found by the swarm. In addition, you can also access various histories by calling on
properties of the class:

.. code-block:: python

    # Obtain the cost history
    optimizer.get_cost_history

    # Obtain the position history
    optimizer.get_pos_history

    # Obtain the velocity history
    optimizer.get_velocity_history

At the same time, you can also obtain the mean personal best and mean neighbor
history for local best PSO implementations. Simply call :code:`mean_pbest_history`
and :code:`optimizer.get_mean_neighbor_history` respectively.

Hyperparameter search tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySwarms implements a grid search and random search technique to find the best
parameters for your optimizer. Setting them up is easy. In this example,
let's try using :code:`pyswarms.utils.search.RandomSearch` to find the optimal
parameters for :code:`LocalBestPSO` optimizer.

Here, we input a range, enclosed in tuples, to define the space in which
the parameters will be found. Thus, :code:`(1,5)` pertains to a range from
1 to 5.

.. code-block:: python

    import numpy as np
    import pyswarms as ps
    from pyswarms.utils.search import RandomSearch
    from pyswarms.utils.functions import single_obj as fx

    # Set-up choices for the parameters
    options = {
        'c1': (1,5),
        'c2': (6,10),
        'w': (2,5),
        'k': (11, 15),
        'p': 1
    }

    # Create a RandomSearch object
    # n_selection_iters is the number of iterations to run the searcher
    # iters is the number of iterations to run the optimizer
    g = RandomSearch(ps.single.LocalBestPSO, n_particles=40,
                dimensions=20, options=options, objective_func=fx.sphere_func,
                iters=10, n_selection_iters=100)

    best_score, best_options = g.search()

This then returns the best score found during optimization, and the
hyperparameter options that enables it.

.. code-block:: python

    >>> best_score
    1.41978545901
    >>> best_options['c1']
    1.543556887693
    >>> best_options['c2']
    9.504769054771

Plotting environments
~~~~~~~~~~~~~~~~~~~~~

It is also possible to plot optimizer performance for the sake of formatting.
The plotting environment is built on top of :code:`matplotlib`, making it
highly-customizable.

The environment takes in the optimizer and its parameters, then performs
a fresh run to plot the cost and create animation.

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx
    from pyswarms.utils.environments import PlotEnvironment

    # Set-up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options)

    # Initialize plot environment
    plt_env = PlotEnvironment(optimizer, fx.sphere_func, 1000)

    # Plot the cost
    plt_env.plot_cost(figsize=(8,6));
    plt.show()

.. image:: docs/examples/output_9_0.png
        :target: docs/examples/output_9_0.png
        :width: 320 px
        :alt: cost history plot

We can also plot the animation,

.. code-block:: python

    plt_env.plot_particles2D(limits=((-1.2,1.2),(-1.2,1.2))

.. image:: docs/examples/output_3d.gif
        :target: docs/examples/output_3d.gif
        :width: 320 px
        :alt: 3d particle plot

Contributing
------------

PySwarms is currently maintained by a single person (me!) with the aid of a
few but very helpful contributors. We would appreciate it if you can lend
a hand with the following:

* Find bugs and fix them
* Update documentation in docstrings
* Implement new optimizers to our collection
* Make utility functions more robust.

If you wish to contribute, check out our contributing guide in this link_.
Moreover, you can also see the list of features that need some help in our
Issues_ page and in this list_.

.. _link: https://pyswarms.readthedocs.io/en/latest/contributing.html
.. _Issues: https://github.com/ljvmiranda921/pyswarms/issues
.. _list: https://github.com/ljvmiranda921/pyswarms/issues/5

**Most importantly**, first time contributors are welcome to join! I try my best
to help you get started and enable you to make your first Pull Request! Let's
learn from each other!

Credits
-------

This project was inspired by the pyswarm_ module that performs PSO with constrained support.
The package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

This is currently maintained by Lester James V. Miranda with other helpful contributors (v.0.1.7):

* Carl-K (`@Carl-K <https://github.com/Carl-K>`_)
* Siobhán Cronin (`@SioKCronin <https://github.com/SioKCronin>`_)
* Andrew Jarcho (`@jazcap53 <https://github.com/jazcap53>`_)
* Charalampos Papadimitriou (`@CPapadim <https://github.com/CPapadim>`_)

.. _pyswarm: https://github.com/tisimst/pyswarm
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Cite us
--------
Are you using PySwarms in your project or research? Please cite us!

* Miranda L.J., (2018). PySwarms: a research toolkit for Particle Swarm Optimization in Python. *Journal of Open Source Software*, 3(21), 433, https://doi.org/joss.00433

.. code-block:: bibtex

    @article{pyswarmsJOSS2018,
        author  = {Lester James V. Miranda},
        title   = "{P}y{S}warms, a research-toolkit for {P}article {S}warm {O}ptimization in {P}ython",
        journal = {Journal of Open Source Software},
        year    = {2018},
        volume  = {3},
        issue   = {21},
        doi     = {10.21105/joss.00433},
        url     = {https://doi.org/10.21105/joss.00433}
    }

Projects using PySwarms
~~~~~~~~~~~~~~~~~~~~~~~

* Nandy, Abhishek, and Manisha Biswas. "Applying Python to Reinforcement Learning." *Reinforcement Learning*. Apress, Berkeley, CA, 2018. 89-128.
* Benedetti, Marcello, et al. "A generative modeling approach for benchmarking and training shallow quantum circuits." *arXiv preprint arXiv:1801.07686* (2018).

Others
------
Like it? Love it? Leave us a star on Github_ to show your appreciation! 

.. _Github: https://github.com/ljvmiranda921/pyswarms
