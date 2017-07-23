========
PySwarms
========


.. image:: https://img.shields.io/pypi/v/pyswarms.svg
        :target: https://pypi.python.org/pypi/pyswarms

.. image:: https://img.shields.io/travis/ljvmiranda921/pyswarms.svg
        :target: https://travis-ci.org/ljvmiranda921/pyswarms

.. image:: https://readthedocs.org/projects/pyswarms/badge/?version=latest
        :target: https://pyswarms.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/ljvmiranda921/pyswarms/shield.svg
     :target: https://pyup.io/repos/github/ljvmiranda921/pyswarms/
     :alt: Updates


PySwarms is a simple, Python-based, Particle Swarm Optimization (PSO) library. 

* Free software: MIT license
* Documentation: https://pyswarms.readthedocs.io.


Features
--------
* High-level module for Particle Swarm Optimization
* Test optimizers using various objective functions
* (For Devs): Highly-extensible API for implementing your own techniques

Dependencies
-------------
* Python 3.4 and above
* numpy >= 1.10.4
* scipy >= 0.17.0

Installation
-------------
To install PySwarms, run this command in your terminal:

.. code-block:: console

    $ pip install pyswarms

This is the preferred method to install PySwarms, as it will always install the most recent stable release.

Basic Usage
------------
To use PySwarms in your project,

.. code-block:: python

    import pyswarms

Suppose you want to find the minima of :math:`f(x) = x^2` using global best PSO, simply import the 
built-in sphere function, :code:`pyswarms.utils.functions.sphere_func()`, and the necessary optimizer:

.. code-block:: python

    from pyswarms.single import GBestPSO
    from pyswarms.utils.functions import sphere_func

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'm':0.9}

    # Call instance of PSO
    optimizer = GBestPSO(n_particles=10, dims=2, **options)

    # Perform optimization
    stats = optimizer.optimize(sphere_func, iters=100)

More examples can be seen in the :code:`./examples` folder.

Credits
---------
This project was inspired by the pyswarm_ module that performs PSO with constrained support.
The package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _pyswarm: https://github.com/tisimst/pyswarm
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


