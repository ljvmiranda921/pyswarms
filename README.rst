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


PySwarms is a simple Python-based Particle Swarm Optimization (PSO) library. It offers an array of
single-objective and multi-objective PSO algorithms, various objective functions to test your optimizer,
and an API for analyzing optimizer performance. 


* Free software: MIT license
* Documentation: https://pyswarms.readthedocs.io.


Features
--------

1. **Particle Swarm Optimization**
    Choose from various PSO implementations to optimize a given problem. Different flavors exist 
    for this task, ranging from single-objective optimization to multi-objective ones. Classical
    techniques, such as global-best and personal-best are also implemented.
    
    a. Single-Objective PSO
        Single-objective optimizers attempt to find the global optima given a single objective
        function. These functions tend to take an array of values and returns the fitness 
        corresponding those values.
         
            * Global-best 
            * Personal-best    
    b. Multi-Objective PSO *(Coming soon)*

2. **Objective Functions for single-objective and multi-objective problems**
    Test your optimizer by having it search for optimum values in different objective functions. 
    Standard functions such as Sphere, Rastrigin's, and Ackley's are implemented. 

3. **Performance analysis API** *(Coming soon)*
    Analyze how your optimizer works: visualize the position over iterations, generate error
    plots, and perform grid search on your hyperparameters using this easy API.

Dependencies
-------------
* Python 3.X
* Numpy 

Credits
---------

This project was inspired by the pyswarm_ module that performs PSO with constrained support.   
The package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _pyswarm: https://github.com/tisimst/pyswarm
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


