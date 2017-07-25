# -*- coding: utf-8 -*-

"""single_obj.py: collection of single-objective functions

All objective functions obj_func() must accept a (numpy.ndarray)
with shape (n_particles, dims). Thus, each row represents a 
particle, and each column represents its position on a specific
dimension of the search-space.

In this context, obj_func() must return an array j of size
(n_particles, ) that contains all the computed fitness for
each particle. 

Whenever you make changes to this file via an implementation
of a new objective function, be sure to perform unittesting
in order to check if all functions implemented adheres to 
the design pattern stated above.
"""

import numpy as np

def sphere_func(x):
    """Sphere objective function.

    Has a global minimum at 0 and with a search domain of
        [-inf, inf]

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    j = (x**2.0).sum(axis=1)

    return j

def rastrigin_func(x):
    """Rastrigin objective function.

    Has a global minimum at f(0,0,...,0) with a search
    domain of -[-5.12, 5.12]

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    assert np.logical_and(x >= -5.12, x <= 5.12).all(), "Input for \
            Rastrigin function must be within [-5.12, 5.12]."

    d = x.shape[1]
    j = 10.0 * d + (x**2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1) 

    return j

def ackley_func(x):
    """Ackley's objective function.

    Has a global minimum at f(0,0,...,0) with a search
    domain of [-32, 32]

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    assert np.logical_and(x >= -32, x <= 32).all(), "Input for \
            Rastrigin function must be within [-32, 32]."

    d = x.shape[1]
    j = (-20.0 * np.exp(-0.2 * np.sqrt((1/d) * (x**2).sum(axis=1)))
        - np.exp((1/d) * np.cos(2 * np.pi * x).sum(axis=1))
        + 20.0
        + np.exp(1))

    return j

def rosenbrock_func(x):
    """Rosenbrock objective function.

    Also known as the Rosenbrock's valley or Rosenbrock's banana
    function. Has a global minimum of np.ones(dims) where dims
    is x.shape[1]. The search domain is [-inf, inf].

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    j = (100 * np.square(x[:,1:] - x[:,:-1]**2.0)
        + (1.0 - x[:,:-1]) ** 2.0)

    return j.ravel()