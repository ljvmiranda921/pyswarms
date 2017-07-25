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


# TODO: Implement Beale's Function
def beale_func(x):
    """Beale objective function.

    Only takes two dimensions and has a global minimum at f([0,3.5])
    Its domain is bounded between [-4.5, 4.5]

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    assert np.logical_and(x >= -4.5, x <= 4.5).all(), "Input for \
            Beale function must be within [-4.5, 4.5]."
    assert x.shape[1] == 2, "Only takes two-dimensional input."

    # TODO: Write actual function here
    
    # TODO: Change this part by returning the actual value when
    # you compute x.
    return np.array([0,0,0])

# TODO: Implement Goldstein-Price's Function
def goldstein_func(x):
    """Goldstein-Price's objective function.

    Only takes two dimensions and has a global minimum at f([0,-1])
    Its domain is bounded between [-2, 2]

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    assert np.logical_and(x >= -2, x <= 2).all(), "Input for \
            Goldstein-Price function must be within [-4.5, 4.5]."
    assert x.shape[1] == 2, "Only takes two-dimensional input."

    # TODO: Write actual function here
    
    # TODO: Change this part by returning the actual value when
    # you compute x.
    return np.array([3,3,3])

# TODO: Implement Goldstein-Price's Function
def booth_func(x):
    """Booth's objective function.

    Only takes two dimensions and has a global minimum at f([1,3])
    Its domain is bounded between [-10, 10]

    Parameters
    ----------
    x : numpy.ndarray 
        set of inputs of shape (n_particles, dims)

    Returns
    -------
    numpy.ndarray 
        computed cost of size (n_particles, )
    """
    assert np.logical_and(x >= -10, x <= 10).all(), "Input for \
            Booth function must be within [-10, 10]."
    assert x.shape[1] == 2, "Only takes two-dimensional input."

    # TODO: Write actual function here
    
    # TODO: Change this part by returning the actual value when
    # you compute x.
    return np.array([0,0,0])