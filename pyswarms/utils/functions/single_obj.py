# -*- coding: utf-8 -*-

"""single_obj.py: collection of single-objective functions

All objective functions :code:`obj_func()` must accept a
:code:`(numpy.ndarray)` with shape :code:`(n_particles, dimensions)`.
Thus, each row represents a  particle, and each column represents its
position on a specific dimension of the search-space.

In this context, :code:`obj_func()` must return an array :code:`j`
of size :code:`(n_particles, )` that contains all the computed fitness
for each particle.

Whenever you make changes to this file via an implementation
of a new objective function, be sure to perform unittesting
in order to check if all functions implemented adheres to
the design pattern stated above.
"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import numpy as np


def sphere_func(x):
    """Sphere objective function.

    Has a global minimum at :code:`0` and with a search domain of
        :code:`[-inf, inf]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    j = (x**2.0).sum(axis=1)

    return j


def rastrigin_func(x):
    """Rastrigin objective function.

    Has a global minimum at :code:`f(0,0,...,0)` with a search
    domain of :code:`[-5.12, 5.12]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -5.12, x <= 5.12).all():
        raise ValueError('Input for Rastrigin function must be within '
                         '[-5.12, 5.12].')

    d = x.shape[1]
    j = 10.0 * d + (x**2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)

    return j


def ackley_func(x):
    """Ackley's objective function.

    Has a global minimum at :code:`f(0,0,...,0)` with a search
    domain of [-32, 32]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not np.logical_and(x >= -32, x <= 32).all():
        raise ValueError('Input for Ackley function must be within [-32, 32].')

    d = x.shape[1]
    j = (-20.0 * np.exp(-0.2 * np.sqrt((1/d) * (x**2).sum(axis=1)))
         - np.exp((1/float(d)) * np.cos(2 * np.pi * x).sum(axis=1))
         + 20.0
         + np.exp(1))

    return j


def rosenbrock_func(x):
    """Rosenbrock objective function.

    Also known as the Rosenbrock's valley or Rosenbrock's banana
    function. Has a global minimum of :code:`np.ones(dimensions)` where
    :code:`dimensions` is :code:`x.shape[1]`. The search domain is
    :code:`[-inf, inf]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`
    """
    j = (100 * np.square(x[:, 1:] - x[:, :-1]**2.0)
         + (1.0 - x[:, :-1]) ** 2.0)

    return j.ravel()


def beale_func(x):
    """Beale objective function.

    Only takes two dimensions and has a global minimum at
    :code:`f([3,0.5])` Its domain is bounded between :code:`[-4.5, 4.5]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError('Beale function only takes two-dimensional input.')
    if not np.logical_and(x >= -4.5, x <= 4.5).all():
        raise ValueError('Input for Beale function must be within '
                         '[-4.5, 4.5].')

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = ((1.5 - x_ + x_ * y_)**2.0
         + (2.25 - x_ + x_ * y_**2.0)**2.0
         + (2.625 - x_ + x_ * y_**3.0)**2.0)

    return j


# TODO: Implement Goldstein-Price's Function
def goldstein_func(x):
    """Goldstein-Price's objective function.

    Only takes two dimensions and has a global minimum at
    :code:`f([0,-1])`. Its domain is bounded between :code:`[-2, 2]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError('Goldstein function only takes two-dimensional '
                         'input.')
    if not np.logical_and(x >= -2, x <= 2).all():
        raise ValueError('Input for Goldstein-Price function must be within '
                         '[-2, 2].')

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = ((1 + (x_ + y_ + 1)**2.0
         * (19 - 14*x_ + 3*x_**2.0 - 14*y_ + 6*x_*y_ + 3*y_**2.0))
         * (30 + (2*x_ - 3 * y_)**2.0
         * (18 - 32*x_ + 12*x_**2.0 + 48*y_ - 36*x_*y_ + 27*y_**2.0)))

    return j


# TODO: Implement Booth's Function
def booth_func(x):
    """Booth's objective function.

    Only takes two dimensions and has a global minimum at
    :code:`f([1,3])`. Its domain is bounded between :code:`[-10, 10]`

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError('Booth function only takes two-dimensional input.')
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError('Input for Booth function must be within [-10, 10].')

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (x_ + 2 * y_ - 7)**2.0 + (2 * x_ + y_ - 5)**2.0

    return j


# TODO: Implement Bukin Function no. 6
def bukin6_func(x):
    """Bukin N. 6 Objective Function

    Only takes two dimensions and has a global minimum at
    :code:`f([-10,1])`. Its coordinates are bounded by:
        * x[:,0] must be within [-15, -5]
        * x[:,1] must be within [-3, 3]

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError('Bukin N. 6 function only takes two-dimensional '
                         'input.')
    if not np.logical_and(x[:, 0] >= -15, x[:, 0] <= -5).all():
        raise ValueError('x-coord for Bukin N. 6 function must be within '
                         '[-15, -5].')
    if not np.logical_and(x[:, 1] >= -3, x[:, 1] <= 3).all():
        raise ValueError('y-coord for Bukin N. 6 function must be within '
                         '[-3, 3].')

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 100 * np.sqrt(np.absolute(y_**2.0 - 0.01*x_**2.0)) + 0.01 * \
        np.absolute(x_ + 10)

    return j


def matyas_func(x):
    """Matyas objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-10,10]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
    """
    if not x.shape[1] == 2:
        raise IndexError('Matyas function only takes two-dimensional input.')
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError('Input for Matyas function must be within '
                         '[-10, 10].')

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = 0.26 * (x_**2.0 + y_**2.0) - 0.48 * x_ * y_

    return j


def levi_func(x):
    """Levi objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([1,1])`. Its coordinates are bounded within
    :code:`[-10,10]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError('Levi function only takes two-dimensional input.')
    if not np.logical_and(x >= -10, x <= 10).all():
        raise ValueError('Input for Levi function must be within [-10, 10].')

    mask = np.full(x.shape, False)
    mask[:, -1] = True
    masked_x = np.ma.array(x, mask=mask)

    w_ = 1 + (x - 1) / 4
    masked_w_ = np.ma.array(w_, mask=mask)
    d_ = x.shape[1] - 1

    j = (np.sin(np.pi * w_[:, 0])**2.0
         + ((masked_x - 1)**2.0).sum(axis=1)
         * (1 + 10 * np.sin(np.pi * (masked_w_).sum(axis=1) + 1)**2.0)
         + (w_[:, d_] - 1)**2.0
         * (1 + np.sin(2 * np.pi * w_[:, d_])**2.0))

    return j


def schaffer2_func(x):
    """Schaffer N.2 objective function

    Only takes two dimensions and has a global minimum at
    :code:`f([0,0])`. Its coordinates are bounded within
    :code:`[-100,100]`.

    Parameters
    ----------
    x : numpy.ndarray
        set of inputs of shape :code:`(n_particles, dimensions)`

    Returns
    -------
    numpy.ndarray
        computed cost of size :code:`(n_particles, )`

    Raises
    ------
    IndexError
        When the input dimensions is greater than what the function
        allows
    ValueError
        When the input is out of bounds with respect to the function
        domain
    """
    if not x.shape[1] == 2:
        raise IndexError('Schaffer N. 2 function only takes two-dimensional '
                         'input.')
    if not np.logical_and(x >= -100, x <= 100).all():
        raise ValueError('Input for Schaffer function must be within '
                         '[-100, 100].')

    x_ = x[:, 0]
    y_ = x[:, 1]
    j = (0.5
         + ((np.sin(x_**2.0 - y_**2.0)**2.0 - 0.5)
             / ((1 + 0.001 * (x_**2.0 + y_**2.0))**2.0)))

    return j
