# -*- coding: utf-8 -*-

"""single_obj.py: collection of single-objective functions

Currently implements:

1. Multi-dimensional Objs
	- objective functions that can be computed along any
		given number of dimensions.
	1.1. Sphere function (Convex), `sphere_func()`
	1.2. Rastrigin function, `rastrigin_func()`
	1.3. Ackley's function, `ackley_func()`
	1.4. Rosenbrock's function, `rosenbrock_func()`

2. Two-dimensional Objs
	- objective functions that are only computed within
		two dimensions.

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

	Inputs:
		- x: (numpy.ndarray) set of inputs of shape: 
			(n_particles, dims)
	Returns: 
		- j: computed cost
	"""
	j = (x**2).sum(axis=1)
	
	return j


def rastrigin_func(x):
	"""Rastrigin objective function.

	Has a global minimum at f(0,0,...,0) with a search
	domain of -[-5.12, 5.12]

	Inputs:
		- x: (numpy.ndarray) set of inputs of shape: 
			(n_particles, dims)
	Returns: 
		- j: computed cost
	"""
	assert np.logical_and(x >= -5.12, x <= 5.12).all(), "Input for \
			Rastrigin function must be within [-5.12, 5.12]."
	
	n = x.shape[1]
	j = 10 * n + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(axis=1) 
	
	return j


def ackley_func(x):
	"""Ackley's objective function.

	Has a global minimum at f(0,0,...,0) with a search
	domain of [-32, 32]

	Inputs:
		- x: (numpy.ndarray) set of inputs of shape: 
			(n_particles, dims)
	Returns: 
		- j: computed cost
	"""
	assert np.logical_and(x >= -32, x <= 32).all(), "Input for \
			Rastrigin function must be within [-32, 32]."

	n = x.shape[1]
	j = -20 * np.exp(-0.2 * np.sqrt((1/n) * x**2))
		- np.exp((1/n) * np.cos(2 * np.pi * x))
		+ 20
		+ np.exp(1)

	return j

def rosenbrock_func(x):
	"""Rosenbrock objective function.

	Also known as the Rosenbrock's valley or Rosenbrock's banana
	function. Has a global minimum of np.ones(dims) where dims 
	is x.shape[1]. The search domain is [-inf, inf].

	Inputs:
		- x: (numpy.ndarray) set of inputs of shape: 
			(n_particles, dims)
	Returns: 
		- j: computed cost
	"""
	i = x.shape[0]
	j = 100 * np.square(x[1:] - x[:i-1]**2)
		+ x[:i-1]**2

	return j