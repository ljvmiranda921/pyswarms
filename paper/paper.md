---
title: 'PySwarms: a research toolkit for Particle Swarm Optimization in Python'
tags:
  - particle swarm optimization
  - swarm intelligence
  - optimization algorithms
authors:
  - name: Lester James V. Miranda
    orcid: 0000-0002-7872-6464
    affiliation: 1
affiliations:
  - name: Waseda University
    index: 1
date: 07 October 2017
bibliography: paper.bib
---

# Summary

_PySwarms_ is a research toolkit for Particle Swarm Optimization (PSO) in Python. It is intended for swarm intelligence researchers, practitioners, and students who would like a high-level declarative interface of implementing PSO in their problems. PySwarms both allows basic optimization with PSO and interaction with swarm optimizations. Interaction is enabled due to object primitives provided by the package for optimization. This makes PySwarms useful for researchers or students.

Various features include:

- __Python implementation of standard PSO algorithms.__ Includes the classic global best and local best PSO [@kennedyIJCNN1995] [@kennedyMHS1995], and binary PSO for discrete optimization [@kennedySMC1997]. These implementations are built natively in `numpy`.
- __Built-in single objective functions for testing.__ Provides an array of single-objective functions to test optimizers. Includes simple variants such as the sphere function, up to complicated ones such as Beale and Rastrigin functions.
- __Plotting environment for cost and swarm animation.__ A wrapper built on top of `matplotlib` to conveniently plot costs and animate swarms (both in 2D and 3D) to assess performance and behavior.
- __Hyperparameter search tools.__ Implements both random and grid search to find optimal hyperparameters for controlling swarm behavior. 
- __Base classes for implementing your own optimizer.__ Provides single-objective base classes for researchers to rapidly prototype and implement their own optimizers.

This package is actively maintained and developed by Lester James V. Miranda with the help of various contributors. The documentation and use-case examples can be seen in the Documentation (https://pyswarms.readthedocs.io/en/latest/).

# References
