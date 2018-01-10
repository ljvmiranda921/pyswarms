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

Particle swarm optimization (PSO) is a heuristic search technique that iteratively improves a set of candidate solutions given an objective measure of fitness [@kennedyIJCNN1995]. Although vanilla implementations of PSO can be found in some Python evolutionary algorithm toolboxes [@deapJMLR2012; @pagmo2017], a PSO-specific library that focuses on the said technique is still an open challenge.

PySwarms is a research toolkit for Particle Swarm Optimization (PSO) that provides a set of class primitives useful for solving continuous and combinatorial optimization problems. It follows a black-box approach, solving optimization tasks with few lines of code, yet allows a white-box framework with a consistent API for rapid prototyping of non-standard swarm models. In addition, benchmark objective functions and parameter-search tools are included to evaluate and improve swarm performance. It is intended for swarm intelligence researchers, practitioners, and students who would like a high-level declarative interface for implementing PSO in their problems.

The main design principle of the package is to balance (1) ease-of-use by providing a rich set of classes to solve optimization tasks, and (2) ease-of-experimentation by defining a consistent API to accommodate non-standard PSO implementations. In this context, PySwarms follows these core principles in its development:

- __Maintain a specific set of conventions that are manageable to understand.__ This enables repeatability in all implementations and creates a single framework where the API will be based. Thus, for a particular swarm $\mathcal{S}$, the particles are defined as an $m \times n$ matrix where $m$ corresponds to the number of particles, and $n$ to the number of dimensions in the search-space. Its fitness is then expressed as an $m$-dimensional array containing the value for each particle. 
- __Define a consistent API for all swarm implementations.__ A consistent API accommodates rapid prototyping of non-standard PSO implementations. As long as the user implements according to the API, all PySwarms capabilities are made available. It consists of an `init` method to initialize the swarm, an `update_position` and `update_velocity` rule to define update behaviour, and an `optimize` method that contains the evolutionary loop.
- __Provide a set of primitive classes for off-the-shelf PSO implementations.__ To deliver easy-access of PSO implementations for common optimization tasks, wrapper classes for standard global-best and local-best PSO are included. These implementations follow the same PySwarms API, and can even be built upon for more advanced applications.

Various features include:

- __Python implementation of standard PSO algorithms.__ Includes the classic global best and local best PSO [@kennedyIJCNN1995; @kennedyMHS1995], and binary PSO for discrete optimization [@kennedySMC1997]. These implementations are built natively in `numpy` [@numpycse; @scipyweb].
- __Built-in single objective functions for testing.__ Provides an array of single-objective functions to test optimizers. Includes simple variants such as the sphere function, up to complicated ones such as Beale and Rastrigin functions.
- __Plotting environment for cost and swarm animation.__ A wrapper built on top of `matplotlib` [@matplotlibcse] to conveniently plot costs and animate swarms (both in 2D and 3D) to assess performance and behavior.
- __Hyperparameter search tools.__ Implements both random and grid search to find optimal hyperparameters for controlling swarm behavior. 
- __Base classes for implementing your own optimizer.__ Provides single-objective base classes for researchers to rapidly prototype and implement their own optimizers.

Example use-cases involve: optimization of continuous and discrete functions, neural network training, feature selection, forward kinematics, and the like. Some of these use-cases are explained, with accompanying code, in the [Documentation](https://pyswarms.readthedocs.io/en/latest/). This package is actively maintained and developed by the author with the help of various contributors.

# References
