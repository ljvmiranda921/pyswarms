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

PySwarms is a research toolkit for Particle Swarm Optimization (PSO) in
Python. This package serves both researchers and practitioners:
first as an environment to test and compare standard PSO algorithms, and
second as a quick-and-easy tool to perform optimization using PSO.

Its high-level features include the following:

- __Python implementation of standard PSO algorithms.__ Includes the classic global best and local best PSO [@kennedyIJCNN1995] [@kennedyMHS1995], and binary PSO for discrete optimization [@kennedySMC1997].
- __Built-in single objective functions for testing.__ Provides an array of single-objective functions to test optimizers.
- __Hyperparameter search tools.__ Implements both random search and grid search to find optimal hyperparameters for controlling swarm behavior.
- __Plotting environment for cost and swarm animation.__ An easy wrapper for plotting costs and animating swarms (both in 2D and 3D) to assess swarm performance.
- __Base classes for implementing your own optimizer.__ Provides single-objective base classes for researchers to implement their own optimizers.

# References