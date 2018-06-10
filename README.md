![PySwarms Logo](docs/pyswarms-header.png)

---


[![PyPI version](https://badge.fury.io/py/pyswarms.svg)](https://badge.fury.io/py/pyswarms)
[![Build Status](https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=master)](https://travis-ci.org/ljvmiranda921/pyswarms)
[![Documentation Status](https://readthedocs.org/projects/pyswarms/badge/?version=master)](https://pyswarms.readthedocs.io/en/master/?badge=development)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/ljvmiranda921/pyswarms/master/LICENSE)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00433/status.svg)](https://doi.org/10.21105/joss.00433)
[![Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pyswarms/Issues)

PySwarms is an extensible research toolkit for particle swarm optimization
(PSO) in Python.

It is intended for swarm intelligence researchers, practitioners, and
students who prefer a high-level declarative interface for implementing PSO
in their problems. PySwarms enables basic optimization with PSO and
interaction with swarm optimizations. Check out more features below!

| Branch      | Status              | Documentation            | Description                   |
|-------------|---------------------|--------------------------|-------------------------------|
| master      | ![alt text][master] | ![alt text][master-docs] | Stable, official PyPI version |
| development | ![alt text][dev]    | ![alt text][dev-docs]    | Bleeding-edge, experimental   |

[master]: https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=master "Master"
[dev]: https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=development "Development"
[master-docs]: https://readthedocs.org/projects/pyswarms/badge/?version=master
[dev-docs]: https://readthedocs.org/projects/pyswarms/badge/?version=development

* **Free software:** MIT license
* **Documentation:** https://pyswarms.readthedocs.io.
* **Python versions:** 3.4 and above

## Features

* High-level module for Particle Swarm Optimization. For a list of all optimizers, check [this link].
* Built-in objective functions to test optimization algorithms.
* Plotting environment for cost histories and particle movement.
* Hyperparameter search tools to optimize swarm behaviour.
* (For Devs and Researchers): Highly-extensible API for implementing your own techniques.

[this link]: https://pyswarms.readthedocs.io/en/latest/features.html

## Dependencies
* numpy >= 1.13.0
* scipy >= 0.17.0
* matplotlib >= 1.3.1

## Installation

To install PySwarms, run this command in your terminal:

```shell
$ pip install pyswarms
```

This is the preferred method to install PySwarms, as it will always install
the most recent stable release.

In case you want to install the bleeding-edge version, clone this repo:

```shell
$ git clone -b development https://github.com/ljvmiranda921/pyswarms.git
```
and then run

```shell
$ cd pyswarms
$ python setup.py install
```

## Basic Usage

PySwarms provides a high-level implementation of various particle swarm
optimization algorithms. Thus, it aims to be user-friendly and customizable.
In addition, supporting modules can be used to help you in your optimization
problem.

### Optimizing a sphere function

You can import PySwarms as any other Python module,

```python
import pyswarms as ps
```

Suppose we want to find the minima of `f(x) = x^2` using global best
PSO, simply import the built-in sphere function,
`pyswarms.utils.functions.sphere_func()`, and the necessary optimizer:

```python
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
# Perform optimization
best_cost, best_pos = optimizer.optimize(fx.sphere_func, iters=100, verbose=3, print_step=25)
```
```s
>>> 2017-10-03 10:12:33,859 - pyswarms.single.global_best - INFO - Iteration 1/100, cost: 0.131244226714
>>> 2017-10-03 10:12:33,878 - pyswarms.single.global_best - INFO - Iteration 26/100, cost: 1.60297958653e-05
>>> 2017-10-03 10:12:33,893 - pyswarms.single.global_best - INFO - Iteration 51/100, cost: 1.60297958653e-05
>>> 2017-10-03 10:12:33,906 - pyswarms.single.global_best - INFO - Iteration 76/100, cost: 2.12638727702e-06
>>> 2017-10-03 10:12:33,921 - pyswarms.single.global_best - INFO - ================================
Optimization finished!
Final cost: 0.0000
Best value: [-0.0003521098028145481, -0.00045459382339127453]
```

This will run the optimizer for `100` iterations, then returns the best cost
and best position found by the swarm. In addition, you can also access
various histories by calling on properties of the class:

```python
# Obtain the cost history
optimizer.get_cost_history
# Obtain the position history
optimizer.get_pos_history
# Obtain the velocity history
optimizer.get_velocity_history
```

At the same time, you can also obtain the mean personal best and mean neighbor
history for local best PSO implementations. Simply call `mean_pbest_history`
and `optimizer.get_mean_neighbor_history` respectively.

### Hyperparameter search tools

PySwarms implements a grid search and random search technique to find the
best parameters for your optimizer. Setting them up is easy. In this example,
let's try using `pyswarms.utils.search.RandomSearch` to find the optimal
parameters for `LocalBestPSO` optimizer.

Here, we input a range, enclosed in tuples, to define the space in which the
parameters will be found. Thus, `(1,5)` pertains to a range from 1 to 5.

```python
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
```

This then returns the best score found during optimization, and the
hyperparameter options that enables it.

```s
>>> best_score
1.41978545901
>>> best_options['c1']
1.543556887693
>>> best_options['c2']
9.504769054771
```

### Plotting environments

It is also possible to plot optimizer performance for the sake of formatting.
The plotting environment is built on top of `matplotlib`, making it
highly-customizable.

The environment takes in the optimizer and its parameters, then performs a
fresh run to plot the cost and create animation.

```python
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
```

<img src="./docs/examples/output_9_0.png" width="460">

We can also plot the animation,

```python
plt_env.plot_particles2D(limits=((-1.2,1.2),(-1.2,1.2))
```

<img src="./docs/examples/output_3d.gif" width="460">


## Contributing

PySwarms is currently maintained by a single person (me!) with the aid of a
few but very helpful contributors. We would appreciate it if you can lend a
hand with the following:

* Find bugs and fix them
* Update documentation in docstrings
* Implement new optimizers to our collection
* Make utility functions more robust.

If you wish to contribute, check out our [contributing guide].
Moreover, you can also see the list of features that need some help in our
[Issues] page.

[contributing guide]: https://pyswarms.readthedocs.io/en/development/contributing.html
[Issues]: https://github.com/ljvmiranda921/pyswarms/issues

**Most importantly**, first time contributors are welcome to join! I try my
best to help you get started and enable you to make your first Pull Request!
Let's learn from each other!

## Credits

This project was inspired by the [pyswarm] module that performs PSO with
constrained support. The package was created with [Cookiecutter] and the
[`audreyr/cookiecutter-pypackage`] project template.

This is currently maintained by Lester James V. Miranda with other helpful
contributors:

* Carl-K ([`@Carl-K`](https://github.com/Carl-K))
* Siobhán Cronin ([`@SioKCronin`](https://github.com/SioKCronin))
* Andrew Jarcho ([`@jazcap53`](https://github.com/jazcap53))
* Charalampos Papadimitriou ([`@CPapadim`](https://github.com/CPapadim))
* Mamady Nabé ([`@mamadyonline`](https://github.com/mamadyonline))
* Erik ([`@slek120`](https://github.com/slek120))

[pyswarm]: https://github.com/tisimst/pyswarm
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[`audreyr/cookiecutter-pypackage`]: https://github.com/audreyr/cookiecutter-pypackage

## Cite us
Are you using PySwarms in your project or research? Please cite us!

* Miranda L.J., (2018). PySwarms: a research toolkit for Particle Swarm Optimization in Python. *Journal of Open Source Software*, 3(21), 433, https://doi.org/joss.00433

```bibtex
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
```

### Projects citing PySwarms
Not on the list? Ping us in the Issue Tracker!

* Gousios, Georgios. Lecture notes for the TU Delft TI3110TU course Algorithms and Data Structures. Accessed May 22, 2018. http://gousios.org/courses/algo-ds/book/string-distance.html#sop-example-using-pyswarms.
* Nandy, Abhishek, and Manisha Biswas., "Applying Python to Reinforcement Learning." *Reinforcement Learning*. Apress, Berkeley, CA, 2018. 89-128.
* Benedetti, Marcello, et al., "A generative modeling approach for benchmarking and training shallow quantum circuits." *arXiv preprint arXiv:1801.07686* (2018).
* Vrbančič et al., "NiaPy: Python microframework for building nature-inspired algorithms." Journal of Open Source Software, 3(23), 613, https://doi.org/10.21105/joss.00613

## Others
Like it? Love it? Leave us a star on [Github] to show your appreciation! 

[Github]: https://github.com/ljvmiranda921/pyswarms