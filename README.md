![PySwarms Logo](https://i.imgur.com/eX8oqPQ.png)
---


[![PyPI version](https://badge.fury.io/py/pyswarms.svg)](https://badge.fury.io/py/pyswarms)
[![Build Status](https://dev.azure.com/ljvmiranda/ljvmiranda/_apis/build/status/ljvmiranda921.pyswarms?branchName=master)](https://dev.azure.com/ljvmiranda/ljvmiranda/_build/latest?definitionId=1&branchName=master)
[![Documentation Status](https://readthedocs.org/projects/pyswarms/badge/?version=latest)](https://pyswarms.readthedocs.io/en/master/?badge=master)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg )](https://raw.githubusercontent.com/ljvmiranda921/pyswarms/master/LICENSE)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00433/status.svg)](https://doi.org/10.21105/joss.00433)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pyswarms/Issues)

PySwarms is an extensible research toolkit for particle swarm optimization
(PSO) in Python.

It is intended for swarm intelligence researchers, practitioners, and
students who prefer a high-level declarative interface for implementing PSO
in their problems. PySwarms enables basic optimization with PSO and
interaction with swarm optimizations. Check out more features below!

* **Free software:** MIT license
* **Documentation:** https://pyswarms.readthedocs.io.
* **Python versions:** 3.5 and above

## Features

* High-level module for Particle Swarm Optimization. For a list of all optimizers, check [this link].
* Built-in objective functions to test optimization algorithms.
* Plotting environment for cost histories and particle movement.
* Hyperparameter search tools to optimize swarm behaviour.
* (For Devs and Researchers): Highly-extensible API for implementing your own techniques.

[this link]: https://pyswarms.readthedocs.io/en/latest/features.html

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

## Running in a Vagrant Box

To run PySwarms in a Vagrant Box, install Vagrant by going to 
https://www.vagrantup.com/downloads.html and downloading the proper packaged from the Hashicorp website. 

Afterward, run the following command in the project directory:

```shell
$ vagrant provision
$ vagrant up
$ vagrant ssh
```
Now you're ready to develop your contributions in a premade virtual environment. 

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
`pyswarms.utils.functions.sphere()`, and the necessary optimizer:

```python
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
# Perform optimization
best_cost, best_pos = optimizer.optimize(fx.sphere, iters=100)
```

![Sphere Optimization](https://i.imgur.com/5LtjROf.gif)

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
            dimensions=20, options=options, objective_func=fx.sphere,
            iters=10, n_selection_iters=100)

best_score, best_options = g.search()
```

This then returns the best score found during optimization, and the
hyperparameter options that enable it.

```s
>>> best_score
1.41978545901
>>> best_options['c1']
1.543556887693
>>> best_options['c2']
9.504769054771
```

### Swarm visualization

It is also possible to plot optimizer performance for the sake of formatting.
The plotters module is built on top of `matplotlib`, making it
highly-customizable.


```python
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
import matplotlib.pyplot as plt
# Set-up optimizer
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)
optimizer.optimize(fx.sphere, iters=100)
# Plot the cost
plot_cost_history(optimizer.cost_history)
plt.show()
```

![CostHistory](https://i.imgur.com/19Iuz4B.png)

We can also plot the animation...

```python
from pyswarms.utils.plotters.formatters import Mesher, Designer
# Plot the sphere function's mesh for better plots
m = Mesher(func=fx.sphere,
           limits=[(-1,1), (-1,1)])
# Adjust figure limits
d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)],
             label=['x-axis', 'y-axis', 'z-axis'])
```

In 2D,

```python
plot_contour(pos_history=optimizer.pos_history, mesher=m, designer=d, mark=(0,0))
```

![Contour](https://i.imgur.com/H3YofJ6.gif)

Or in 3D!

```python
pos_history_3d = m.compute_history_3d(optimizer.pos_history) # preprocessing
animation3d = plot_surface(pos_history=pos_history_3d,
                           mesher=m, designer=d,
                           mark=(0,0,0))    
```

![Surface](https://i.imgur.com/kRb61Hx.gif)

## Contributing

PySwarms is currently maintained by a small yet dedicated team:
- Lester James V. Miranda ([@ljvmiranda921](https://github.com/ljvmiranda921))
- Siobhán K. Cronin ([@SioKCronin](https://github.com/SioKCronin))
- Aaron Moser ([@whzup](https://github.com/whzup))
- Steven Beardwell ([@stevenbw](https://github.com/stevenbw))

And we would appreciate it if you can lend a hand with the following:

* Find bugs and fix them
* Update documentation in docstrings
* Implement new optimizers to our collection
* Make utility functions more robust.

We would also like to acknowledge [all our
contributors](http://pyswarms.readthedocs.io/en/latest/authors.html), past and
present, for making this project successful!

If you wish to contribute, check out our [contributing guide].
Moreover, you can also see the list of features that need some help in our
[Issues] page.

[contributing guide]: https://pyswarms.readthedocs.io/en/development/contributing.html
[Issues]: https://github.com/ljvmiranda921/pyswarms/issues

**Most importantly**, first-time contributors are welcome to join! I try my
best to help you get started and enable you to make your first Pull Request!
Let's learn from each other!

## Credits

This project was inspired by the [pyswarm] module that performs PSO with
constrained support. The package was created with [Cookiecutter] and the
[`audreyr/cookiecutter-pypackage`] project template.

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
* Häse, Florian, et al. "Phoenics: A Bayesian optimizer for chemistry." *ACS Central Science.* 4.9 (2018): 1134-1145. 
* Szynkiewicz, Pawel. "A Comparative Study of PSO and CMA-ES Algorithms on Black-box Optimization Benchmarks." *Journal of Telecommunications and Information Technology* 4 (2018): 5.
* Mistry, Miten, et al. "Mixed-Integer Convex Nonlinear Optimization with Gradient-Boosted Trees Embedded." Imperial College London (2018).
* Vishwakarma, Gaurav. *Machine Learning Model Selection for Predicting Properties of High Refractive Index Polymers* Dissertation. State University of New York at Buffalo, 2018.
* Uluturk Ismail, et al. "Efficient 3D Placement of Access Points in an Aerial Wireless Network." *2019 16th IEEE Anual Consumer Communications and Networking Conference (CCNC)* IEEE (2019): 1-7.

## Others
Like it? Love it? Leave us a star on [Github] to show your appreciation! 

[Github]: https://github.com/ljvmiranda921/pyswarms

## Contributors

Thanks goes to these wonderful people ([emoji key](https://github.com/all-contributors/all-contributors#emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
<table><tr><td align="center"><a href="https://github.com/whzup"><img src="https://avatars3.githubusercontent.com/u/39431903?v=4" width="100px;" alt="Aaron"/><br /><sub><b>Aaron</b></sub></a><br /><a href="#maintenance-whzup" title="Maintenance">🚧</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=whzup" title="Code">💻</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=whzup" title="Documentation">📖</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=whzup" title="Tests">⚠️</a> <a href="#ideas-whzup" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-whzup" title="Reviewed Pull Requests">👀</a></td><td align="center"><a href="https://github.com/Carl-K"><img src="https://avatars2.githubusercontent.com/u/13661469?v=4" width="100px;" alt="Carl-K"/><br /><sub><b>Carl-K</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=Carl-K" title="Code">💻</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=Carl-K" title="Tests">⚠️</a></td><td align="center"><a href="http://www.siobhankcronin.com/"><img src="https://avatars2.githubusercontent.com/u/19956669?v=4" width="100px;" alt="Siobhán K Cronin"/><br /><sub><b>Siobhán K Cronin</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=SioKCronin" title="Code">💻</a> <a href="#maintenance-SioKCronin" title="Maintenance">🚧</a> <a href="#ideas-SioKCronin" title="Ideas, Planning, & Feedback">🤔</a></td><td align="center"><a href="http://andrewjarcho.com"><img src="https://avatars3.githubusercontent.com/u/1452951?v=4" width="100px;" alt="Andrew Jarcho"/><br /><sub><b>Andrew Jarcho</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=jazcap53" title="Tests">⚠️</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=jazcap53" title="Code">💻</a></td><td align="center"><a href="https://github.com/mamadyonline"><img src="https://avatars1.githubusercontent.com/u/20543370?v=4" width="100px;" alt="Mamady"/><br /><sub><b>Mamady</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=mamadyonline" title="Code">💻</a></td><td align="center"><a href="https://github.com/jayspeidell"><img src="https://avatars3.githubusercontent.com/u/26357788?v=4" width="100px;" alt="Jay Speidell"/><br /><sub><b>Jay Speidell</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=jayspeidell" title="Code">💻</a></td><td align="center"><a href="https://github.com/slek120"><img src="https://avatars2.githubusercontent.com/u/3589574?v=4" width="100px;" alt="Eric"/><br /><sub><b>Eric</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/issues?q=author%3Aslek120" title="Bug reports">🐛</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=slek120" title="Code">💻</a></td></tr><tr><td align="center"><a href="https://github.com/CPapadim"><img src="https://avatars1.githubusercontent.com/u/13984473?v=4" width="100px;" alt="CPapadim"/><br /><sub><b>CPapadim</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/issues?q=author%3ACPapadim" title="Bug reports">🐛</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=CPapadim" title="Code">💻</a></td><td align="center"><a href="https://github.com/dfhljf"><img src="https://avatars1.githubusercontent.com/u/7887642?v=4" width="100px;" alt="JiangHui"/><br /><sub><b>JiangHui</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=dfhljf" title="Code">💻</a></td><td align="center"><a href="https://github.com/nik1082"><img src="https://avatars3.githubusercontent.com/u/17260188?v=4" width="100px;" alt="Jericho Arcelao"/><br /><sub><b>Jericho Arcelao</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=nik1082" title="Code">💻</a></td><td align="center"><a href="http://www.jdbohrman.xyz"><img src="https://avatars2.githubusercontent.com/u/27848025?v=4" width="100px;" alt="James D. Bohrman"/><br /><sub><b>James D. Bohrman</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=jdbohrman" title="Code">💻</a></td><td align="center"><a href="https://github.com/bradahoward"><img src="https://avatars2.githubusercontent.com/u/24660861?v=4" width="100px;" alt="bradahoward"/><br /><sub><b>bradahoward</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=bradahoward" title="Code">💻</a></td><td align="center"><a href="https://github.com/ThomasCES"><img src="https://avatars2.githubusercontent.com/u/18325841?v=4" width="100px;" alt="ThomasCES"/><br /><sub><b>ThomasCES</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=ThomasCES" title="Code">💻</a></td><td align="center"><a href="https://github.com/danielcorreia96"><img src="https://avatars0.githubusercontent.com/u/17486065?v=4" width="100px;" alt="Daniel Correia"/><br /><sub><b>Daniel Correia</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/issues?q=author%3Adanielcorreia96" title="Bug reports">🐛</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=danielcorreia96" title="Code">💻</a></td></tr><tr><td align="center"><a href="https://github.com/fluencer"><img src="https://avatars3.githubusercontent.com/u/6614307?v=4" width="100px;" alt="fluencer"/><br /><sub><b>fluencer</b></sub></a><br /><a href="#example-fluencer" title="Examples">💡</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=fluencer" title="Documentation">📖</a></td><td align="center"><a href="https://github.com/miguelcocruz"><img src="https://avatars0.githubusercontent.com/u/45034603?v=4" width="100px;" alt="miguelcocruz"/><br /><sub><b>miguelcocruz</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=miguelcocruz" title="Documentation">📖</a> <a href="#example-miguelcocruz" title="Examples">💡</a></td><td align="center"><a href="https://github.com/stevenbw"><img src="https://avatars1.githubusercontent.com/u/46458390?v=4" width="100px;" alt="Steven Beardwell"/><br /><sub><b>Steven Beardwell</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=stevenbw" title="Code">💻</a> <a href="#maintenance-stevenbw" title="Maintenance">🚧</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=stevenbw" title="Documentation">📖</a> <a href="#ideas-stevenbw" title="Ideas, Planning, & Feedback">🤔</a></td><td align="center"><a href="https://github.com/ndngo"><img src="https://avatars1.githubusercontent.com/u/16291290?v=4" width="100px;" alt="Nathaniel Ngo"/><br /><sub><b>Nathaniel Ngo</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=ndngo" title="Documentation">📖</a></td><td align="center"><a href="https://github.com/Aneal-Sharma"><img src="https://avatars1.githubusercontent.com/u/19873846?v=4" width="100px;" alt="Aneal Sharma"/><br /><sub><b>Aneal Sharma</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=Aneal-Sharma" title="Documentation">📖</a></td><td align="center"><a href="https://github.com/citomcclure"><img src="https://avatars2.githubusercontent.com/u/38021988?v=4" width="100px;" alt="Chris McClure"/><br /><sub><b>Chris McClure</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=citomcclure" title="Documentation">📖</a> <a href="#example-citomcclure" title="Examples">💡</a></td><td align="center"><a href="http://se4.space/"><img src="https://avatars2.githubusercontent.com/u/42605993?v=4" width="100px;" alt="Christopher Angell"/><br /><sub><b>Christopher Angell</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=ctangell" title="Documentation">📖</a></td></tr><tr><td align="center"><a href="https://github.com/Kutim"><img src="https://avatars3.githubusercontent.com/u/8309533?v=4" width="100px;" alt="Kutim"/><br /><sub><b>Kutim</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/issues?q=author%3AKutim" title="Bug reports">🐛</a></td><td align="center"><a href="https://github.com/ichbinjakes"><img src="https://avatars1.githubusercontent.com/u/20906800?v=4" width="100px;" alt="Jake Souter"/><br /><sub><b>Jake Souter</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/issues?q=author%3Aichbinjakes" title="Bug reports">🐛</a> <a href="https://github.com/ljvmiranda921/pyswarms/commits?author=ichbinjakes" title="Code">💻</a></td><td align="center"><a href="https://github.com/IanBoyanZhang"><img src="https://avatars3.githubusercontent.com/u/4110995?v=4" width="100px;" alt="Ian Zhang"/><br /><sub><b>Ian Zhang</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=IanBoyanZhang" title="Documentation">📖</a> <a href="#example-IanBoyanZhang" title="Examples">💡</a></td><td align="center"><a href="https://www.zachariahcarmichael.com/"><img src="https://avatars2.githubusercontent.com/u/20629897?v=4" width="100px;" alt="Zach"/><br /><sub><b>Zach</b></sub></a><br /><a href="https://github.com/ljvmiranda921/pyswarms/commits?author=craymichael" title="Documentation">📖</a></td></tr></table>

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
