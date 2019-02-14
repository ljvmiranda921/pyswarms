![PySwarms Logo](https://i.imgur.com/eX8oqPQ.png)
---


[![PyPI version](https://badge.fury.io/py/pyswarms.svg)](https://badge.fury.io/py/pyswarms)
[![Build Status](https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=master)](https://travis-ci.org/ljvmiranda921/pyswarms)
[![Documentation Status](https://readthedocs.org/projects/pyswarms/badge/?version=master)](https://pyswarms.readthedocs.io/en/master/?badge=development)
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
* **Python versions:** 3.5 and above

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

## Running in a Vagrant Box

To run PySwarms in a Vagrant Box, install Vagrant by going to 
https://www.vagrantup.com/downloads.html and downloading the proper packaged from the Hashicorp website. 

Afterward, run the following command in the project directory:

```shell
$ vagrant provision
$ vagrant up
$ vagrant ssh
```
Now you're ready to develop your contributions in a premade virtual enviroment. 

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
best_cost, best_pos = optimizer.optimize(fx.sphere, iters=100, verbose=3, print_step=25)
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
            dimensions=20, options=options, objective_func=fx.sphere,
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

### Swarm visualization

It is also possible to plot optimizer performance for the sake of formatting.
The plotters moule is built on top of `matplotlib`, making it
highly-customizable.


```python
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
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
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters.formatters import Designer
# Plot the sphere function's mesh for better plots
m = Mesher(func=fx.sphere_func,
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

**Most importantly**, first time contributors are welcome to join! I try my
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

## Others
Like it? Love it? Leave us a star on [Github] to show your appreciation! 

[Github]: https://github.com/ljvmiranda921/pyswarms

## Contributors

Thanks goes to these wonderful people ([emoji key](https://github.com/all-contributors/all-contributors#emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore -->
| [<img src="https://avatars0.githubusercontent.com/u/17486065?v=4" width="100px;" alt="Daniel Correia"/><br /><sub><b>Daniel Correia</b></sub>](https://github.com/danielcorreia96)<br />[🐛](https://github.com/ljvmiranda921/pyswarms/issues?q=author%3Adanielcorreia96 "Bug reports") [💻](https://github.com/ljvmiranda921/pyswarms/commits?author=danielcorreia96 "Code") | [<img src="https://avatars3.githubusercontent.com/u/39431903?v=4" width="100px;" alt="Aaron"/><br /><sub><b>Aaron</b></sub>](https://github.com/whzup)<br />[🚧](#maintenance-whzup "Maintenance") [💻](https://github.com/ljvmiranda921/pyswarms/commits?author=whzup "Code") [📖](https://github.com/ljvmiranda921/pyswarms/commits?author=whzup "Documentation") [⚠️](https://github.com/ljvmiranda921/pyswarms/commits?author=whzup "Tests") [🤔](#ideas-whzup "Ideas, Planning, & Feedback") [👀](#review-whzup "Reviewed Pull Requests") | [<img src="https://avatars2.githubusercontent.com/u/13661469?v=4" width="100px;" alt="Carl-K"/><br /><sub><b>Carl-K</b></sub>](https://github.com/Carl-K)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=Carl-K "Code") [⚠️](https://github.com/ljvmiranda921/pyswarms/commits?author=Carl-K "Tests") | [<img src="https://avatars2.githubusercontent.com/u/19956669?v=4" width="100px;" alt="Siobhán K Cronin"/><br /><sub><b>Siobhán K Cronin</b></sub>](http://www.siobhankcronin.com/)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=SioKCronin "Code") [🚧](#maintenance-SioKCronin "Maintenance") [🤔](#ideas-SioKCronin "Ideas, Planning, & Feedback") | [<img src="https://avatars3.githubusercontent.com/u/1452951?v=4" width="100px;" alt="Andrew Jarcho"/><br /><sub><b>Andrew Jarcho</b></sub>](http://andrewjarcho.com)<br />[⚠️](https://github.com/ljvmiranda921/pyswarms/commits?author=jazcap53 "Tests") [💻](https://github.com/ljvmiranda921/pyswarms/commits?author=jazcap53 "Code") | [<img src="https://avatars1.githubusercontent.com/u/20543370?v=4" width="100px;" alt="Mamady"/><br /><sub><b>Mamady</b></sub>](https://github.com/mamadyonline)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=mamadyonline "Code") | [<img src="https://avatars3.githubusercontent.com/u/26357788?v=4" width="100px;" alt="Jay Speidell"/><br /><sub><b>Jay Speidell</b></sub>](https://github.com/jayspeidell)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=jayspeidell "Code") |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [<img src="https://avatars2.githubusercontent.com/u/3589574?v=4" width="100px;" alt="Eric"/><br /><sub><b>Eric</b></sub>](https://github.com/slek120)<br />[🐛](https://github.com/ljvmiranda921/pyswarms/issues?q=author%3Aslek120 "Bug reports") [💻](https://github.com/ljvmiranda921/pyswarms/commits?author=slek120 "Code") | [<img src="https://avatars1.githubusercontent.com/u/13984473?v=4" width="100px;" alt="CPapadim"/><br /><sub><b>CPapadim</b></sub>](https://github.com/CPapadim)<br />[🐛](https://github.com/ljvmiranda921/pyswarms/issues?q=author%3ACPapadim "Bug reports") [💻](https://github.com/ljvmiranda921/pyswarms/commits?author=CPapadim "Code") | [<img src="https://avatars1.githubusercontent.com/u/7887642?v=4" width="100px;" alt="JiangHui"/><br /><sub><b>JiangHui</b></sub>](https://github.com/dfhljf)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=dfhljf "Code") | [<img src="https://avatars3.githubusercontent.com/u/17260188?v=4" width="100px;" alt="Jericho Arcelao"/><br /><sub><b>Jericho Arcelao</b></sub>](https://github.com/nik1082)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=nik1082 "Code") | [<img src="https://avatars2.githubusercontent.com/u/27848025?v=4" width="100px;" alt="James D. Bohrman"/><br /><sub><b>James D. Bohrman</b></sub>](http://www.jdbohrman.xyz)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=jdbohrman "Code") | [<img src="https://avatars2.githubusercontent.com/u/24660861?v=4" width="100px;" alt="bradahoward"/><br /><sub><b>bradahoward</b></sub>](https://github.com/bradahoward)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=bradahoward "Code") | [<img src="https://avatars2.githubusercontent.com/u/18325841?v=4" width="100px;" alt="ThomasCES"/><br /><sub><b>ThomasCES</b></sub>](https://github.com/ThomasCES)<br />[💻](https://github.com/ljvmiranda921/pyswarms/commits?author=ThomasCES "Code") |
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
