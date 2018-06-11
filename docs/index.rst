.. image:: pyswarms-header.png
        :alt: PySwarms Logo
        :align: center

Welcome to PySwarms's documentation!
======================================

.. image:: https://badge.fury.io/py/pyswarms.svg
        :target: https://badge.fury.io/py/pyswarms
        :alt: PyPI Version

.. image:: https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=master
        :target: https://travis-ci.org/ljvmiranda921/pyswarms
        :alt: Build Status

.. image:: https://readthedocs.org/projects/pyswarms/badge/?version=latest
        :target: https://pyswarms.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg   
        :target: https://raw.githubusercontent.com/ljvmiranda921/pyswarms/master/LICENSE
        :alt: License

.. image:: http://joss.theoj.org/papers/10.21105/joss.00433/status.svg
        :target: https://doi.org/10.21105/joss.00433
        :alt: Citation

.. image:: https://badges.gitter.im/Join%20Chat.svg
        :target: https://gitter.im/pyswarms/Issues
        :alt: Gitter Chat

PySwarms is an extensible research toolkit for particle swarm optimization
(PSO) in Python.

It is intended for swarm intelligence researchers, practitioners, and
students who prefer a high-level declarative interface for implementing PSO
in their problems. PySwarms enables basic optimization with PSO and
interaction with swarm optimizations. Check out more features below!

+------------+------------+---------------+------------------------------+
| Branch     | Status     | Documentation | Description                  |
+============+============+===============+==============================+
| master     | |master|   | |master-docs| |Stable, official PyPI version |
+------------+------------+---------------+------------------------------+
| development| |dev|      | |dev-docs|    |Bleeding-edge, experimental   |
+------------+------------+---------------+------------------------------+

.. |dev| image:: https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=development
    :target: https://travis-ci.org/ljvmiranda921/pyswarms

.. |master| image:: https://travis-ci.org/ljvmiranda921/pyswarms.svg?branch=master
        :target: https://travis-ci.org/ljvmiranda921/pyswarms

.. |master-docs| image:: https://readthedocs.org/projects/pyswarms/badge/?version=master
        :target: https://pyswarms.readthedocs.io/en/master/

.. |dev-docs| image:: https://readthedocs.org/projects/pyswarms/badge/?version=development
        :target: https://pyswarms.readthedocs.io/en/development/

* **Free software:** MIT license
* **Github repository:** https://github.com/ljvmiranda921/pyswarms
* **Python versions:** 3.4, 3.5 and 3.6

Launching pad
-------------

* If you don't know what Particle Swarm Optimization is, read up this short `Introduction <http://pyswarms.readthedocs.io/en/latest/intro.html>`_! Then, if you plan to use PySwarms in your project, check the `Installation guide <https://pyswarms.readthedocs.io/en/latest/installation.html>`_ and `use-case examples <https://pyswarms.readthedocs.io/en/latest/examples/usecases.html>`_.

* If you are a researcher in the field of swarm intelligence, and would like to include your technique in our list of optimizers, check our `contributing <https://pyswarms.readthedocs.io/en/latest/contributing.html>`_ page to see how to implement your optimizer using the current base classes in the library.

* If you are an open-source contributor, and would like to help PySwarms grow, be sure to check our `Issues <https://github.com/ljvmiranda921/pyswarms/issues>`_ page in Github, and see the open issues with the tag `help-wanted <https://github.com/ljvmiranda921/pyswarms/labels/help%20wanted>`_. Moreover, we accommodate contributions from first-time contributors! Just check our `first-timers-only <https://github.com/ljvmiranda921/pyswarms/labels/first-timers-only>`_ tag for open issues (Don't worry! We're happy to help you make your first PR!).


.. toctree::
   :maxdepth: 2
   :caption: General

   intro
   features
   installation
   authors
   history

.. toctree::
   :maxdepth: 2
   :caption: Examples

   Use-cases <examples/usecases>

.. toctree::
   :maxdepth: 2
   :caption: Developer's Guide

   contributing
   dev.api
   dev.loop
   dev.optimizer

.. toctree::
   :caption: API Documentation

   api/_pyswarms.backend
   api/_pyswarms.base.classes
   api/_pyswarms.optimizers
   api/_pyswarms.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
