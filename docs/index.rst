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

.. image:: https://zenodo.org/badge/97002861.svg
        :target: https://zenodo.org/badge/latestdoi/97002861
        :alt: Citation

.. image:: https://badges.gitter.im/Join%20Chat.svg
        :target: https://gitter.im/pyswarms/Issues
        :alt: Gitter Chat

PySwarms is a an extensible research toolkit for particle swarm optimization (PSO) in Python.

It is intended for swarm intelligence researchers, practitioners, and students who would like a high-level declarative interface of implementing PSO in their problems. PySwarms both allows basic optimization with PSO and interaction with swarm optimizations. Interaction is enabled due to object primitives provided by the package for optimization. This makes PySwarms useful for researchers or students.

* **Free software:** MIT license
* **Github repository:** https://github.com/ljvmiranda921/pyswarms
* **Python versions:** 2.7, 3.4, 3.5 and above

Launching pad
-------------

* If you don't know what Particle Swarm Optimization is, read up this short `Introduction <http://pyswarms.readthedocs.io/en/latest/intro.html>`_! Then, if you plan to use PySwarms in your project, check the `Installation guide <https://pyswarms.readthedocs.io/en/latest/installation.html>`_ and the `use-case examples <https://pyswarms.readthedocs.io/en/latest/examples/usecases.html>`_ in this documentation.

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
   contributing.optimizer

.. toctree::
   :caption: API Documentation

   api/_pyswarms.base.classes
   api/_pyswarms.optimizers
   api/_pyswarms.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
