PySwarms package
=================

.. automodule:: pyswarms


Base Classes
------------

The base classes are inherited by various PSO implementations throughout the library.
It supports a simple skeleton to construct a customized PSO algorithm. 

.. toctree::

   pyswarms.base


Optimizers
-----------

The optimizers include the actual PSO implementations for various tasks. Generally,
there are two ways to implement an optimizer from this library: (1) as an easy
off-the-shelf algorithm, and (2) as an experimental custom-made algorithm.

1. Easy off-the-shelf implementations include those that are already considered as standard in literature. This may include the classics such as global-best and local-best. Their topologies are hardcoded already, and there is no need for prior set-up in order to use. This is useful for quick-and-easy optimization problems.

2. Experimental PSO algorithms are like standard PSO algorithms but without a defined topology. Instead, an object that inherits from a :code:`Topology` class is passed to an optimizer to define swarm behavior. Although the standard PSO implementations can be done through this, this is more experimental.

.. toctree::

   pyswarms.single
   pyswarms.discrete


Utilities
----------

This includes various utilities to help in optimization. In the future,
parameter search and plotting techniques will be incoroporated in this
module.

.. toctree::

   pyswarms.utils.functions