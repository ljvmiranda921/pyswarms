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

The optimizers include the actual PSO implementations for various tasks.
These include easy, off-the-shelf implementations include those that are
already considered as standard in literature. This may include the
classics such as global-best and local-best. Useful for quick-and-easy
optimization problems.

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