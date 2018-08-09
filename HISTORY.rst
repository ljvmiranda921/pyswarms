=======
History
=======

0.1.0 (2017-07-12)
------------------

* First release on PyPI.
* Includes primary optimization techniques such as global-best PSO and local-best PSO - `#1`_, `#3`_

.. _#1: https://github.com/ljvmiranda921/pyswarms/issues/1
.. _#3: https://github.com/ljvmiranda921/pyswarmsissues/3

0.1.1 (2017-07-25)
~~~~~~~~~~~~~~~~~~

* Patch on LocalBestPSO implementation. It seems that it's not returning the best value of the neighbors, this fixes the problem .
* **New feature:** Test functions for single-objective problems - `#6`_, `#10`_, `#14`_. Contributed by `@Carl-K <https://github.com/Carl-K>`_. Thank you!

.. _#6: https://github.com/ljvmiranda921/pyswarms/issues/6
.. _#10: https://github.com/ljvmiranda921/pyswarms/pull/10
.. _#14: https://github.com/ljvmiranda921/pyswarms/pull/14

0.1.2 (2017-08-02)
~~~~~~~~~~~~~~~~~~

* **New feature:** Binary Particle Swarm Optimization - `#7`_, `#17`_
* Patch on Ackley function return error - `#22`_
* Improved documentation and unit tests - `#16`_

.. _#7: https://github.com/ljvmiranda921/pyswarms/issues/7
.. _#16: https://github.com/ljvmiranda921/pyswarms/issues/16
.. _#17: https://github.com/ljvmiranda921/pyswarms/issues/17
.. _#22: https://github.com/ljvmiranda921/pyswarms/issues/22


0.1.4 (2017-08-03)
~~~~~~~~~~~~~~~~~~

* Added a patch to fix :code:`pip` installation

0.1.5 (2017-08-11)
~~~~~~~~~~~~~~~~~~

* **New feature:** easy graphics environment. This new plotting environment makes it easier to plot the costs and swarm movement in 2-d or 3-d planes - `#30`_, `#31`_

.. _#30: https://github.com/ljvmiranda921/pyswarms/issues/30
.. _#31: https://github.com/ljvmiranda921/pyswarms/pull/31

0.1.6 (2017-09-24)
~~~~~~~~~~~~~~~~~~

* **New feature:** Native GridSearch and RandomSearch implementations for finding the best hyperparameters in controlling swarm behaviour - `#4`_, `#20`_, `#25`_. Contributed by `@SioKCronin <https://github.com/SioKCronin>`_. Thanks a lot!
* Added tests for hyperparameter search techniques - `#27`_, `#28`_, `#40`_. Contributed by `@jazcap53 <https://github.com/jazcap53>`_. Thank you so much!
* Updated structure of Base classes for higher extensibility

.. _#4: https://github.com/ljvmiranda921/pyswarms/issues/4
.. _#20: https://github.com/ljvmiranda921/pyswarms/pull/20
.. _#25: https://github.com/ljvmiranda921/pyswarms/pull/25
.. _#27: https://github.com/ljvmiranda921/pyswarms/issues/27
.. _#28: https://github.com/ljvmiranda921/pyswarms/pull/28
.. _#40: https://github.com/ljvmiranda921/pyswarms/pull/40

0.1.7 (2017-09-25)
~~~~~~~~~~~~~~~~~~

* Fixed patch on :code:`local_best.py`  and :code:`binary.py` - `#33`_, `#34`_. Thanks for the awesome fix, `@CPapadim <https://github.com/CPapadim>`_!
* Git now ignores IPython notebook checkpoints

.. _#33: https://github.com/ljvmiranda921/pyswarms/issues/33
.. _#34: https://github.com/ljvmiranda921/pyswarms/pull/34

0.1.8 (2018-01-11)
~~~~~~~~~~~~~~~~~~

* PySwarms is now published on the Journal of Open Source Software (JOSS)! You can check the review here_. In addition, you can also find our paper in this link_. Thanks a lot to `@kyleniemeyer <https://github.com/kyleniemeyer>`_ and `@stsievert <https://github.com/stsievert>`_ for the thoughtful reviews and comments.

.. _here: https://github.com/openjournals/joss-reviews/issues/433
.. _link: http://joss.theoj.org/papers/235299884212b9223bce909631e3938b

0.1.9 (2018-04-20)
~~~~~~~~~~~~~~~~~~

* You can now set the initial position wherever you want - `#93`_
* Quick-fix for the Rosenbrock function - `#98`_
* Tolerance can now be set to break during iteration - `#100`_

Thanks for all the wonderful Pull Requests, `@mamadyonline <https://github.com/mamadyonline>`_!

.. _#93: https://github.com/ljvmiranda921/pyswarms/pull/93
.. _#98: https://github.com/ljvmiranda921/pyswarms/pull/98
.. _#100: https://github.com/ljvmiranda921/pyswarms/pull/100


0.2.0 (2018-06-11)
------------------

* New PySwarms backend. You can now build native swarm implementations using this module! -  `#115`_, `#116`_, `#117`_
* Drop Python 2.7 version support. This package now supports Python 3.4 and up - `#113`_
* All tests were ported into pytest - `#114`_

.. _#113: https://github.com/ljvmiranda921/pyswarms/pull/113
.. _#114: https://github.com/ljvmiranda921/pyswarms/pull/114
.. _#115: https://github.com/ljvmiranda921/pyswarms/pull/115
.. _#116: https://github.com/ljvmiranda921/pyswarms/pull/116
.. _#117: https://github.com/ljvmiranda921/pyswarms/pull/117


0.2.1 (2018-06-27)
~~~~~~~~~~~~~~~~~~

* Fix sigmoid function in BinaryPSO - `#145`_. Thanks a lot `@ThomasCES <https://github.com/ThomasCES>`_!

.. _#145: https://github.com/ljvmiranda921/pyswarms/pull/145

0.3.0 (2018-08-10)
------------------

* New topologies: Pyramid, Random, and Von Neumann. More ways for your particles to interact! - `#176`_, `#177`_, `#155`_, `#142`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* New GeneralOptimizer algorithm that allows you to switch-out topologies for your optimization needs - `#151`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* All topologies now have a static attribute. Neigbors can now be set initially or computed dynamically - `#164`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* New single-objective functions - `#168`_. Awesome work, `@jayspeidell <https://github.com/jayspeidell>`_!
* New tutorial on Inverse Kinematics using Particle Swarm Optimization - `#141`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* New plotters module for visualization. The environment module is now deprecated - `#135`_
* Keyword arguments can now be passed in the :code:`optimize()` method for your custom objective functions - `#144`_. Great job, `@bradahoward <https://github.com/bradahoward>`_

.. _#135: https://github.com/ljvmiranda921/pyswarms/pull/135
.. _#141: https://github.com/ljvmiranda921/pyswarms/pull/141
.. _#142: https://github.com/ljvmiranda921/pyswarms/pull/142
.. _#144: https://github.com/ljvmiranda921/pyswarms/pull/144
.. _#151: https://github.com/ljvmiranda921/pyswarms/pull/151
.. _#155: https://github.com/ljvmiranda921/pyswarms/pull/155
.. _#164: https://github.com/ljvmiranda921/pyswarms/pull/164
.. _#168: https://github.com/ljvmiranda921/pyswarms/pull/168
.. _#176: https://github.com/ljvmiranda921/pyswarms/pull/176
.. _#177: https://github.com/ljvmiranda921/pyswarms/pull/177
