=======
History
=======

0.1.0 (2017-07-12)
------------------

* First release on PyPI.
* Includes primary optimization techniques such as global-best PSO and local-best PSO (# 1_) (# 3_).

.. _1: https://github.com/ljvmiranda921/pyswarms/issues/1
.. _3: https://github.com/ljvmiranda921/pyswarmsissues/3

0.1.1 (2017-07-25)
~~~~~~~~~~~~~~~~~~

* Patch on LocalBestPSO implementation. It seems that it's not returning the best value of the neighbors, this fixes the problem .
* **New feature:** Test functions for single-objective problems (# 6_) (# 10_) (PR# 14_). Contributed by `@Carl-K <https://github.com/Carl-K>`_. Thank you!

.. _6: https://github.com/ljvmiranda921/pyswarms/issues/6
.. _10: https://github.com/ljvmiranda921/pyswarms/pull/10
.. _14: https://github.com/ljvmiranda921/pyswarms/pull/14

0.1.2 (2017-08-02)
~~~~~~~~~~~~~~~~~~

* **New feature:** Binary Particle Swarm Optimization (# 7_) (# 17_). 
* Patch on Ackley function return error (# 22_).
* Improved documentation and unit tests (# 16_).

.. _7: https://github.com/ljvmiranda921/pyswarms/issues/7
.. _16: https://github.com/ljvmiranda921/pyswarms/issues/16
.. _17: https://github.com/ljvmiranda921/pyswarms/issues/17
.. _22: https://github.com/ljvmiranda921/pyswarms/issues/22


0.1.4 (2017-08-03)
~~~~~~~~~~~~~~~~~~

* Added a patch to fix :code:`pip` installation

0.1.5 (2017-08-11)
~~~~~~~~~~~~~~~~~~

* **New feature:** easy graphics environment. This new plotting environment makes it easier to plot the costs and swarm movement in 2-d or 3-d planes (# 30_) (PR# 31_).

.. _30: https://github.com/ljvmiranda921/pyswarms/issues/30
.. _31: https://github.com/ljvmiranda921/pyswarms/pull/31

0.1.6 (2017-09-24)
~~~~~~~~~~~~~~~~~~

* **New feature:** Native GridSearch and RandomSearch implementations for finding the best hyperparameters in controlling swarm behaviour (# 4_) (PR# 20_) (PR# 25_). Contributed by `@SioKCronin <https://github.com/SioKCronin>`_. Thanks a lot!
* Added tests for hyperparameter search techniques (# 27_) (PR# 28_) (PR# 40_). Contributed by `@jazcap53 <https://github.com/jazcap53>`_. Thank you so much!
* Updated structure of Base classes for higher extensibility

.. _4: https://github.com/ljvmiranda921/pyswarms/issues/4
.. _20: https://github.com/ljvmiranda921/pyswarms/pull/20
.. _25: https://github.com/ljvmiranda921/pyswarms/pull/25
.. _27: https://github.com/ljvmiranda921/pyswarms/issues/27
.. _28: https://github.com/ljvmiranda921/pyswarms/pull/28
.. _40: https://github.com/ljvmiranda921/pyswarms/pull/40

0.1.7 (2017-09-25)
~~~~~~~~~~~~~~~~~~

* Fixed patch on :code:`local_best.py`  and :code:`binary.py` (# 33_) (PR# 34_). Thanks for the awesome fix, `@CPapadim <https://github.com/CPapadim>`_!
* Git now ignores IPython notebook checkpoints

.. _33: https://github.com/ljvmiranda921/pyswarms/issues/33
.. _34: https://github.com/ljvmiranda921/pyswarms/pull/34

0.1.8 (2018-01-11)
~~~~~~~~~~~~~~~~~~

* PySwarms is now published on the Journal of Open Source Software (JOSS)! You can check the review here_. In addition, you can also find our paper in this link_. Thanks a lot to `@kyleniemeyer <https://github.com/kyleniemeyer>`_ and `@stsievert <https://github.com/stsievert>`_ for the thoughtful reviews and comments.

.. _here: https://github.com/openjournals/joss-reviews/issues/433
.. _link: http://joss.theoj.org/papers/235299884212b9223bce909631e3938b

0.1.9 (2018-04-20)
~~~~~~~~~~~~~~~~~~

* You can now set the initial position wherever you want (PR# 93_).
* Quick-fix for the rosenbrock function (PR# 98_).
* Tolerance can now be set to break during iteration (PR# 100_).

Thanks for all the wonderful Pull Requests, `@mamadyonline <https://github.com/mamadyonline>`_!

.. _93: https://github.com/ljvmiranda921/pyswarms/pull/93
.. _98: https://github.com/ljvmiranda921/pyswarms/pull/98
.. _100: https://github.com/ljvmiranda921/pyswarms/pull/100


0.2.0 (2018-06-11)
------------------

* New PySwarms backend. You can now build native swarm implementations using this module! (PR# 115_) (PR# 116_) (PR# 117_)
* Drop Python 2.7 version support. This package now supports Python 3.4 and up (PR# 114_).
* All tests were ported into pytest (PR# 113_).

.. _113: https://github.com/ljvmiranda921/pyswarms/pull/113
.. _114: https://github.com/ljvmiranda921/pyswarms/pull/114
.. _115: https://github.com/ljvmiranda921/pyswarms/pull/115
.. _116: https://github.com/ljvmiranda921/pyswarms/pull/116
.. _117: https://github.com/ljvmiranda921/pyswarms/pull/117