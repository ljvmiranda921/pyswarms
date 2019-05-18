=======
History
=======

0.1.0 (2017-07-12)
------------------

* First release on PyPI.
* **NEW:** Includes primary optimization techniques such as global-best PSO and local-best PSO - `#1`_, `#3`_

.. _#1: https://github.com/ljvmiranda921/pyswarms/issues/1
.. _#3: https://github.com/ljvmiranda921/pyswarmsissues/3

0.1.1 (2017-07-25)
~~~~~~~~~~~~~~~~~~

* **FIX:** Patch on LocalBestPSO implementation. It seems that it's not returning the best value of the neighbors, this fixes the problem .
* **NEW:** Test functions for single-objective problems - `#6`_, `#10`_, `#14`_. Contributed by `@Carl-K <https://github.com/Carl-K>`_. Thank you!

.. _#6: https://github.com/ljvmiranda921/pyswarms/issues/6
.. _#10: https://github.com/ljvmiranda921/pyswarms/pull/10
.. _#14: https://github.com/ljvmiranda921/pyswarms/pull/14

0.1.2 (2017-08-02)
~~~~~~~~~~~~~~~~~~

* **NEW:** Binary Particle Swarm Optimization - `#7`_, `#17`_
* **FIX:**  Fix on Ackley function return error - `#22`_
* **IMPROVED:** Documentation and unit tests - `#16`_

.. _#7: https://github.com/ljvmiranda921/pyswarms/issues/7
.. _#16: https://github.com/ljvmiranda921/pyswarms/issues/16
.. _#17: https://github.com/ljvmiranda921/pyswarms/issues/17
.. _#22: https://github.com/ljvmiranda921/pyswarms/issues/22


0.1.4 (2017-08-03)
~~~~~~~~~~~~~~~~~~

* **FIX:** Added a patch to fix :code:`pip` installation

0.1.5 (2017-08-11)
~~~~~~~~~~~~~~~~~~

* **NEW:** easy graphics environment. This new plotting environment makes it easier to plot the costs and swarm movement in 2-d or 3-d planes - `#30`_, `#31`_

.. _#30: https://github.com/ljvmiranda921/pyswarms/issues/30
.. _#31: https://github.com/ljvmiranda921/pyswarms/pull/31

0.1.6 (2017-09-24)
~~~~~~~~~~~~~~~~~~

* **NEW:** Native GridSearch and RandomSearch implementations for finding the best hyperparameters in controlling swarm behaviour - `#4`_, `#20`_, `#25`_. Contributed by `@SioKCronin <https://github.com/SioKCronin>`_. Thanks a lot!
* **NEW:** Added tests for hyperparameter search techniques - `#27`_, `#28`_, `#40`_. Contributed by `@jazcap53 <https://github.com/jazcap53>`_. Thank you so much!
* **IMPROVED:** Updated structure of Base classes for higher extensibility

.. _#4: https://github.com/ljvmiranda921/pyswarms/issues/4
.. _#20: https://github.com/ljvmiranda921/pyswarms/pull/20
.. _#25: https://github.com/ljvmiranda921/pyswarms/pull/25
.. _#27: https://github.com/ljvmiranda921/pyswarms/issues/27
.. _#28: https://github.com/ljvmiranda921/pyswarms/pull/28
.. _#40: https://github.com/ljvmiranda921/pyswarms/pull/40

0.1.7 (2017-09-25)
~~~~~~~~~~~~~~~~~~

* **FIX:** Fixed patch on :code:`local_best.py`  and :code:`binary.py` - `#33`_, `#34`_. Thanks for the awesome fix, `@CPapadim <https://github.com/CPapadim>`_!
* **NEW:** Git now ignores IPython notebook checkpoints

.. _#33: https://github.com/ljvmiranda921/pyswarms/issues/33
.. _#34: https://github.com/ljvmiranda921/pyswarms/pull/34

0.1.8 (2018-01-11)
~~~~~~~~~~~~~~~~~~

* **NEW:** PySwarms is now published on the Journal of Open Source Software (JOSS)! You can check the review here_. In addition, you can also find our paper in this link_. Thanks a lot to `@kyleniemeyer <https://github.com/kyleniemeyer>`_ and `@stsievert <https://github.com/stsievert>`_ for the thoughtful reviews and comments.

.. _here: https://github.com/openjournals/joss-reviews/issues/433
.. _link: http://joss.theoj.org/papers/235299884212b9223bce909631e3938b

0.1.9 (2018-04-20)
~~~~~~~~~~~~~~~~~~

* **NEW:** You can now set the initial position wherever you want - `#93`_
* **FIX:** Quick-fix for the Rosenbrock function - `#98`_
* **NEW:** Tolerance can now be set to break during iteration - `#100`_

Thanks for all the wonderful Pull Requests, `@mamadyonline <https://github.com/mamadyonline>`_!

.. _#93: https://github.com/ljvmiranda921/pyswarms/pull/93
.. _#98: https://github.com/ljvmiranda921/pyswarms/pull/98
.. _#100: https://github.com/ljvmiranda921/pyswarms/pull/100


0.2.0 (2018-06-11)
------------------

* **NEW:** New PySwarms backend. You can now build native swarm implementations using this module! -  `#115`_, `#116`_, `#117`_
* **DEPRECATED:** Drop Python 2.7 version support. This package now supports Python 3.4 and up - `#113`_
* **IMPROVED:** All tests were ported into pytest - `#114`_

.. _#113: https://github.com/ljvmiranda921/pyswarms/pull/113
.. _#114: https://github.com/ljvmiranda921/pyswarms/pull/114
.. _#115: https://github.com/ljvmiranda921/pyswarms/pull/115
.. _#116: https://github.com/ljvmiranda921/pyswarms/pull/116
.. _#117: https://github.com/ljvmiranda921/pyswarms/pull/117


0.2.1 (2018-06-27)
~~~~~~~~~~~~~~~~~~

* **FIX:** Fix sigmoid function in BinaryPSO - `#145`_. Thanks a lot `@ThomasCES <https://github.com/ThomasCES>`_!

.. _#145: https://github.com/ljvmiranda921/pyswarms/pull/145

0.3.0 (2018-08-10)
------------------

* **NEW:** New topologies: Pyramid, Random, and Von Neumann. More ways for your particles to interact! - `#176`_, `#177`_, `#155`_, `#142`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* **NEW:** New GeneralOptimizer algorithm that allows you to switch-out topologies for your optimization needs - `#151`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* **NEW:** All topologies now have a static attribute. Neigbors can now be set initially or computed dynamically - `#164`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* **NEW:** New single-objective functions - `#168`_. Awesome work, `@jayspeidell <https://github.com/jayspeidell>`_!
* **NEW:** New tutorial on Inverse Kinematics using Particle Swarm Optimization - `#141`_. Thanks a lot `@whzup <https://github.com/whzup>`_!
* **NEW:** New plotters module for visualization. The environment module is now deprecated - `#135`_
* **IMPROVED:** Keyword arguments can now be passed in the :code:`optimize()` method for your custom objective functions - `#144`_. Great job, `@bradahoward <https://github.com/bradahoward>`_

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

0.3.1 (2018-08-13)
~~~~~~~~~~~~~~~~~~

* **NEW:** New collaboration tool using Vagrantfiles - `#193`_. Thanks a lot `@jdbohrman <https://github.com/jdbohrman>`_!
* **NEW:** Add configuration file for pyup.io - `#210`_
* **FIX:** Fix incomplete documentation in ReadTheDocs - `#208`_
* **IMPROVED:** Update dependencies via pyup - `#204`_

.. _#193: https://github.com/ljvmiranda921/pyswarms/pull/193
.. _#204: https://github.com/ljvmiranda921/pyswarms/pull/204
.. _#208: https://github.com/ljvmiranda921/pyswarms/pull/208
.. _#210: https://github.com/ljvmiranda921/pyswarms/pull/210

0.4.0 (2019-01-29)
------------------

* **NEW:** The console output is now generated by the :code:`Reporter` module - `#227`_
* **NEW:** A :code:`@cost` decorator which automatically scales to the whole swarm - `#226`_
* **FIX:** A bug in the topologies where the best position in some topologies was not calculated using the nearest  neighbours - `#253`_
* **FIX:** Swarm init positions - `#249`_ Thanks `@dfhljf`_!
* **IMPROVED:** Better naming for the benchmark functions - `#222`_ Thanks `@nik1082`_!
* **IMPROVED:** Error handling in the :code:`Optimizers` - `#232`_
* **IMPROVED:** New management method for dependencies - `#263`_
* **DEPRECATED:** The `environments` module is now deprecated - `#217`_

.. _#217: https://github.com/ljvmiranda921/pyswarms/pull/217
.. _#222: https://github.com/ljvmiranda921/pyswarms/pull/222
.. _#226: https://github.com/ljvmiranda921/pyswarms/pull/226
.. _#227: https://github.com/ljvmiranda921/pyswarms/pull/227
.. _#232: https://github.com/ljvmiranda921/pyswarms/pull/232
.. _#249: https://github.com/ljvmiranda921/pyswarms/pull/249
.. _#253: https://github.com/ljvmiranda921/pyswarms/pull/253
.. _#263: https://github.com/ljvmiranda921/pyswarms/pull/263
.. _@nik1082: https://github.com/nik1082
.. _@dfhljf: https://github.com/dfhljf

1.0.0 (2019-02-08)
------------------

This is the first major release of PySwarms. Starting today, we will be adhering to a `better semantic versioning guidelines`_. We will be updating the project wikis shortly after. The maintainers believe that PySwarms is mature enough to merit a version 1, this would also help us release more often (mostly minor releases) and create patch releases as soon as possible.

Also, we will be maintaining a quarterly release cycle, where the **next minor release (v.1.0.0) will be on June**. All enhancements and new features will be staged on the :code:`development` branch, then will be merged back to the :code:`master` branch at the end of the cycle. However, bug fixes and documentation errors will merit a patch release, and will be merged to :code:`master` immediately.

* **NEW:** Boundary and velocity handlers to resolve stuck particles - `#238`_ . All thanks for our maintainer, `@whzup`_ !
* **FIX:** Duplication function calls during optimization, hopefully your long-running objective functions won't take doubly long. - `#266`_. Thank you `@danielcorreia96`_ !

.. _better semantic versioning guidelines: https://semver.org/
.. _#238: https://github.com/ljvmiranda921/pyswarms/pull/238
.. _#266: https://github.com/ljvmiranda921/pyswarms/pull/266
.. _@whzup: https://github.com/whzup
.. _@danielcorreia96: https://github.com/danielcorreia96


1.0.1 (2019-02-14)
~~~~~~~~~~~~~~~~~~

* **FIX:** Handlers memory management so that it works all the time - `#286`_ . Thanks for this `@whzup`_ !
* **FIX:** Re-introduce fix for multiple optimization function calls - `#290`_ . Thank you once more `@danielcorreia96`_ !

.. _#286: https://github.com/ljvmiranda921/pyswarms/pull/286
.. _#290: https://github.com/ljvmiranda921/pyswarms/pull/290
.. _@whzup: https://github.com/whzup
.. _@danielcorreia96: https://github.com/danielcorreia96

1.0.2 (2019-02-17)
~~~~~~~~~~~~~~~~~~

* **FIX:** BinaryPSO should return final best position instead of final swarm - `#293`_ . Thank you once more `@danielcorreia96`_ !

.. _#293: https://github.com/ljvmiranda921/pyswarms/pull/293
.. _@danielcorreia96: https://github.com/danielcorreia96

1.1.0 (2019-05-18)
------------------

This new version adds support for parallel particle evaluation, better
documentation, multiple fixes, and updated build dependencies.

- **NEW**: Updated API documentation - `#344`_
- **NEW**: Relaxed dependencies when installing pyswarms - `#345`_
- **NEW**: We're now using Azure Pipelines for our builds! - `#327`_ 
- **NEW**: Add notebook for electric circuits  - `#288`_ . Thank you `@miguelcocruz`_ !
- **NEW**: Parallel particle evaluation - `#312`_ . Thahnk you once more `@danielcorreia96`_ !
- **FIX**: Fix optimise methods returning incorrect best_pos - `#322`_ . Thank you `@ichbinjakes`_ !
- **FIX**: Fix SearchBase parameter - `#328`_ . Thank you `@Kutim` !
- **FIX**: Fix basic optimization example - `#329`_ . Thank you `@IanBoyanZhang`_ !
- **FIX**: Fix global best velocity equation - `#330`_ . Thank you `@craymichael`_ !
- **FIX**: Update sample code to new API - `#296`_ . Thank you `@ndngo`_ !

.. _#288: https://github.com/ljvmiranda921/pyswarms/pull/288
.. _#296: https://github.com/ljvmiranda921/pyswarms/pull/296
.. _#312: https://github.com/ljvmiranda921/pyswarms/pull/312
.. _#322: https://github.com/ljvmiranda921/pyswarms/pull/322
.. _#327: https://github.com/ljvmiranda921/pyswarms/pull/327
.. _#328: https://github.com/ljvmiranda921/pyswarms/pull/328
.. _#329: https://github.com/ljvmiranda921/pyswarms/pull/329
.. _#330: https://github.com/ljvmiranda921/pyswarms/pull/330
.. _#344: https://github.com/ljvmiranda921/pyswarms/pull/344
.. _#345: https://github.com/ljvmiranda921/pyswarms/pull/345
.. _@danielcorreia96: https://github.com/danielcorreia96
.. _@miguelcocruz: https://github.com/miguelcocruz
.. _@ichbinjakes: https://github.com/ichbinjakes
.. _@Kutim: https://github.com/Kutim
.. _@IanBoyanZhang: https://github.com/IanBoyanZhang
.. _@craymichael: https://github.com/craymichael
.. _@ndngo: https://github.com/ndngo
