Solving the Inverse Kinematics problem using Particle Swarm Optimization
========================================================================

In this example, we are going to use the ``pyswarms`` library to solve a
6-DOF (Degrees of Freedom) Inverse Kinematics (IK) problem by treating
it as an optimization problem. We will use the ``pyswarms`` library to
find an *optimal* solution from a set of candidate solutions.

.. code:: python

    import sys
    # Change directory to access the pyswarms module
    sys.path.append('../')

.. code:: python

    print('Running on Python version: {}'.format(sys.version))


.. parsed-literal::

    Running on Python version: 3.6.0 (v3.6.0:41df79263a11, Dec 23 2016, 07:18:10) [MSC v.1900 32 bit (Intel)]


.. code:: python

    # Import modules
    import numpy as np

    # Import PySwarms
    import pyswarms as ps

    # Some more magic so that the notebook will reload external python modules;
    # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
    %load_ext autoreload
    %autoreload 2

.. code:: python

    %%html
    <style>
      table {margin-left: 0 !important;}
    </style>
    # Styling for the text below



.. raw:: html

    <style>
      table {margin-left: 0 !important;}
    </style>
    # Styling for the text below


Introduction
============

Inverse Kinematics is one of the most challenging problems in robotics.
The problem involves finding an optimal *pose* for a manipulator given
the position of the end-tip effector as opposed to forward kinematics,
where the end-tip position is sought given the pose or joint
configuration. Normally, this position is expressed as a point in a
coordinate system (e.g., in a Cartesian system with :math:`x`, :math:`y`
and :math:`z` coordinates). However, the pose of the manipulator can
also be expressed as the collection of joint variables that describe the
angle of bending or twist (in revolute joints) or length of extension
(in prismatic joints).

IK is particularly difficult because an abundance of solutions can
arise. Intuitively, one can imagine that a robotic arm can have multiple
ways of reaching through a certain point. It’s the same when you touch
the table and move your arm without moving the point you’re touching the
table at. Moreover, the calculation of these positions can be very
difficult. Simple solutions can be found for 3-DOF manipulators but
trying to solve the problem for 6 or even more DOF can lead to
challenging algebraic problems.

IK as an Optimization Problem
=============================

In this implementation, we are going to use a *6-DOF Stanford
Manipulator* with 5 revolute joints and 1 prismatic joint. Furthermore,
the constraints of the joints are going to be as follows:

+------------------+--------------------------+-------------------------+
| Parameters       | Lower Boundary           | Upper Boundary          |
+==================+==========================+=========================+
| :math:`\theta_1` | :math:`-\pi`             | :math:`\pi`             |
+------------------+--------------------------+-------------------------+
| :math:`\theta_2` | :math:`-\frac{\pi}{2}`   | :math:`\frac{\pi}{2}`   |
+------------------+--------------------------+-------------------------+
| :math:`d_3`      | :math:`1`                | :math:`3`               |
+------------------+--------------------------+-------------------------+
| :math:`\theta_4` | :math:`-\pi`             | :math:`\pi`             |
+------------------+--------------------------+-------------------------+
| :math:`\theta_5` | :math:`-\frac{5\pi}{36}` | :math:`\frac{5\pi}{36}` |
+------------------+--------------------------+-------------------------+
| :math:`\theta_6` | :math:`-\pi`             | :math:`\pi`             |
+------------------+--------------------------+-------------------------+

**Table 1**: *Physical constraints for the joint variables*

Now, if we are given an *end-tip position* (in this case a :math:`xyz`
coordinate) we need to find the optimal parameters with the constraints
imposed in **Table 1**. These conditions are then sufficient in order to
treat this problem as an optimization problem. We define our parameter
vector :math:`\mathbf{X}` as follows:

.. math:: \mathbf{X}\,:=\, [ \, \theta_1 \quad \theta_2 \quad d_3\ \quad \theta_4 \quad \theta_5 \, ]

And for our end-tip position we define the target vector
:math:`\mathbf{T}` as:

.. math:: \mathbf{T}\,:=\, [\, T_x \quad T_y \quad T_z \,]

We can then start implementing our optimization algorithm.

Initializing the Swarm
======================

The main idea for PSO is that we set a swarm :math:`\mathbf{S}` composed
of particles :math:`\mathbf{P}_n` into a search space in order to find
the optimal solution. The movement of the swarm depends on the cognitive
(:math:`c_1`) and social (:math:`c_2`) of all the particles. The
cognitive component speaks of the particle’s bias towards its personal
best from its past experience (i.e., how attracted it is to its own best
position). The social component controls how the particles are attracted
to the best score found by the swarm (i.e., the global best). High
:math:`c_1` paired with low :math:`c_2` values can often cause the swarm
to stagnate. The inverse can cause the swarm to converge too fast,
resulting in suboptimal solutions.

We define our particle :math:`\mathbf{P}` as:

.. math:: \mathbf{P}\,:=\,\mathbf{X}

And the swarm as being composed of :math:`N` particles with certain
positions at a timestep :math:`t`:

.. math:: \mathbf{S}_t\,:=\,[\,\mathbf{P}_1\quad\mathbf{P}_2\quad ... \quad\mathbf{P}_N\,]

In this implementation, we designate :math:`\mathbf{P}_1` as the initial
configuration of the manipulator at the zero-position. This means that
the angles are equal to 0 and the link offset is also zero. We then
generate the :math:`N-1` particles using a uniform distribution which is
controlled by the hyperparameter :math:`\epsilon`.

Finding the global optimum
==========================

In order to find the global optimum, the swarm must be moved. This
movement is then translated by an update of the current position given
the swarm’s velocity :math:`\mathbf{V}`. That is:

.. math:: \mathbf{S}_{t+1} = \mathbf{S}_t + \mathbf{V}_{t+1}

The velocity is then computed as follows:

.. math:: \mathbf{V}_{t+1} = w\mathbf{V}_t + c_1 r_1 (\mathbf{p}_{best} - \mathbf{p}) + c_2 r_2(\mathbf{g}_{best} - \mathbf{p})

Where :math:`r_1` and :math:`r_2` denote random values in the intervall
:math:`[0,1]`, :math:`\mathbf{p}_{best}` is the best and
:math:`\mathbf{p}` is the current personal position and
:math:`\mathbf{g}_{best}` is the best position of all the particles.
Moreover, :math:`w` is the inertia weight that controls the “memory” of
the swarm’s previous position.

Preparations
------------

Let us now see how this works with the ``pyswarms`` library. We use the
point :math:`[-2,2,3]` as our target for which we want to find an
optimal pose of the manipulator. We start by defining a function to get
the distance from the current position to the target position:

.. code:: python

    def distance(query, target):
        x_dist = (target[0] - query[0])**2
        y_dist = (target[1] - query[1])**2
        z_dist = (target[2] - query[2])**2
        dist = np.sqrt(x_dist + y_dist + z_dist)
        return dist

We are going to use the distance function to compute the cost, the
further away the more costly the position is.

The optimization algorithm needs some parameters (the swarm size,
:math:`c_1`, :math:`c_2` and :math:`\epsilon`). For the *options*
(:math:`c_1`,\ :math:`c_2` and :math:`w`) we have to create a dictionary
and for the constraints a tuple with a list of the respective minimal
values and a list of the respective maximal values. The rest can be
handled with variables. Additionally, we define the joint lengths to be
3 units long:

.. code:: python

    swarm_size = 20
    dim = 6        # Dimension of X
    epsilon = 1.0
    options = {'c1': 1.5, 'c2':1.5, 'w':0.5}

    constraints = (np.array([-np.pi , -np.pi/2 , 1 , -np.pi , -5*np.pi/36 , -np.pi]),
                   np.array([np.pi  ,  np.pi/2 , 3 ,  np.pi ,  5*np.pi/36 ,  np.pi]))

    d1 = d2 = d3 = d4 = d5 = d6 = 3

In order to obtain the current position, we need to calculate the
matrices of rotation and translation for every joint. Here we use the
`Denvait-Hartenberg
parameters <https://en.wikipedia.org/wiki/Denavit–Hartenberg_parameters>`__
for that. So we define a function that calculates these. The function
uses the rotation angle and the extension :math:`d` of a prismatic joint
as input:

.. code:: python

    def getTransformMatrix(theta, d, a, alpha):
        T = np.array([[np.cos(theta) , -np.sin(theta)*np.cos(alpha) ,  np.sin(theta)*np.sin(alpha) , a*np.cos(theta)],
                      [np.sin(theta) ,  np.cos(theta)*np.cos(alpha) , -np.cos(theta)*np.sin(alpha) , a*np.sin(theta)],
                      [0             ,  np.sin(alpha)               ,  np.cos(alpha)               , d              ],
                      [0             ,  0                           ,  0                           , 1              ]
                     ])
        return T

Now we can calculate the transformation matrix to obtain the end tip
position. For this we create another function that takes our vector
:math:`\mathbf{X}` with the joint variables as input:

.. code:: python

    def get_end_tip_position(params):
        # Create the transformation matrices for the respective joints
        t_00 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        t_01 = getTransformMatrix(params[0] , d2        , 0 , -np.pi/2)
        t_12 = getTransformMatrix(params[1] , d2        , 0 , -np.pi/2)
        t_23 = getTransformMatrix(0         , params[2] , 0 , -np.pi/2)
        t_34 = getTransformMatrix(params[3] , d4        , 0 , -np.pi/2)
        t_45 = getTransformMatrix(params[4] , 0         , 0 ,  np.pi/2)
        t_56 = getTransformMatrix(params[5] , d6        ,0  ,  0)

        # Get the overall transformation matrix
        end_tip_m = t_00.dot(t_01).dot(t_12).dot(t_23).dot(t_34).dot(t_45).dot(t_56)

        # The coordinates of the end tip are the 3 upper entries in the 4th column
        pos = np.array([end_tip_m[0,3],end_tip_m[1,3],end_tip_m[2,3]])
        return pos

The last thing we need to prepare in order to run the algorithm is the
actual function that we want to optimize. We just need to calculate the
distance between the position of each swarm particle and the target
point:

.. code:: python

    def opt_func(X):
        n_particles = X.shape[0]  # number of particles
        target = np.array([-2,2,3])
        dist = [distance(get_end_tip_position(X[i]), target) for i in range(n_particles)]
        return np.array(dist)

Running the algorithm
---------------------

Braced with these preparations we can finally start using the algorithm:

.. code:: python

    %%time
    # Call an instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                        dimensions=dim,
                                        options=options,
                                        bounds=constraints)

    # Perform optimization
    cost, joint_vars = optimizer.optimize(opt_func, print_step=100, iters=1000, verbose=3)


.. parsed-literal::

    INFO:pyswarms.single.global_best:Iteration 1/1000, cost: 0.9638223076369133
    INFO:pyswarms.single.global_best:Iteration 101/1000, cost: 2.5258875519324167e-07
    INFO:pyswarms.single.global_best:Iteration 201/1000, cost: 4.7236564972673785e-14
    INFO:pyswarms.single.global_best:Iteration 301/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 401/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 501/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 601/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 701/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 801/1000, cost: 0.0
    INFO:pyswarms.single.global_best:Iteration 901/1000, cost: 0.0
    INFO:pyswarms.single.global_best:================================
    Optimization finished!
    Final cost: 0.0000
    Best value: [ -2.182725 1.323111 1.579636 ...]



.. parsed-literal::

    Wall time: 13.6 s


Now let’s see if the algorithm really worked and test the output for
``joint_vars``:

.. code:: python

    print(get_end_tip_position(joint_vars))


.. parsed-literal::

    [-2.  2.  3.]


Hooray! That’s exactly the position we wanted the tip to be in. Of
course this example is quite primitive. Some extensions of this idea
could involve the consideration of the current position of the
manipulator and the amount of rotation and extension in the optimization
function such that the result is the path with the least movement.
