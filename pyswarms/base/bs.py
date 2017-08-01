# -*- coding: utf-8 -*-

r"""
Base class for single-objective Particle Swarm Optimization 
implementations.

All methods here are abstract and raises a :code:`NotImplementedError` 
when not used. When defining your own swarm implementation,
create another class,

    >>> class MySwarm(SwarmBaseClass):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

and define all the necessary methods needed.

Take note that there is no velocity nor position update in this
base class. This enables this class to accommodate any variation
of the position or velocity update, without enforcing a specific
structure. As a guide, check the global best and local best
implementations in this package.

.. note:: Regarding :code:`**kwargs`, it is highly recommended to
    include parameters used in position and velocity updates as
    keyword arguments. For parameters that affect the topology of
    the swarm, it may be much better to have them as positional
    arguments.

See Also
--------
:mod:`pyswarms.single.gb`: global-best PSO implementation
:mod:`pyswarms.single.lb`: local-best PSO implementation
"""



import numpy as np 


class SwarmBase(object):

    def assertions(self):
        """Assertion method to check various inputs.
        
        Raises
        ------
        TypeError
            When the :code:`bounds` is not of type tuple
        IndexError
            When the :code:`bounds` is not of size 2.
            When the arrays in :code:`bounds` is not of equal size.
            When the shape of :code:`bounds` is not the same as `dims`.
        ValueError
            When the value of :code:`bounds[1]` is less than
            :code:`bounds[0]`.
        """

        # Check setting of bounds
        if self.bounds is not None:
            if not type(self.bounds) == tuple:
                raise TypeError('Parameter `bound` must be a tuple.')
            if not len(self.bounds) == 2:
                raise IndexError('Parameter `bound` must be of size 2.')
            if not self.bounds[0].shape == self.bounds[1].shape:
                raise IndexError('Arrays in `bound` must be of equal shapes')
            if not self.bounds[0].shape[0] == self.bounds[1].shape[0] == self.dims:
                raise IndexError('Parameter `bound` must be the shape as dims.')
            if not (self.bounds[1] > self.bounds[0]).all():
                raise ValueError('Values of `bounds[1]` must be greater than `bounds[0]`.')

        # Check clamp settings
        if self.v_clamp is not None:
            if not type(self.v_clamp) == tuple:
                raise TypeError('Parameter `v_clamp` must be a tuple')
            if not len(self.v_clamp) == 2:
                raise IndexError('Parameter `v_clamp` must be of size 2')
            if not self.v_clamp[0] < self.v_clamp[1]:
                raise ValueError('Make sure that v_clamp is in the form (v_min, v_max)')

    def __init__(self, n_particles, dims, bounds=None, v_clamp=None, **kwargs):
        """Initializes the swarm. 

        Creates a :code:`numpy.ndarray` of positions depending on the
        number of particles needed and the number of dimensions.
        The initial positions of the particles are sampled from a
        uniform distribution.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dims : int
            number of dimensions in the space.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape :code:`(dims,)`.
        v_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It 
            sets the limits for velocity clamping. 
        **kwargs: dict
            a dictionary containing various keyword arguments for a
            specific optimization technique
        """
        # Initialize primary swarm attributes
        self.n_particles = n_particles
        self.dims = dims
        self.bounds = bounds
        self.v_clamp = v_clamp
        self.swarm_size = (n_particles, dims)
        self.kwargs = kwargs

        # Invoke assertions
        self.assertions()

        # Initialize resettable attributes
        self.reset()

    def optimize(self, f, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective 
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        f : function
            objective function to be evaluated
        iters : int 
            number of iterations 
        print_step : int (the default is 1)
            amount of steps for printing into console.
        verbose : int (the default is 1)
            verbosity setting.

        Raises
        ------
        NotImplementedError
            When this method is not implemented.
        """
        raise NotImplementedError("SwarmBase::optimize()")

    def reset(self):
        """Resets the attributes of the optimizer.
        
        All variables/atributes that will be re-initialized when this
        method is defined here. Note that this method
        can be called twice: (1) during initialization, and (2) when
        this is called from an instance.

        It is good practice to keep the number of resettable
        attributes at a minimum. This is to prevent spamming the same
        object instance with various swarm definitions.

        Normally, swarm definitions are as atomic as possible, where
        each type of swarm is contained in its own instance. Thus, the
        following attributes are the only ones recommended to be
        resettable:
            * Swarm position matrix (self.pos)
            * Velocity matrix (self.pos)
            * Best scores and positions (gbest_cost, gbest_pos, etc.)

        Otherwise, consider using positional arguments.
        """
        # Broadcast the bounds and initialize the swarm
        if self.bounds is not None:
            self.min_bounds = np.repeat(self.bounds[0][np.newaxis,:],
                                        self.n_particles,
                                        axis=0)
            self.max_bounds = np.repeat(self.bounds[1][np.newaxis,:],
                                        self.n_particles,
                                        axis=0)
            self.pos = np.random.uniform(low=self.min_bounds,
                                        high=self.max_bounds,
                                        size=self.swarm_size)
        else:
            self.pos = np.random.uniform(size=self.swarm_size)

        # Initialize velocity vectors
        if self.v_clamp is not None:
            v_min, v_max = self.v_clamp[0], self.v_clamp[1]
            self.velocity = ((v_max - v_min) 
                            * np.random.random_sample(size=self.swarm_size) 
                            + v_min)
        else:
            self.velocity = np.random.random_sample(size=self.swarm_size)