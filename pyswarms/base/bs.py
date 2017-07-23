# -*- coding: utf-8 -*-

""" bs.py: base class for single-objective PSO """

import numpy as np 


class SwarmBase(object):
    """Base class for single-objective Particle Swarm Optimization 
    implementations.

    Note that all methods here are abstract and raises a 
    NotImplementedError when not used. When defining your own swarm, 
    create another class,

    >>> class MySwarm(SwarmBaseClass):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

    and define all the necessary methods needed for your 
    implementation.

    Take note that there is no velocity nor position update in this 
    base class. This enables this class to accommodate any variation 
    of the position or velocity update, without enforcing a specific 
    structure.

    If you wish to pattern your update rules to the original PSO by 
    Eberhant et al., simply check the global best and local best
    implementations in this package

    See Also
    --------
    swarms.standard.pso.GBestPSO: global-best PSO implementation
    swarms.standard.pso.LBestPSO: local-best PSO implementation

    """
    def assertions(self):
        """Assertion method to check various inputs."""

        # Check setting of bounds
        if self.bounds is not None:
            assert type(self.bounds) == tuple, "bound must be a tuple."
            assert len(self.bounds) == 2, "bounds must be of size 2."
            assert self.bounds[0].shape == self.bounds[1].shape, "unequal bound shapes"
            assert self.bounds[0].shape[0] == self.bounds[1].shape[0] == self.dims, "bounds must be the same size as dims."
            assert (self.bounds[1] > self.bounds[0]).all(), "all values of max bounds should be greater than min bounds"

    def __init__(self, n_particles, dims, bounds=None, **kwargs):
        """Initializes the swarm. 

        Creates a numpy.ndarray of positions depending on the number 
        of particles needed and the number of dimensions. The initial 
        positions of the particles are sampled from a uniform 
        distribution.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dims : int
            number of dimensions in the space.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound
            while the second entry is the maximum bound. Each array must
            be of shape (dims,).
        **kwargs: dict
            a dictionary containing various kwargs for a specific 
            optimization technique

        """
        # Initialize primary swarm attributes
        self.n_particles = n_particles
        self.dims = dims
        self.bounds = bounds
        self.swarm_size = (n_particles, dims)
        self.kwargs = kwargs

        # Invoke assertions
        self.assertions()

        # Initialize resettable attributes
        self.reset()

    def optimize(self, f, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective 
        function `f` for a number of iterations `iter.`

        Parameters
        ----------
        f : function
            objective function to be evaluated
        iters : int 
            number of iterations 
        print_step : int
            amount of steps for printing into console
            (the default is 1).
        verbose : int
            verbosity setting (the default is 1).

        Raises
        ------
        NotImplementedError
            When this method is not implemented.

        """
        raise NotImplementedError("SwarmBase::optimize()")

    def reset(self):
        """Resets the attributes of the optimizer.
        
        All variables/atributes that will be re-initialized when this
        method is called should be defined here. Note that this method
        can be called twice: (1) during initialization, and (2) when
        this is called from an instance.

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