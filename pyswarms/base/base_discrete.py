# -*- coding: utf-8 -*-

r"""
Base class for single-objective discrete Particle Swarm Optimization
implementations.

All methods here are abstract and raises a :code:`NotImplementedError`
when not used. When defining your own swarm implementation,
create another class,

    >>> class MySwarm(DiscreteSwarmBase):
    >>>     def __init__(self):
    >>>        super(MySwarm, self).__init__()

and define all the necessary methods needed.

As a guide, check the discrete PSO implementations in this package.

.. note:: Regarding :code:`options`, it is highly recommended to
    include parameters used in position and velocity updates as
    keyword arguments. For parameters that affect the topology of
    the swarm, it may be much better to have them as positional
    arguments.

See Also
--------
:mod:`pyswarms.discrete.binary`: binary PSO implementation

"""

import os
import yaml
import logging
import numpy as np
import logging.config
from collections import namedtuple


class DiscreteSwarmBase(object):

    def assertions(self):
        """Assertion method to check various inputs.

        Raises
        ------
        TypeError
            When the :code:`bounds` is not of type tuple
        IndexError
            When the :code:`bounds` is not of size 2.
            When the arrays in :code:`bounds` is not of equal size.
            When the shape of :code:`bounds` is not the same as `dimensions`.
        ValueError
            When the value of :code:`bounds[1]` is less than
            :code:`bounds[0]`.
        """

        # Check clamp settings
        if self.velocity_clamp is not None:
            if not isinstance(self.velocity_clamp, tuple):
                raise TypeError('Parameter `velocity_clamp` must be a tuple')
            if not len(self.velocity_clamp) == 2:
                raise IndexError('Parameter `velocity_clamp` must be of '
                                 'size 2')
            if not self.velocity_clamp[0] < self.velocity_clamp[1]:
                raise ValueError('Make sure that velocity_clamp is in the '
                                 'form (v_min, v_max)')

        # Required keys in options argument
        if not all(key in self.options for key in ('c1', 'c2', 'w')):
            raise KeyError('Missing either c1, c2, or w in options')

    def setup_logging(self, default_path='./config/logging.yaml',
                      default_level=logging.INFO, env_key='LOG_CFG'):
        """Setup logging configuration

        Parameters
        ----------
        default_path : str (default is `./config/logging.yaml`)
            the path where the logging configuration is stored
        default_level: logging.LEVEL (default is `logging.INFO`)
            the default logging level
        env_key : str
            the environment key for accessing the setup
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def __init__(self, n_particles, dimensions, binary, options,
                 velocity_clamp=None):
        """Initializes the swarm.

        Creates a :code:`numpy.ndarray` of positions depending on the
        number of particles needed and the number of dimensions.
        The initial positions of the particles depends on the argument
        :code:`binary`, which governs if a binary matrix will be produced.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        binary : boolean
            a trigger to generate a binary matrix for the swarm's
            initial positions. When passed with a :code:`False` value,
            random integers from 0 to :code:`dimensions` are generated.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        options: dict
            a dictionary containing the parameters for a specific
            optimization technique
        """
        self.setup_logging()
        # Initialize primary swarm attributes
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.binary = binary
        self.velocity_clamp = velocity_clamp
        self.swarm_size = (n_particles, dimensions)
        self.options = options
        # Initialize named tuple for populating the history list
        self.ToHistory = namedtuple('ToHistory',
                                    ['best_cost', 'mean_pbest_cost',
                                     'mean_neighbor_cost', 'position',
                                     'velocity'])
        # Invoke assertions
        self.assertions()

        # Initialize resettable attributes
        self.reset()

    def _populate_history(self, hist):
        """Populates all history lists

        The :code:`cost_history`, :code:`mean_pbest_history`, and
        :code:`neighborhood_best` is expected to have a shape of
        :code:`(iters,)`,on the other hand, the :code:`pos_history`
        and :code:`velocity_history` are expected to have a shape of
        :code:`(iters, n_particles, dimensions)`

        Parameters
        ----------
        hist : namedtuple
            Must be of the same type as self.ToHistory
        """
        self.cost_history.append(hist.best_cost)
        self.mean_pbest_history.append(hist.mean_pbest_cost)
        self.mean_neighbor_history.append(hist.mean_neighbor_cost)
        self.pos_history.append(hist.position)
        self.velocity_history.append(hist.velocity)

    @property
    def get_cost_history(self):
        """Get cost history"""
        return np.array(self.cost_history)

    @property
    def get_mean_pbest_history(self):
        """Get mean personal best history"""
        return np.array(self.mean_pbest_history)

    @property
    def get_mean_neighbor_history(self):
        """Get mean neighborhood cost history"""
        return np.array(self.mean_neighbor_history)

    @property
    def get_pos_history(self):
        """Get position history"""
        return np.array(self.pos_history)

    @property
    def get_velocity_history(self):
        """Get velocity history"""
        return np.array(self.velocity_history)

    def optimize(self, objective_func, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function :code:`objective_func` for a number of iterations
                 :code:`iter.`

        Parameters
        ----------
        objective_func : function
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

    def _update_velocity(self):
        """Updates the velocity matrix.

        Raises
        ------
        NotImplementedError
            When this method is not implemented.
        """
        raise NotImplementedError("SwarmBase::_update_velocity()")

    def _update_position(self):
        """Updates the position matrix.

        Raises
        ------
        NotImplementedError
            When this method is not implemented.
        """
        raise NotImplementedError("SwarmBase::_update_position()")

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
        # Initialize history lists
        self.cost_history = []
        self.mean_pbest_history = []
        self.mean_neighbor_history = []
        self.pos_history = []
        self.velocity_history = []

        # Generate initial position
        self.pos = np.random.random_sample(size=self.swarm_size).\
            argsort(axis=1)
        if self.binary:
            self.pos = np.random.randint(2, size=self.swarm_size)

        # Initialize velocity vectors
        if self.velocity_clamp is not None:
            min_velocity, max_velocity = self.velocity_clamp[0],\
                                         self.velocity_clamp[1]
            self.velocity = ((max_velocity - min_velocity)
                             * np.random.random_sample(size=self.swarm_size)
                             + min_velocity)
        else:
            self.velocity = np.random.random_sample(size=self.swarm_size)

        # Initialize the best cost of the swarm
        self.best_cost = np.inf
        self.best_pos = None

        # Initialize the personal best of each particle
        self.personal_best_pos = self.pos
