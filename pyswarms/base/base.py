import numpy as np 


class SwarmBase(object):
	"""Base class for all Particle Swarm Optimization implementations.

	Note that all methods here are abstract and raises a 
	NotImplementedError when not used. When defining your own swarm, 
	create another class,

		> class MySwarm(SwarmBaseClass):
		>     def __init__(self):
		>        super(MySwarm, self).__init__()

	and define all the necessary methods needed for your 
	implementation.

	Take note that there is no velocity nor position update in this 
	base class. This enables this class to accommodate any variation 
	of the position or velocity update, without enforcing a specific 
	structure.

	If you wish to pattern your update rules to the original PSO by 
	Eberhant et al., simply check the swarms.standard.pso.GBestPSO and 
	swarms.standard.pso.LBestPSO classes that were implemented.

	"""
	def assertions(self):
		"""Assertion method to check various inputs."""
		if self.bounds is not None:
			assert len(self.bounds) == 2, "bounds must be of size 2."
			assert self.bounds[0].shape == self.bounds[1].shape, "unequal bound shapes"
			assert self.bounds[0].shape == self.dims.shape, "bounds must be the same size as dims."

	def __init__(self, n_particles, dims, bounds=None, **kwargs):
		"""Initializes the swarm. 

		Creates a numpy.ndarray of positions depending on the number 
		of particles needed and the number of dimensions. The initial 
		positions of the particles are sampled from a uniform 
		distribution.

		Inputs:
			- n_particles: (int) number of particles in the swarm.
			- dims: (int) number of dimensions in the space.
			- bounds: (np.ndarray, np.ndarray)  a tuple of np.ndarrays 
				where the first entry is the minimum bound while the 
				second entry is the maximum bound. Each array must be 
				of shape (dims,).
			- **kwargs: a dictionary containing various kwargs for a 
				specific optimization technique
		"""

		# Convenience attributes
		self.n_particles = n_particles
		self.dims = dims
		self.bounds = bounds

		# Broadcast the bounds and initialize the swarm 
		if bounds is not None:
			self.min_bounds = np.repeat(bounds[0][np.newaxis,:], n_particles, axis=0)
			self.max_bounds = np.repeat(bounds[1][np.newaxis,:], n_particles, axis=0)
			self.pos = np.random.uniform(low=self.min_bounds, high=self.max_bounds, size=(n_particles,dims))
		else:
			self.pos = np.random.uniform(size=(n_particles,dims))

		# Initialize the global best of the swarm. 
		self.gbest_cost = np.inf
		self.gbest_pos = None

		# Initialize the personal best of each particle.
		self.pbest_pos = self.pos

		# List of kwargs
		self.kwargs = kwargs

		# Invoke assertions
		self.assertions()


	def optimize(self, f, iters, print_step=1, verbose=1):
		"""Optimizes the swarm for a number of iterations.

		Performs the optimization to evaluate the objective 
		function `f` for a number of iterations `iter.`

		Inputs:
			- f: (method) objective function to be evaluated
			- iters: (int) nb. of iterations 
			- print_step: amount of steps for printing into console.
			- verbose: verbosity setting
		Raises:
			- NotImplementedError: This is an abstract method.
		"""
		raise NotImplementedError("SwarmBase::optimize()")