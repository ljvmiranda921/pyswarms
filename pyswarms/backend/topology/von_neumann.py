# -*- coding: utf-8 -*-

"""
A Von Neumann Network Topology

This class implements a Von Neumann topology.
"""

# Import standard library
import logging

from ...utils.reporter import Reporter
from .ring import Ring


class VonNeumann(Ring):
    def __init__(self, static=None):
        # static = None is just an artifact to make the API consistent
        # Setting it will not change swarm behavior
        super(VonNeumann, self).__init__(static=True)
        self.rep = Reporter(logger=logging.getLogger(__name__))

    def compute_gbest(self, swarm, p, r, **kwargs):
        """Updates the global best using a neighborhood approach

        The Von Neumann topology inherits from the Ring topology and uses
        the same approach to calculate the global best. The number of
        neighbors is determined by the dimension and the range. This
        topology is always a :code:`static` topology.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        r : int
            range of the Von Neumann topology
        p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        k = VonNeumann.delannoy(swarm.dimensions, r)
        return super(VonNeumann, self).compute_gbest(swarm, p, k)

    @staticmethod
    def delannoy(d, r):
        """Static helper method to compute Delannoy numbers

        This method computes the number of neighbours of a Von Neumann
        topology, i.e. a Delannoy number, dependent on the range and the
        dimension of the search space. The Delannoy numbers are computed
        recursively.

        Parameters
        ----------
        d : int
            dimension of the search space
        r : int
            range of the Von Neumann topology

        Returns
        -------
        int
            Delannoy number"""
        if d == 0 or r == 0:
            return 1
        else:
            del_number = (
                VonNeumann.delannoy(d - 1, r)
                + VonNeumann.delannoy(d - 1, r - 1)
                + VonNeumann.delannoy(d, r - 1)
            )
            return del_number
