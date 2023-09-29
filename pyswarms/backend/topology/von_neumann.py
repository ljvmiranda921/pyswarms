# -*- coding: utf-8 -*-

"""
A Von Neumann Network Topology

This class implements a Von Neumann topology.
"""

from typing import Literal

from pyswarms.backend.topology.ring import Ring


class VonNeumann(Ring):
    def __init__(self, dimensions: int, r: int, p: Literal[1, 2], static: bool = False):
        """Initializes the class

        Parameters
        ----------
        dimensions: int
            Number of dimensions of the swarm
        r : int
            range of the Von Neumann topology
        p: int {1,2}
            the Minkowski p-norm to use. 1 is the
            sum-of-absolute values (or L1 distance) while 2 is
            the Euclidean (or L2) distance.
        static : bool (Default is :code:`False`)
            a boolean that decides whether the topology
            is static or dynamic
        """
        # static = None is just an artifact to make the API consistent
        # Setting it will not change swarm behavior
        k = self.delannoy(dimensions, r)
        super().__init__(p, k)

    @staticmethod
    def delannoy(d: int, r: int) -> int:
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
                VonNeumann.delannoy(d - 1, r) + VonNeumann.delannoy(d - 1, r - 1) + VonNeumann.delannoy(d, r - 1)
            )
            return del_number
