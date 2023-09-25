# -*- coding: utf-8 -*-

"""
A Von Neumann Network Topology

This class implements a Von Neumann topology.
"""

from typing import Any, Dict, Literal

from pyswarms.backend.swarms import Swarm
from pyswarms.backend.topology.ring import Ring


class VonNeumann(Ring):
    def __init__(self, dimensions: int, r: int, p: Literal[1, 2], static: bool = True):
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
        super(VonNeumann, self).__init__(p, k)

    def compute_gbest(self, swarm: Swarm, **kwargs: Dict[str, Any]):
        """Updates the global best using a neighborhood approach

        The Von Neumann topology inherits from the Ring topology and uses
        the same approach to calculate the global best. The number of
        neighbors is determined by the dimension and the range. This
        topology is always a :code:`static` topology.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance

        Returns
        -------
        numpy.ndarray
            Best position of shape :code:`(n_dimensions, )`
        float
            Best cost
        """
        return super(VonNeumann, self).compute_gbest(swarm)

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


if __name__ == "__main__":
    print(VonNeumann.delannoy(20, 30))

    def delannoy(m: int, n: int) -> int:
        # Ensure m is the larger dimension
        if m < n:
            m, n = n, m

        # Initialize an array to store the current row's Delannoy numbers
        current_row = [1] * (n + 1)

        for _ in range(1, m + 1):
            # Initialize the first element of the current row
            current_value = 1

            for j in range(1, n + 1):
                # Calculate the Delannoy number for the current cell
                current_value = current_row[j] + current_value

                # Update the current row for the next iteration
                current_row[j] = current_value

        # The last element of the current_row contains the Delannoy number for m rows and n columns
        return current_row[n]

    # Example usage:
    m = 20
    n = 30
    result = delannoy(m, n)
    print(f"Delannoy({m}, {n}) = {result}")
