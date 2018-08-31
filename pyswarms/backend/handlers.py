import logging

import numpy as np

from ..util.reporter import Reporter

rep = Reporter(logger=logging.getLogger(__name__))


class BoundaryHandler:
    def __init__(self, strategy):
        self.strategy = strategy

    def __call__(self, position, bounds, *args, **kwargs):
        # Assign new attributes
        self.position = position
        self.lower_bound, self.upper_bound = bounds
        self.__out_of_bounds()

        if self.strategy == "nearest":
            new_position = self.nearest()
        elif self.strategy == "reflective":
            new_position = self.reflective()
        elif self.strategy == "shrink":
            new_position = self.shrink()
        elif self.strategy == "random":
            new_position = self.random()
        elif self.strategy == "intermediate":
            new_position = self.random()
        elif self.strategy == "resample":
            new_position = self.resample()

        return self.position

    def __out_of_bounds(self):
        """
        Return the indices of the particles that are out of bounds
        """
        self.greater_than_bound = np.nonzero(self.position > self.upper_bound)
        self.lower_than_bound = np.nonzero(self.position < self.lower_bound)

    def nearest(self):
        """
        Set position to nearest bound
        """
        self.position[self.greater_than_bound] = self.upper_bound[self.greater_than_bound[1]]
        self.position[self.lower_than_bound] = self.lower_bound[self.lower_than_bound[1]]

    def reflective(self):
        pass

    def shrink(self):
        pass

    def random(self):
        pass

    def intermediate(self):
        pass

    def resample(self):
        pass

