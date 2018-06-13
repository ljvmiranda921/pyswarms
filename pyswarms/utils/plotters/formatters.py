# -*- coding: utf-8 -*-

"""
Plot Formatters

This module implements helpful classes to format your plots or create meshes.
"""

# Import modules
import numpy as np
from attr import (attrs, attrib)
from attr.validators import instance_of

@attrs
class Designer(object):
    """Designer class for specifying a plot's formatting and design"""
    # Overall plot design
    figsize = attrib(type=tuple, validator=instance_of(tuple), default=(10,8))
    title_fontsize = attrib(validator=instance_of((str, int, float)),
                            default='large')
    text_fontsize = attrib(validator=instance_of((str, int, float)),
                           default='medium')
    label = attrib(validator=instance_of((str, list, tuple)), default='Cost')
    limits = attrib(validator=instance_of((list, tuple)), 
                    default=[(-1,1),(-1,1)])

@attrs
class Animator(object):
    """Animator class for specifying animation behavior"""
    interval = attrib(type=int, validator=instance_of(int), default=80)
    repeat_delay = attrib(default=None)
    repeat = attrib(type=bool, validator=instance_of(bool), default=True)

@attrs
class Mesher(object):
    """Mesher class for plotting contours of objective functions"""
    func = attrib()
    # For mesh creation
    delta = attrib(type=float, default=0.001)
    limits = attrib(validator=instance_of((list, tuple)),
                    default=[(-1,1),(-1,1)])
    levels = attrib(type=list, default=np.arange(-2.0,2.0,0.070))
    # Surface transparency
    alpha = attrib(type=float, validator=instance_of(float), default=0.3)

    def compute_history_3d(self, pos_history):
        """Computes a 3D position matrix

        The first two columns are the 2D position in the x and y axes
        respectively, while the third column is the fitness on that given
        position.

        Parameters
        ----------
        pos_history : numpy.ndarray
            Two-dimensional position matrix history of shape
            :code:`(iterations, n_particles, 2)`

        Returns
        -------
        numpy.ndarray
            3D position matrix of shape :code:`(iterations, n_particles, 3)`
        """
        fitness = np.array(list(map(self.func, pos_history)))
        return np.dstack((pos_history, fitness))