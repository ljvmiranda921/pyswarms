# -*- coding: utf-8 -*-

"""
Plot Formatters

This module implements helpful classes to format your plots or create meshes.
"""

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib import cm, colors


@dataclass
class Designer(object):
    """Designer class for specifying a plot's formatting and design

    You can use this class for specifying design-related customizations to
    your plot. This can be passed in various functions found in the
    :mod:`pyswarms.utils.plotters` module.

    .. code-block :: python

        from pyswarms.utils.plotters import plot_cost_history
        from pyswarms.utils.plotters.formatters import Designer

        # Set title_fontsize into 20
        my_designer = Designer(title_fontsize=20)

        # Assuming we already had an optimizer ready
        plot_cost_history(cost_history, designer=my_designer)

    Attributes
    ----------
    figsize : tuple
        Overall figure size. Default is `(10, 8)`
    title_fontsize : str, int, or float
        Size of the plot's title. Default is `large`
    text_fontsize : str, int, or float
        Size of the plot's labels and legend. Default is `medium`
    legend : str
        Label to show in the legend. For cost histories, it states
        the label of the line plot. Default is `Cost`
    label : array_like
        Label to show in the x, y, or z-axis. For a 3D plot, please pass an
        iterable with three elements. Default is :code:`['x-axis', 'y-axis',
        'z-axis']`
    limits : list
        The x-, y-, z- limits of the axes. Pass an iterable with the number of
        elements representing the number of axes. Default is :code:`[(-1, 1),
        (-1, 1), (-1, 1)]`
    colormap : matplotlib.cm.Colormap
        Colormap for contour plots. Default is `cm.viridis`
    """

    # Overall plot design
    figsize: Tuple[float, float] = field(default_factory=lambda: (10, 8))
    title_fontsize: str | int | float = field(default="large")
    text_fontsize: str | int | float = field(default="medium")
    legend: str = field(default="Cost")
    label: str | List[str] | Tuple[str, ...] = field(default_factory=lambda: ["x-axis", "y-axis", "z-axis"])
    limits: List[Tuple[int, ...]] | Tuple[Tuple[int, ...]] = field(default_factory=lambda: [(-1, 1), (-1, 1), (-1, 1)])
    colormap: colors.Colormap = field(default=cm.viridis)  # type: ignore


@dataclass
class Animator(object):
    """Animator class for specifying animation behavior

    You can use this class to modify options on how the animation will be run
    in the :func:`pyswarms.utils.plotters.plot_contour` and
    :func:`pyswarms.utils.plotters.plot_surface` methods.

    .. code-block :: python

        from pyswarms.utils.plotters import plot_contour
        from pyswarms.utils.plotters.formatters import Animator

        # Do not repeat animation
        my_animator = Animator(repeat=False)

        # Assuming we already had an optimizer ready
        plot_contour(pos_history, animator=my_animator)

    Attributes
    ----------
    interval : int
        Sets the interval or speed into which the animation is played.
        Default is `80`
    repeat_delay : int or float, optional
        Sets the delay before repeating the animation again.
    repeat : bool, optional
        Pass `False` if you don't want to repeat the animation.
        Default is `True`
    """

    interval: int = 80
    repeat_delay: Optional[int | float] = None
    repeat: bool = True


@dataclass
class Mesher(object):
    """Mesher class for plotting contours of objective functions

    This class enables drawing a surface plot of a given objective function.
    You can customize how this plot is drawn with this class. Pass an instance
    of this class to enable meshing.

    .. code-block :: python

        from pyswarms.utils.plotters import plot_surface
        from pyswarms.utils.plotters.formatters import Mesher
        from pyswarms.utils.functions import single_obj as fx

        # Use sphere function
        my_mesher = Mesher(func=fx.sphere)

        # Assuming we already had an optimizer ready
        plot_surface(pos_history, mesher=my_mesher)

    Attributes
    ----------
    func : callable
        Objective function to plot a surface of.
    delta : float
        Number of steps when generating the surface plot
        Default is `0.001`
    limits : list or tuple
        The range, in each axis, where the mesh will be drawn.
        Default is :code:`[(-1,1), (-1,1)]`
    levels : list or int, optional
        Levels on which the contours are shown. If :code:`int` is passed,
        then `matplotlib` automatically computes for the level positions.
        Default is :code:`numpy.arange(-2.0, 2.0, 0.070)`
    alpha : float, optional
        Transparency of the surface plot. Default is `0.3`
    limits : list, optional
        The x-, y-, z- limits of the axes. Pass an iterable with the number of
        elements representing the number of axes. Default is :code:`[(-1, 1),
        (-1, 1)]`
    """

    func: Callable[..., float]
    # For mesh creation
    delta: float = 0.001
    limits: List[Tuple[int, ...]] | Tuple[Tuple[int, ...], ...] = field(default_factory=lambda: [(-1, 1), (-1, 1)])
    levels: npt.NDArray[Any] = field(default_factory=lambda: np.arange(-2.0, 2.0, 0.070))
    # Surface transparency
    alpha: float = 0.3

    def compute_history_3d(self, pos_history: npt.NDArray[Any], n_processes: Optional[int] = None):
        """Compute a 3D position matrix

        The first two columns are the 2D position in the x and y axes
        respectively, while the third column is the fitness on that given
        position.

        Parameters
        ----------
        pos_history : numpy.ndarray
            Two-dimensional position matrix history of shape
            :code:`(iterations, n_particles, 2)`
        n_processes : int, optional
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)

        Returns
        -------
        numpy.ndarray
            3D position matrix of shape :code:`(iterations, n_particles, 3)`
        """
        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        if pool is None:
            fitness = np.array(list(map(self.func, pos_history)))
        else:
            iter_r: List[npt.NDArray[Any]] = []
            # Iterate over iterations
            for i in range(len(pos_history)):
                # Parallelize particles
                r_map_split = pool.map(
                    self.func,
                    np.array_split(np.array(pos_history[i]), pool._processes),  # type: ignore
                )
                iter_r.append(np.array(np.concatenate(r_map_split)))
            fitness = np.array(iter_r)

        # Close Pool of Processes
        if pool is not None:
            pool.close()

        return np.dstack((pos_history, fitness))
