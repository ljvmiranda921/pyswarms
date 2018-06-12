# -*- coding: utf-8 -*-

r"""
Plotting tool for Optimizer Analysis

This module is built on top of :code:`matplotlib` to render quick and easy
plots for your optimizer. It can plot the best cost for each iteration, and
show animations of the particles in 2-D and 3-D space. Furthermore, because
it has :code:`matplotlib` running under the hood, the plots are easily
customizable.

For example, if we want to plot the cost, simply run the optimizer, get the
cost history from the optimizer instance, and pass it to the
:code:`plot_cost_history()` method

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere_func
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain cost history from optimizer instance
    cost_history = optimizer.cost_history

    # Plot!
    plot_cost_history(cost_history)
    plt.show()

In case you want to plot the particle movement, it is important that either
one of the :code:`matplotlib` animation :code:`Writers` is installed. These
doesn't come out of the box for :code:`pyswarms`, and must be installed
separately. For example, in a Linux or Windows distribution, you can install
:code:`ffmpeg` as

    >>> conda install -c conda-forge ffmpeg

Now, if you want to plot your particles in a 2-D environment, simply pass
the position history of your swarm (obtainable from swarm instance):


.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere_func
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain pos history from optimizer instance
    pos_history = optimizer.get_pos_history

    # Plot!
    plot_trajectory2D(pos_history)

You can also supply various arguments in this method: the indices of the
specific dimensions to be used, the limits of the axes, and the interval/
speed of animation.
"""

# Import modules
import logging
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Initialize logger
logger = logging.getLogger(__name__)

def plot_cost_history(cost_history, ax=None, label='Best cost',
                      title='Cost History', figsize=None,
                      title_fontsize="large", text_fontsize="medium",
                      **kwargs):
    """Creates a simple line plot with the cost in the y-axis and
    the iteration at the x-axis

    Parameters
    ----------
    cost_history : numpy.ndarray
        Cost history of shape (iters, ) where each element contains the cost
        for the given iteration.
    ax : :class:`matplotlib.axes.Axes` (default is :code:`None`)
        The axes where the plot is to be drawn. If :code:`None` is
        passed, then the plot will be drawn to a new set of axes.
    label : str (default is :code:`'Best cost'`)
        Label that will appear in the legend.
    title : str (default is :code:`'Cost History'`)
        The title of the plotted graph.
    figsize : tuple (default is None)
        Sets the size of the plot figure.
    title_fontsize : str or int (default is :code:`large`)
        This is a :class:`matplotlib.axes.Axes` argument that
        specifies the size of the title. Available values are
        ['small', 'medium', 'large'] or integer values.
    text_fontsize : str or int (default is :code:`large`)
        This is a :class:`matplotlib.axes.Axes` argument that
        specifies the size of various texts around the plot.
        Available values are ['small', 'medium', 'large'] or integer
        values.
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :class:`matplotlib.axes.Axes`

    Returns
    -------
    :class:`matplotlib.axes._subplots.AxesSubplot`
        The axes on which the plot was drawn.
    """
    try:
        # Infer number of iterations based on the length
        # of the passed array
        iters = len(cost_history)

        # If no ax supplied, create new instance
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=figsize)

        # Plot with iters in x-axis and the cost in y-axis
        ax.plot(np.arange(iters), cost_history, 'k', lw=2, label=label)

        # Customize plot depending on parameters
        ax.set_title(title, fontsize=title_fontsize)
        ax.legend(fontsize=text_fontsize)
        ax.set_xlabel('Iterations', fontsize=text_fontsize)
        ax.set_ylabel('Cost', fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)
    except TypeError:
        logger.error('Please pass an iterable type. You are passing: {}'.format(type(cost_history)))
        raise
    else:
        return ax
