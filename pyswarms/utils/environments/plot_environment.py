# -*- coding: utf-8 -*-

r"""
Plot environment for Optimizer Analysis

The class PlotEnvironment is built on top of :code:`matplotlib` in order
to render quick and easy plots for your optimizer. It can plot the best
cost for each iteration, and show animations of the particles in 2-D and
3-D space. Furthermore, because it has :code:`matplotlib` running under
the hood, the plots are easily customizable.

For example, if we want to plot the cost using PlotEnvironment, simply
pass the optimizer object when initializing the class, and the
PlotEnvironment will do a fresh run of your optimizer. After that,
various plotting methods can now be done:

.. code-block:: python
    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere_func
    from pyswarms.utils.environments import PlotEnvironment

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Pass optimizer  inside the environment. You also need to pass some
    # of the required arguments on how your optimizer will be evaluated.
    plt_env = PlotEnvironment(optimizer, sphere_func, 1000)

    # To plot the cost
    plt_env.plot_cost()
    plt.show()

In case you want to plot the particle movement, it is important that either
one of the :code:`matplotlib` animation :code:`Writers` is installed. These
doesn't come out of the box for :code:`pyswarms`, and must be installed
separately. For example, in a Linux or Windows distribution, you can
install :code:`ffmpeg` as

    >>> conda install -c conda-forge ffmpeg

Now, if you want to plot your particles in a 2-D environment, simply call
the following function:

    >>> plt_env.plot_particles2d()

You can also supply various arguments in this method: the indices of the
specific dimensions to be used, the limits of the axes, and the interval/
speed of animation.
"""

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import logging
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
from matplotlib import animation
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D


class PlotEnvironment(object):

    def assertions(self):
        """Assertion check"""
        # Check if the objective_func is a callable
        if not callable(self.objective_func):
            raise TypeError('Must pass a callable')

        # Check if getters exist in the optimizer
        if not (hasattr(self.optimizer, 'get_cost_history')
                & hasattr(self.optimizer, 'get_pos_history')
                & hasattr(self.optimizer, 'get_velocity_history')):
            raise AttributeError('Missing getters in optimizer, check '
                                 'pyswarms.base module')

        # Check if important methods exist in the optimizer
        if not (hasattr(self.optimizer, 'optimize')
                & hasattr(self.optimizer, 'reset')):
            raise AttributeError('Missing methods in optimizer, check '
                                 'pyswarms.base module')

    def __init__(self, optimizer, objective_func, iters):
        """Runs the optimizer against an objective function for a number
        of iterations

        Upon initialization, the :code:`optimize` method of the optimizer
        will be called, passing the arguments :code:`objective_func` and
        :code:`iters`. The results of the optimization scheme is then
        stored as attributes of this class.

        Parameters
        ----------
        optimizer : object instance
            An instance of an optimizer class that was derived from any
            of the :mod:`pyswarms.base` classes.
        objective_func : method
            An objective function to be optimized using the :code:`optimizer`.
            This argument is passed to the :code:`optimize` method of the
            :code:`optimizer`.
        iters : int
            The number of iterations to run the optimizer. This argument
            is passed to the :code:`optimize` method of the :code:`optimizer`.
        """
        self.logger = logging.getLogger(__name__)
        # Store the arguments
        self.optimizer = optimizer
        self.objective_func = objective_func
        self.iters = iters
        # Check assertions
        self.assertions()
        # Run the optimizer
        self.optimizer.reset()
        self.status = self.optimizer.optimize(objective_func, iters, 1, 0)
        # Initialize tuples for particle plotting
        self.Index = namedtuple('Index', ['x', 'y', 'z'])
        self.Limit = namedtuple('Limit', ['x', 'y', 'z'])
        self.Label = namedtuple('Label', ['x', 'y', 'z'])

    def plot_cost(self, title='Cost History', ax=None, figsize=None,
                  title_fontsize="large", text_fontsize="medium", **kwargs):
        """Creates a simple line plot with the cost in the y-axis and
        the iteration at the x-axis

        Parameters
        ----------
        title : str (default is :code:`'Cost History'`)
            The title of the plotted graph.
        ax : :class:`matplotlib.axes.Axes` (default is :code:`None`)
            The axes where the plot is to be drawn. If :code:`None` is
            passed, then the plot will be drawn to a new set of axes.
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
        # Get cost history from the optimizer method
        cost_history = self.optimizer.get_cost_history
        mean_pbest_history = self.optimizer.get_mean_pbest_history
        mean_neighbor_history = self.optimizer.get_mean_neighbor_history

        # If ax is default, then create new plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot with self.iters as x-axis and cost_history as
        # y-axis.
        ax.plot(np.arange(self.iters), cost_history, 'k', lw=2,
                label='Best cost')
        ax.plot(np.arange(self.iters), mean_pbest_history, 'k--', lw=2,
                label='Avg. personal best cost')
        ax.plot(np.arange(self.iters), mean_neighbor_history, 'k:', lw=2,
                label='Avg. neighborhood cost')

        # Customize plot depending on parameters
        ax.set_title(title, fontsize=title_fontsize)
        ax.legend(fontsize=text_fontsize)
        ax.set_xlabel('Iterations', fontsize=text_fontsize)
        ax.set_ylabel('Cost', fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)

        return ax

    def plot_particles2D(self, index=(0, 1), limits=((-1, 1), (-1, 1)),
                         labels=('x-axis', 'y-axis'), interval=80,
                         title='Particle Movement in 2D space',
                         ax=None, figsize=None, title_fontsize="large",
                         text_fontsize="medium"):
        """Creates an animation of particle movement in 2D-space

        Parameters
        ----------
        index : n-tuple (default is :code:`(0,1)`)
            The index in which a specific dimension will be plotted. For
            example, :code:`(idx_1, idx_2)` for two dimensions.
        limits : n-tuple of 2-tuples (default is :code:`((-1,1),(-1,1))`)
            The limits of the x-y axes for 2D. For example,
            :code:`((xmin, xmax),(ymin, ymax))`
        labels : 2-tuple (default is :code:`('x-axis', 'y-axis')`
            Sets the x and y labels of the 2D plot. For example,
            :code:`('label_x_axis', 'label_y_axis')`
        interval : int (default is 80)
            The speed of update, in milliseconds
        title : str (default is :code:`'Particle Movement in 2D space'`)
            The title of the plotted graph.
        ax : :class:`matplotlib.axes.Axes` (default is :code:`None`)
            The axes where the plot is to be drawn. If :code:`None` is
            passed, then the plot will be drawn to a new set of axes.
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
        :class:`matplotlib.animation.FuncAnimation`
            The drawn animation that can be saved to mp4 or other
            third-party tools
        """
        # Check inconsistencies with input
        if not (len(index) == len(limits) == 2):
            raise ValueError('The index and limits should be of length 2')

        # Set-up tuples for plotting environment
        idx = self.Index(x=index[0], y=index[1], z=None)
        lmt = self.Limit(x=limits[0], y=limits[1], z=None)
        lbl = self.Label(x=labels[0], y=labels[1], z=None)

        # If ax is default, then create new plot. Set-up the figure, the
        # acis, and the plot element that we want to animate
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set plot title
        ax.set_title(title, fontsize=title_fontsize)

        # Set plot labels
        ax.set_xlabel(lbl.x, fontsize=text_fontsize)
        ax.set_ylabel(lbl.y, fontsize=text_fontsize)

        # Set plot limits
        ax.set_xlim(lmt.x)
        ax.set_ylim(lmt.y)

        # Plot data
        plot = ax.scatter(x=[], y=[], c='red')
        data = self.optimizer.get_pos_history

        # Get the number of iterations
        n_iters = self.optimizer.get_pos_history.shape[0]

        # Perform animation
        anim = animation.FuncAnimation(fig, func=self._animate2D,
                                       frames=xrange(n_iters),
                                       fargs=(data, plot, idx),
                                       interval=interval, blit=True)
        return anim

    def plot_particles3D(self, index=(0, 1, 2),
                         limits=((-1, 1), (-1, 1), (-1, 1)),
                         labels=('x-axis', 'y-axis', 'z-axis'),
                         interval=80,
                         title='Particle Movement in 3D space', ax=None,
                         figsize=None, title_fontsize="large",
                         text_fontsize="medium"):
        """Creates an animation of particle movement in 2D-space

        Parameters
        ----------
        index : n-tuple (default is :code:`(0,1,2)`)
            The index in which a specific dimension will be plotted. For
            example, :code:`(idx_1, idx_2, idx_3)` for three dimensions.
        limits : n-tuple of 2-tuples (default is
                 :code:`((-1,1),(-1,1),(-1,1))`)
            The limits of the x-y axes for 3D. For example,
            :code:`((xmin, xmax),(ymin, ymax))`
        labels : 2-tuple (default is :code:`('x-axis', 'y-axis', 'z-axis')`
            Sets the x and y labels of the 2D plot. For example,
            :code:`('label_x_axis', 'label_y_axis', 'label_z_axis')`
        interval : int (default is 80)
            The speed of update, in milliseconds
        title : str (default is :code:`'Particle Movement in 3D space'`)
            The title of the plotted graph.
        ax : :class:`matplotlib.axes.Axes` (default is :code:`None`)
            The axes where the plot is to be drawn. If :code:`None` is
            passed, then the plot will be drawn to a new set of axes.
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
        :class:`matplotlib.animation.FuncAnimation`
            The drawn animation that can be saved to mp4 or other
            third-party tools
        """
        # Check inconsistencies with input
        if not (len(index) == len(limits) == 3):
            raise ValueError('The index and limits should be of length 3')

        # Set-up tuples for plotting environment
        idx = self.Index(x=index[0], y=index[1], z=index[2])
        lmt = self.Limit(x=limits[0], y=limits[1], z=limits[2])
        lbl = self.Label(x=labels[0], y=labels[1], z=labels[2])

        # If ax is default, then create new plot. Set-up the figure, the
        # acis, and the plot element that we want to animate
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax = Axes3D(fig)

        # Set plot title
        ax.set_title(title, fontsize=title_fontsize)

        # Set plot axes labels
        ax.set_xlabel(lbl.x, fontsize=text_fontsize)
        ax.set_ylabel(lbl.y, fontsize=text_fontsize)
        ax.set_zlabel(lbl.z, fontsize=text_fontsize)

        # Set plot limits
        ax.set_xlim(lmt.x)
        ax.set_ylim(lmt.y)
        ax.set_zlim(lmt.z)

        # Plot data
        plot = ax.scatter(xs=[], ys=[], zs=[], c='red')
        data = self.optimizer.get_pos_history

        # Get the number of iterations
        n_iters = self.optimizer.get_pos_history.shape[0]

        # Perform animation
        anim = animation.FuncAnimation(fig, func=self._animate3D,
                                       frames=xrange(n_iters),
                                       fargs=(data, plot, idx),
                                       interval=interval)
        return anim

    def _animate2D(self, i, data, plot, idx):
        """Helper animation function that is called seqentially
        :class:`matplotlib.animation.FuncAnimation`

        Parameters
        ----------
        i : int
            Required argument for :code:`matplotlib.animation.FuncAnimation`,
            basis for indexing the current position of the swarm.
        data : numpy.ndarray
            The position matrix where the particles' position
            will be taken from.
        plot : matplotlib.Axes
            The plot environment where the update operations will be drawn
        idx : namedtuple
            The chosen indices for plotting the dimensions

        Returns
        -------
        :class:`matplotlib.artist.Artist`
            iterable of artists
        """
        current_pos = data[i]
        xy = current_pos[:, (idx.x, idx.y)]
        plot.set_offsets(xy)
        return plot,

    def _animate3D(self, i, data, plot, idx):
        """Helper animation function that is called seqentially
        :class:`matplotlib.animation.FuncAnimation`

        Parameters
        ----------
        i : int
            Required argument for :code:`matplotlib.animation.FuncAnimation`,
            basis for indexing the current position of the swarm.
        data : numpy.ndarray
            The position matrix where the particles' position
            will be taken from.
        plot : matplotlib.Axes
            The plot environment where the update operations will be drawn
        idx : namedtuple
            The chosen indices for plotting the dimensions

        Returns
        -------
        :class:`matplotlib.artist.Artist`
            iterable of artists
        """
        current_pos = data[i]
        x, y, z = current_pos[:, idx.x], current_pos[:, idx.y], \
            current_pos[:, idx.z]
        plot._offsets3d = (x, y, z)
        return plot,
