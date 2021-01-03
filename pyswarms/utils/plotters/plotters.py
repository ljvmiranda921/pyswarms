# -*- coding: utf-8 -*-

"""
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
    from pyswarms.utils.functions.single_obj import sphere
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
    from pyswarms.utils.functions.single_obj import sphere
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain pos history from optimizer instance
    pos_history = optimizer.pos_history

    # Plot!
    plot_contour(pos_history)

You can also supply various arguments in this method: the indices of the
specific dimensions to be used, the limits of the axes, and the interval/
speed of animation.
"""

# Import standard library
import logging

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D

from ..reporter import Reporter
from .formatters import Animator, Designer

rep = Reporter(logger=logging.getLogger(__name__))


def plot_cost_history(
    cost_history, ax=None, title="Cost History", designer=None, **kwargs
):
    """Create a simple line plot with the cost in the y-axis and
    the iteration at the x-axis

    Parameters
    ----------
    cost_history : array_like
        Cost history of shape :code:`(iters, )` or length :code:`iters` where
        each element contains the cost for the given iteration.
    ax : :obj:`matplotlib.axes.Axes`, optional
        The axes where the plot is to be drawn. If :code:`None` is
        passed, then the plot will be drawn to a new set of axes.
    title : str, optional
        The title of the plotted graph. Default is `Cost History`
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :class:`matplotlib.axes.Axes`

    Returns
    -------
    :obj:`matplotlib.axes._subplots.AxesSubplot`
        The axes on which the plot was drawn.
    """
    try:
        # Infer number of iterations based on the length
        # of the passed array
        iters = len(cost_history)

        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(legend="Cost", label=["Iterations", "Cost"])

        # If no ax supplied, create new instance
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=designer.figsize)

        # Plot with iters in x-axis and the cost in y-axis
        ax.plot(
            np.arange(iters), cost_history, "k", lw=2, label=designer.legend
        )

        # Customize plot depending on parameters
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.legend(fontsize=designer.text_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.tick_params(labelsize=designer.text_fontsize)
    except TypeError:
        rep.logger.exception("Please check your input type")
        raise
    else:
        return ax


def plot_contour(
    pos_history,
    canvas=None,
    title="Trajectory",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
    n_processes=None,
    **kwargs
):
    """Draw a 2D contour map for particle trajectories

    Here, the space is represented as a flat plane. The contours indicate the
    elevation with respect to the objective function. This works best with
    2-dimensional swarms with their fitness in z-space.

    Parameters
    ----------
    pos_history : numpy.ndarray or list
        Position history of the swarm with shape
        :code:`(iteration, n_particles, dimensions)`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking
        the optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :obj:`matplotlib.axes.Axes` plotting function

    Returns
    -------
    :obj:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """

    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(
                limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"]
            )

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(1, 1, figsize=designer.figsize)
        else:
            fig, ax = canvas

        # Get number of iterations
        n_iters = len(pos_history)

        # Customize plot
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])

        # Make a contour map if possible
        if mesher is not None:
            (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
            ax.contour(xx, yy, zz, levels=mesher.levels)

        # Mark global best if possible
        if mark is not None:
            ax.scatter(mark[0], mark[1], color="red", marker="x")

        # Put scatter skeleton
        plot = ax.scatter(x=[], y=[], c="black", alpha=0.6, **kwargs)

        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters),
            fargs=(pos_history, plot),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        rep.logger.exception("Please check your input type")
        raise
    else:
        return anim


def plot_surface(
    pos_history,
    canvas=None,
    title="Trajectory",
    designer=None,
    mesher=None,
    animator=None,
    mark=None,
    n_processes=None,
    **kwargs
):
    """Plot a swarm's trajectory in 3D

    This is useful for plotting the swarm's 2-dimensional position with
    respect to the objective function. The value in the z-axis is the fitness
    of the 2D particle when passed to the objective function. When preparing the
    position history, make sure that the:

    * first column is the position in the x-axis,
    * second column is the position in the y-axis; and
    * third column is the fitness of the 2D particle

    The :class:`pyswarms.utils.plotters.formatters.Mesher` class provides a
    method that prepares this history given a 2D pos history from any
    optimizer.

    .. code-block:: python

        import pyswarms as ps
        from pyswarms.utils.functions.single_obj import sphere
        from pyswarms.utils.plotters import plot_surface
        from pyswarms.utils.plotters.formatters import Mesher

        # Run optimizer
        options = {'c1':0.5, 'c2':0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options)

        # Prepare position history
        m = Mesher(func=sphere)
        pos_history_3d = m.compute_history_3d(optimizer.pos_history)

        # Plot!
        plot_surface(pos_history_3d)

    Parameters
    ----------
    pos_history : numpy.ndarray
        Position history of the swarm with shape
        :code:`(iteration, n_particles, 3)`
    objective_func : callable
        The objective function that takes a swarm of shape
        :code:`(n_particles, 2)` and returns a fitness array
        of :code:`(n_particles, )`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking the
        optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :class:`matplotlib.axes.Axes` plotting function

    Returns
    -------
    :class:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """
    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(
                limits=[(-1, 1), (-1, 1), (-1, 1)],
                label=["x-axis", "y-axis", "z-axis"],
                colormap=cm.viridis,
            )

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # Get number of iterations
        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(1, 1, figsize=designer.figsize)
        else:
            fig, ax = canvas

        # Initialize 3D-axis
        ax = Axes3D(fig)

        n_iters = len(pos_history)

        # Customize plot
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.set_zlabel(designer.label[2], fontsize=designer.text_fontsize)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])
        ax.set_zlim(designer.limits[2])

        # Make a contour map if possible
        if mesher is not None:
            (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
            ax.plot_surface(
                xx, yy, zz, cmap=designer.colormap, alpha=mesher.alpha
            )

        # Mark global best if possible
        if mark is not None:
            ax.scatter(mark[0], mark[1], mark[2], color="red", marker="x")

        # Put scatter skeleton
        plot = ax.scatter(xs=[], ys=[], zs=[], c="black", alpha=0.6, **kwargs)

        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters),
            fargs=(pos_history, plot),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        rep.logger.exception("Please check your input type")
        raise
    else:
        return anim


def _animate(i, data, plot):
    """Helper animation function that is called sequentially
    :class:`matplotlib.animation.FuncAnimation`
    """
    current_pos = data[i]
    if np.array(current_pos).shape[1] == 2:
        plot.set_offsets(current_pos)
    else:
        plot._offsets3d = current_pos.T
    return (plot,)


def _mesh(mesher, n_processes=None):
    """Helper function to make a mesh"""
    xlim = mesher.limits[0]
    ylim = mesher.limits[1]
    x = np.arange(xlim[0], xlim[1], mesher.delta)
    y = np.arange(ylim[0], ylim[1], mesher.delta)
    xx, yy = np.meshgrid(x, y)
    xypairs = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

    # Get z-value

    # Setup Pool of processes for parallel evaluation
    pool = None if n_processes is None else mp.Pool(n_processes)

    if pool is None:
        z = mesher.func(xypairs)
    else:
        results = pool.map(
            mesher.func, np.array_split(xypairs, pool._processes)
        )
        z = np.concatenate(results)

    # Close Pool of Processes
    if n_processes is not None:
        pool.close()

    zz = z.reshape(xx.shape)
    return (xx, yy, zz)
