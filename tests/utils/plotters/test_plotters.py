#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import from standard library
import os

# Import modules
import pytest
import matplotlib as mpl

# Set $DISPLAY environmental variable
if os.environ.get("DISPLAY", "") == "":
    print("No display found. Using non-interactive Agg backend.")
    mpl.use("Agg")

from matplotlib.animation import FuncAnimation
from matplotlib.axes._subplots import SubplotBase

# Import from pyswarms
from pyswarms.utils.plotters import (
    plot_contour,
    plot_cost_history,
    plot_surface,
)
from pyswarms.utils.plotters.plotters import _animate, _mesh


@pytest.mark.parametrize(
    "history", ["cost_history", "mean_neighbor_history", "mean_pbest_history"]
)
def test_plot_cost_history_return_type(trained_optimizer, history):
    """Tests if plot_cost_history() returns a SubplotBase instance"""
    opt_params = vars(trained_optimizer)
    plot = plot_cost_history(opt_params[history])
    assert isinstance(plot, SubplotBase)


@pytest.mark.parametrize("bad_values", [2, 43.14])
def test_plot_cost_history_error(bad_values):
    """Tests if plot_cost_history() raises an error given bad values"""
    with pytest.raises(TypeError):
        plot_cost_history(bad_values)


def test_plot_contour_return_type(pos_history):
    """Tests if the animation function returns the expected type"""
    assert isinstance(plot_contour(pos_history), FuncAnimation)


def test_plot_surface_return_type(pos_history):
    """Tests if the animation function returns the expected type"""
    assert isinstance(plot_surface(pos_history), FuncAnimation)


def test_mesh_hidden_function_shape(mesher):
    """Tests if the hidden _mesh() function returns the expected shape"""
    xx, yy, zz = _mesh(mesher)
    assert xx.shape == yy.shape == zz.shape == (20, 20)


def test_parallel_mesh(mesher):
    """Test if parallelization breaks the optimization process"""
    import multiprocessing

    xx, yy, zz = _mesh(mesher)
    xx_p, yy_p, zz_p = _mesh(mesher, n_processes=multiprocessing.cpu_count())
    assert (
        xx.shape
        == yy.shape
        == zz.shape
        == xx_p.shape
        == yy_p.shape
        == zz_p.shape
        == (20, 20)
    )


def test_animate_hidden_function_type(pos_history):
    """Tests if the hidden _animate() function returns the expected type"""
    fig, ax = mpl.pyplot.subplots(1, 1)
    ax = mpl.pyplot.scatter(x=[], y=[])
    return_plot = _animate(i=1, data=pos_history, plot=ax)
    assert isinstance(return_plot, tuple)
