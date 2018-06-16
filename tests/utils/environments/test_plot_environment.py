#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import os
import pytest
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "":
    mpl.use("Agg")

from matplotlib.axes._subplots import SubplotBase
from matplotlib.animation import FuncAnimation

# Import from package
from pyswarms.utils.environments import PlotEnvironment
from pyswarms.utils.functions.single_obj import sphere_func

class_methods = [
    "cost_history",
    "pos_history",
    "velocity_history",
    "optimize",
    "reset",
]


@pytest.mark.parametrize("attributes", [i for i in enumerate(class_methods)])
def test_getters_pso(mock_pso, attributes):
    """Tests an instance of the PSO class and should raise an exception when the class has missing attributes"""
    idx, _ = attributes
    with pytest.raises(AttributeError):
        m = mock_pso(idx)
        PlotEnvironment(m, sphere_func, 100)


@pytest.mark.xfail
def test_plot_cost_return_type(plot_environment):
    """Tests if plot_cost() returns a SubplotBase instance"""
    assert isinstance(plot_environment.plot_cost(), SubplotBase)


@pytest.mark.xfail
def test_plot2D_return_type(plot_environment):
    """Test if plot_particles2D() returns a FuncAnimation instance"""
    assert isinstance(plot_environment.plot_particles2D(), FuncAnimation)


@pytest.mark.xfail
def test_plot3D_return_type(plot_environment):
    """Test if plot_particles3D() returns a FuncAnimation instance"""
    assert isinstance(plot_environment.plot_particles3D(), FuncAnimation)
