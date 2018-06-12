#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import os
import pytest
import numpy as np
import matplotlib as mpl

if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')

from matplotlib.axes._subplots import SubplotBase
from matplotlib.animation import FuncAnimation

# Import from package
from pyswarms.utils.plotters import (plot_cost_history)

@pytest.mark.parametrize('history', ['cost_history', 'mean_neighbor_history',
                                     'mean_pbest_history'])
def test_plot_cost_history_return_type(trained_optimizer, history):
    """Tests if plot_cost_history() returns a SubplotBase instance"""
    opt_params = vars(trained_optimizer)
    plot = plot_cost_history(opt_params[history])
    assert isinstance(plot, SubplotBase)
