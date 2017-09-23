#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import from __future__
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import print_function

# Import modules
import os
import unittest
import numpy as np
from mock import Mock
import matplotlib as mpl
from matplotlib.axes._subplots import SubplotBase
from matplotlib.animation import FuncAnimation

# Fix for display
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')

# Import from package
from pyswarms.utils.environments import PlotEnvironment
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions.single_obj import sphere_func

class Base(unittest.TestCase):

    def setUp(self):
        """Sets up test fixtures"""
        self.optimizer = GlobalBestPSO(n_particles=10,dimensions=3,
            options={'c1':0.5,'c2':0.3,'w':0.9})
        self.class_methods = [
            'get_cost_history',
            'get_pos_history',
            'get_velocity_history',
            'optimize',
            'reset']
        self.get_specs = lambda idx: [x for i,x in enumerate(self.class_methods) if i!=idx]
        self.plt_env = PlotEnvironment(self.optimizer, sphere_func, 1000)

class Instantiation(Base):

    def test_objective_function_fail(self):
        """Tests if exception is thrown when objective function is not callable"""
        pass_a_list = [1, 2, 3, 4, 5]
        pass_a_dict = {'a':0, 'b':2, 'c':4}
        with self.assertRaises(TypeError):
            plt_env = PlotEnvironment(self.optimizer, pass_a_list, 100)
        with self.assertRaises(TypeError):
            plt_env = PlotEnvironment(self.optimizer, pass_a_dict, 100)

    def test_optimizer_getters_get_cost_history_fail(self):
        """Test if exception is thrown if get_cost_history is missing"""
        m = Mock(spec=self.get_specs(0))
        with self.assertRaises(AttributeError):
            plt_env = PlotEnvironment(m, sphere_func, 100)

    def test_optimizer_getters_get_pos_history_fail(self):
        """Test if exception is thrown if get_cost_history is missing"""
        m = Mock(spec=self.get_specs(1))
        with self.assertRaises(AttributeError):
            plt_env = PlotEnvironment(m, sphere_func, 100)

    def test_optimizer_getters_get_velo_history_fail(self):
        """Test if exception is thrown if get_velocity_history is missing"""
        m = Mock(spec=self.get_specs(2))
        with self.assertRaises(AttributeError):
            plt_env = PlotEnvironment(m, sphere_func, 100)

    def test_attribute_optimize_fail(self):
        """Test if exception is thrown if optimize attr is missing"""
        m = Mock(spec=self.get_specs(3))
        with self.assertRaises(AttributeError):
            plt_env = PlotEnvironment(m, sphere_func, 100)

    def test_attribute_reset_fail(self):
        """Test if exception is thrown if reset attr is missing"""
        m = Mock(spec=self.get_specs(4))
        with self.assertRaises(AttributeError):
            plt_env = PlotEnvironment(m, sphere_func, 100)

class MethodReturnTypes(Base):

    def test_plot_cost_return_type(self):
        """Test if plot_cost() returns a SubplotBase instance"""
        # It turns out that AxesSubplot is just a factory-created class,
        # that's why checking on it explicitly is impossible.
        self.assertIsInstance(self.plt_env.plot_cost(), SubplotBase)

    def test_plot2D_return_type(self):
        """Test if plot_particles2D() returns a FuncAnimation instance"""
        self.assertIsInstance(self.plt_env.plot_particles2D(), FuncAnimation)

    def test_plot3D_return_type(self):
        """Test if plot_particles3D() returns a FuncAnimation instance"""
        self.assertIsInstance(self.plt_env.plot_particles3D(), FuncAnimation)