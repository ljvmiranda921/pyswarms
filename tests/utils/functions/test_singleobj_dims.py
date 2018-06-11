#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pyswarms` package."""

# Import modules
import pytest
import numpy as np
from collections import namedtuple

# Import from package
from pyswarms.utils.functions import single_obj as fx

def test_beale_dim_fail(outdim):
    """Test beale dim exception"""
    with pytest.raises(IndexError):
        fx.beale_func(outdim)

def test_goldstein_dim_fail(outdim):
    """Test goldstein dim exception"""
    with pytest.raises(IndexError):
        fx.goldstein_func(outdim)

def test_booth_dim_fail(outdim):
    """Test booth dim exception"""
    with pytest.raises(IndexError):
        fx.booth_func(outdim)

def test_bukin6_dim_fail(outdim):
    """Test bukin6 dim exception"""
    with pytest.raises(IndexError):
        fx.bukin6_func(outdim)

def test_matyas_dim_fail(outdim):
    """Test matyas dim exception"""
    with pytest.raises(IndexError):
        fx.matyas_func(outdim)

def test_levi_dim_fail(outdim):
    """Test levi dim exception"""
    with pytest.raises(IndexError):
        fx.levi_func(outdim)

def test_schaffer2_dim_fail(outdim):
    """Test schaffer2 dim exception"""
    with pytest.raises(IndexError):
        fx.schaffer2_func(outdim)