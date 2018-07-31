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


def test_booth_dim_fail(outdim):
    """Test booth dim exception"""
    with pytest.raises(IndexError):
        fx.booth_func(outdim)


def test_bukin6_dim_fail(outdim):
    """Test bukin6 dim exception"""
    with pytest.raises(IndexError):
        fx.bukin6_func(outdim)


def test_crossintray_dim_fail(outdim):
    """Test crossintray dim exception"""
    with pytest.raises(IndexError):
        fx.crossintray_func(outdim)


def test_easom_dim_fail(outdim):
    """Test easom dim exception"""
    with pytest.raises(IndexError):
        fx.easom_func(outdim)

def test_goldstein_dim_fail(outdim):
    """Test goldstein dim exception"""
    with pytest.raises(IndexError):
        fx.goldstein_func(outdim)


def test_eggholder_dim_fail(outdim):
    """Test eggholder dim exception"""
    with pytest.raises(IndexError):
        fx.eggholder_func(outdim)


def test_himmelblau_dim_fail(outdim):
    """Test himmelblau dim exception"""
    with pytest.raises(IndexError):
        fx.himmelblau_func(outdim)


def test_holdertable_dim_fail(outdim):
    """Test holdertable dim exception"""
    with pytest.raises(IndexError):
        fx.holdertable_func(outdim)


def test_levi_dim_fail(outdim):
    """Test levi dim exception"""
    with pytest.raises(IndexError):
        fx.levi_func(outdim)


def test_matyas_dim_fail(outdim):
    """Test matyas dim exception"""
    with pytest.raises(IndexError):
        fx.matyas_func(outdim)


def test_schaffer2_dim_fail(outdim):
    """Test schaffer2 dim exception"""
    with pytest.raises(IndexError):
        fx.schaffer2_func(outdim)


def test_threehump_dim_fail(outdim):
    """Test threehump dim exception"""
    with pytest.raises(IndexError):
        fx.threehump_func(outdim)
