#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest


@pytest.mark.parametrize("maximum", [True, False])
def test_search_best_options_return_type(grid, maximum):
    """Tests if best options returns a dictionary"""
    _, best_options = grid.search(maximum)
    assert isinstance(best_options, dict)


def test_grid_output(grid_mini):
    """Tests if generate_grid function returns expected value"""
    expected = [
        {"c1": 1, "c2": 6, "k": 5, "w": 0.9, "p": 0},
        {"c1": 2, "c2": 6, "k": 5, "w": 0.9, "p": 0},
    ]
    assert grid_mini.generate_grid() == expected
