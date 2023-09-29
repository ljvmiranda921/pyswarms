#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import pytest

from pyswarms.utils.search.random_search import RandomSearch


@pytest.mark.parametrize("maximum", [True, False])
def test_search_best_options_return_type(random_unbounded: RandomSearch, maximum: bool):
    """Tests if best options returns a dictionary"""
    _, best_options = random_unbounded.search(maximum)
    assert isinstance(best_options, dict)


def test_generate_grid_combinations(random_bounded: RandomSearch):
    """Test that the number of combinations in grid equals the number
    parameter selection iterations specficied"""
    expected = 100
    grid = random_bounded.generate_grid()
    assert len(grid) == expected


@pytest.mark.parametrize("option", ["c1", "c2", "w"])
def test_generate_grid_parameter_mapping(random_bounded: RandomSearch, option: str):
    """Test that generated values are correctly mapped to each parameter and
    are within the specified bounds"""
    grid = random_bounded.generate_grid()
    values: List[float] = [x[0][option] for x in grid]
    for val in values:
        assert val >= random_bounded.option_ranges[option][0]
        assert val <= random_bounded.option_ranges[option][1]
