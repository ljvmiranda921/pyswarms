from typing import Dict, Optional, get_args

import numpy as np
import pytest

from pyswarms.backend.handlers import (
    BoundaryHandler,
    BoundaryStrategy,
    OptionsHandler,
    OptionsStrategy,
    VelocityHandler,
    VelocityStrategy,
)
from pyswarms.backend.velocity import SwarmOptions
from pyswarms.utils.types import Bounds, Clamp, Position, SwarmOption, Velocity

bh_strategies = list(get_args(BoundaryStrategy))
vh_strategies = list(get_args(VelocityStrategy))
oh_strategies = list(get_args(OptionsStrategy))


def test_out_of_bounds(bounds: Bounds, positions_inbound: Position, positions_out_of_bound: Position):
    bh = BoundaryHandler.factory("periodic", bounds)
    idx_in_bound = bh._out_of_bounds(positions_inbound)
    idx_out_of_bounds = bh._out_of_bounds(positions_out_of_bound)

    expected_idx = np.array([
        [True, False, False],
        [False, True, False],
        [True, True, False],
        [False, False, True],
        [False, True, False],
        [True, False, True],
    ])

    assert idx_in_bound.sum() == 0
    assert np.all(idx_out_of_bounds == expected_idx)


@pytest.mark.parametrize("strategy", bh_strategies)
def test_bound_handling(
    bounds: Bounds, positions_inbound: Position, positions_out_of_bound: Position, strategy: BoundaryStrategy
):
    bh = BoundaryHandler.factory(strategy, bounds)
    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound)
    lower_than_bound = outbound_handled >= bounds[0]
    greater_than_bound = outbound_handled <= bounds[1]
    assert lower_than_bound.all()
    assert greater_than_bound.all()


# def test_nearest_strategy(bounds, positions_inbound, positions_out_of_bound):
#     bh = BoundaryHandler(strategy="nearest")
#     # TODO Add strategy specific tests


# def test_reflective_strategy(bounds, positions_inbound, positions_out_of_bound):
#     bh = BoundaryHandler(strategy="reflective")
#     pass
#     # TODO Add strategy specific tests


# def test_shrink_strategy(bounds, positions_inbound, positions_out_of_bound):
#     bh = BoundaryHandler(strategy="shrink")
#     # TODO Add strategy specific tests


# def test_random_strategy(bounds, positions_inbound, positions_out_of_bound):
#     bh = BoundaryHandler(strategy="random")
#     # TODO Add strategy specific tests


# def test_intermediate_strategy(bounds, positions_inbound, positions_out_of_bound):
#     bh = BoundaryHandler(strategy="intermediate")
#     # TODO Add strategy specific tests


# def test_periodic_strategy(bounds, positions_inbound, positions_out_of_bound):
#     bh = BoundaryHandler(strategy="periodic")
#     # TODO Add strategy specific tests


def assert_clamp(
    clamp: Clamp,
    velocities_inbound: Velocity,
    velocities_out_of_bound: Velocity,
    positions_inbound: Position,
    positions_out_of_bound: Position,
    vh: VelocityHandler,
):
    # Test if it doesn't handle inclamp velocities
    inbound_handled = vh(velocities_inbound, positions_inbound)
    assert inbound_handled.all() == velocities_inbound.all()

    # Test if all particles are handled to a velocity inside the clamp
    outbound_handled = vh(velocities_out_of_bound, positions_out_of_bound)
    lower_than_clamp = outbound_handled < clamp[0]
    greater_than_clamp = outbound_handled > clamp[1]
    assert not lower_than_clamp.all()
    assert not greater_than_clamp.all()


def test_unmodified_strategy(clamp: Clamp, velocities_inbound: Velocity, velocities_out_of_bound: Velocity):
    vh = VelocityHandler.factory("unmodified", clamp)
    inbound_handled = vh(velocities_inbound, None)
    outbound_handled = vh(velocities_out_of_bound, None)
    assert inbound_handled.all() == velocities_inbound.all()
    assert outbound_handled.all() == velocities_out_of_bound.all()


def test_adjust_strategy(
    clamp: Clamp,
    velocities_inbound: Velocity,
    velocities_out_of_bound: Velocity,
    positions_inbound: Position,
    positions_out_of_bound: Position,
):
    vh = VelocityHandler.factory("adjust", clamp)
    assert_clamp(
        clamp,
        velocities_inbound,
        velocities_out_of_bound,
        positions_inbound,
        positions_out_of_bound,
        vh,
    )
    # TODO Add strategy specific tests
    pass


def test_invert_strategy(
    clamp: Clamp,
    velocities_inbound: Velocity,
    velocities_out_of_bound: Velocity,
    positions_inbound: Position,
    positions_out_of_bound: Position,
    bounds: Bounds,
):
    vh = VelocityHandler.factory("invert", clamp, bounds)
    assert_clamp(
        clamp,
        velocities_inbound,
        velocities_out_of_bound,
        positions_inbound,
        positions_out_of_bound,
        vh,
    )
    # TODO Add strategy specific tests
    pass


# def test_zero_strategy(
#     clamp,
#     velocities_inbound,
#     velocities_out_of_bound,
#     positions_inbound,
#     positions_out_of_bound,
#     bounds,
# ):
#     vh = VelocityHandler(strategy="zero")
#     # TODO Add strategy specific tests
#     pass


def assert_option_strategy(
    strategy: Dict[str, OptionsStrategy],
    init_opts: SwarmOptions,
    exp_opts: Dict[SwarmOption, float],
    end_opts: Dict[str, Optional[float]],
):
    """Test for any strategy for options handler
    strategy : strategy to use
    init_opts : dict with keys :code:`{'c1', 'c2', 'w'}`
    exp_opts: dict with expected values after strategy with given parameters
    kwargs: arguments to use for given strategy
    """
    assert len(init_opts) == len(exp_opts), "Size of initial options and expected options must be same"
    for opt, value in exp_opts.items():
        oh = OptionsHandler.factory(strategy[opt], opt, value, end_opts[opt])
        np.isclose(oh.__call__(100, 100), value, atol=0.001, rtol=0)


def test_option_strategy():
    init_opts = SwarmOptions({"c1": 0.5, "c2": 0.3, "w": 0.9})
    end_opts = {"c1": None, "c2": 0.1, "w": 0.2}  # use default for c1
    strategy: Dict[str, OptionsStrategy] = {"w": "exp_decay", "c1": "lin_variation", "c2": "nonlin_mod"}
    exp_opts: Dict[SwarmOption, float] = {"c1": 0.4, "c2": 0.1, "w": 0.567}

    assert_option_strategy(
        strategy,
        init_opts,
        exp_opts,
        end_opts,
    )
