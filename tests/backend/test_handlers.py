import pytest
import inspect
import numpy as np

from pyswarms.backend.handlers import (
    BoundaryHandler,
    VelocityHandler,
    HandlerMixin,
)
import pyswarms.backend.handlers as h

bh_strategies = [name for name, _ in inspect.getmembers(h.BoundaryHandler(""),
    predicate=inspect.ismethod) if not name.startswith(("__", "_"))]
vh_strategies = [name for name, _ in inspect.getmembers(h.VelocityHandler(""),
    predicate=inspect.ismethod) if not name.startswith(("__", "_"))]

def test_out_of_bounds(bounds, positions_inbound, positions_out_of_bound):
    hm = HandlerMixin()
    out_of_bounds = hm._out_of_bounds
    idx_inbound = out_of_bounds(positions_inbound, bounds)
    idx_out_of_bounds = out_of_bounds(positions_out_of_bound, bounds)

    print(bh_strategies)
    expected_idx = (
        (np.array([2, 3, 5]), np.array([1, 2, 0])),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 1, 1, 2, 1, 0])),
    )
    assert np.ravel(idx_inbound[0]).size == 0
    assert np.ravel(idx_inbound[1]).size == 0
    assert (
        np.ravel(idx_out_of_bounds[0]).all() == np.ravel(expected_idx[0]).all()
    )
    assert (
        np.ravel(idx_out_of_bounds[1]).all() == np.ravel(expected_idx[1]).all()
    )

@pytest.mark.parametrize(
    "strategy", bh_strategies
)
def test_bound_handling(
    bounds, positions_inbound, positions_out_of_bound, strategy
):
    bh = BoundaryHandler(strategy=strategy)
    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound, bounds)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound, bounds)
    lower_than_bound = outbound_handled >= bounds[0]
    greater_than_bound = outbound_handled <= bounds[1]
    assert lower_than_bound.all()
    assert greater_than_bound.all()

def test_nearest_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="nearest")
    # TODO Add strategy specific tests


def test_reflective_strategy(
    bounds, positions_inbound, positions_out_of_bound
):
    bh = BoundaryHandler(strategy="reflective")
    pass
    # TODO Add strategy specific tests


def test_shrink_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="shrink")
    # TODO Add strategy specific tests


def test_random_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="random")
    # TODO Add strategy specific tests


def test_intermediate_strategy(
    bounds, positions_inbound, positions_out_of_bound
):
    bh = BoundaryHandler(strategy="intermediate")
    # TODO Add strategy specific tests


def test_periodic_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="periodic")
    # TODO Add strategy specific tests

def assert_clamp(
clamp,
velocities_inbound,
velocities_out_of_bound,
positions_inbound,
positions_out_of_bound,
vh,
bounds=None,
):
# Test if it doesn't handle inclamp velocities
    inbound_handled = vh(
        velocities_inbound, clamp, position=positions_inbound, bounds=bounds
    )
    assert inbound_handled.all() == velocities_inbound.all()

    # Test if all particles are handled to a velocity inside the clamp
    outbound_handled = vh(
        velocities_out_of_bound,
        clamp,
        position=positions_out_of_bound,
        bounds=bounds,
    )
    lower_than_clamp = outbound_handled < clamp[0]
    greater_than_clamp = outbound_handled > clamp[1]
    assert not lower_than_clamp.all()
    assert not greater_than_clamp.all()




def test_unmodified_strategy(
    clamp, velocities_inbound, velocities_out_of_bound
):
    vh = VelocityHandler(strategy="unmodified")
    inbound_handled = vh(velocities_inbound, clamp)
    outbound_handled = vh(velocities_out_of_bound, clamp)
    assert inbound_handled.all() == velocities_inbound.all()
    assert outbound_handled.all() == velocities_out_of_bound.all()


def test_adjust_strategy(
    clamp,
    velocities_inbound,
    velocities_out_of_bound,
    positions_inbound,
    positions_out_of_bound,
):
    vh = VelocityHandler(strategy="adjust")
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
    clamp,
    velocities_inbound,
    velocities_out_of_bound,
    positions_inbound,
    positions_out_of_bound,
    bounds,
):
    vh = VelocityHandler(strategy="invert")
    assert_clamp(
        clamp,
        velocities_inbound,
        velocities_out_of_bound,
        positions_inbound,
        positions_out_of_bound,
        vh,
        bounds=bounds,
    )
    # TODO Add strategy specific tests
    pass


def test_zero_strategy(
    clamp,
    velocities_inbound,
    velocities_out_of_bound,
    positions_inbound,
    positions_out_of_bound,
    bounds,
):
    vh = VelocityHandler(strategy="zero")
    # TODO Add strategy specific tests
    pass
