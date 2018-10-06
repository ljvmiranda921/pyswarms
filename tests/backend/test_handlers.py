import pytest
import numpy as np

from pyswarms.backend.handlers import BoundaryHandler, VelocityHandler, HandlerMixin

def test_out_of_bounds(bounds, positions_inbound, positions_out_of_bound):
    hm = HandlerMixin()
    out_of_bounds = hm._HandlerMixin__out_of_bounds
    idx_inbound = out_of_bounds(positions_inbound, bounds)
    idx_out_of_bounds = out_of_bounds(positions_out_of_bounds, bounds)

    expected_idx = np.array([[0, 1],
                             [1, 2],
                             [2, 1],
                             [2, 2],
                             [3, 3],
                             [4, 2],
                             [5, 1],
                             [5, 3]
                            ])
    assert idx_inbound.all() == None
    assert idx_out_of_bounds.all() == expected_idx

def test_nearest_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="nearest")

    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound, bounds)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound, bounds)
    lower_than_bound = outbound_handled < bounds[0]
    greater_than_bound = outbound_handled > bounds[1]
    assert not lower_than_bound.all()
    assert not greater_than_bound.all()


    # TODO Add strategy specific tests

def test_reflective_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="reflective")

    # Test if it doesn't handle inbound positions
    # inbound_handled = bh(positions_inbound, bounds)
    # assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    # outbound_handled = bh(positions_out_of_bound, bounds)
    # lower_than_bound = outbound_handled < bounds[0]
    # greater_than_bound = outbound_handled > bounds[1]
    # assert not lower_than_bound.all()
    # assert not greater_than_bound.all()
    pass


    # TODO Add strategy specific tests

def test_shrink_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="shrink")

    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound, bounds)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound, bounds)
    lower_than_bound = outbound_handled < bounds[0]
    greater_than_bound = outbound_handled > bounds[1]
    assert not lower_than_bound.all()
    assert not greater_than_bound.all()


    # TODO Add strategy specific tests

def test_random_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="random")

    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound, bounds)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound, bounds)
    lower_than_bound = outbound_handled < bounds[0]
    greater_than_bound = outbound_handled > bounds[1]
    assert not lower_than_bound.all()
    assert not greater_than_bound.all()


    # TODO Add strategy specific tests

def test_intermediate_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="intermediate")

    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound, bounds)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound, bounds)
    lower_than_bound = outbound_handled < bounds[0]
    greater_than_bound = outbound_handled > bounds[1]
    assert not lower_than_bound.all()
    assert not greater_than_bound.all()


    # TODO Add strategy specific tests

def test_periodic_strategy(bounds, positions_inbound, positions_out_of_bound):
    bh = BoundaryHandler(strategy="periodic")

    # Test if it doesn't handle inbound positions
    inbound_handled = bh(positions_inbound, bounds)
    assert inbound_handled.all() == positions_inbound.all()

    # Test if all particles are handled to a position inside the boundaries
    outbound_handled = bh(positions_out_of_bound, bounds)
    lower_than_bound = outbound_handled < bounds[0]
    greater_than_bound = outbound_handled > bounds[1]
    assert not lower_than_bound.all()
    assert not greater_than_bound.all()


    # TODO Add strategy specific tests

