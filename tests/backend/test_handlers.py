import pytest
import numpy as np

from pyswarms.backend.handlers import BoundaryHandler

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

