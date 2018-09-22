#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import pytest
import numpy as np

# Import from package
from pyswarms.backend.topology import VonNeumann


@pytest.mark.parametrize("r", [0, 1])
@pytest.mark.parametrize("p", [1, 2])
def test_update_gbest_neighborhood(swarm, p, r):
    """Test if update_gbest_neighborhood gives the expected return values"""
    topology = VonNeumann()
    pos, cost = topology.compute_gbest(swarm, p=p, r=r)
    if p==2 and r==1:
        expected_pos = np.array([[ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.99959923e-01, -5.32665972e-03, -1.53685870e-02],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05]])
    elif p==2 and r==0:
        expected_pos = np.array([[ 9.98033031e-01,  4.97392619e-03,  3.07726256e-03],
                                 [ 1.00665809e+00,  4.22504014e-02,  9.84334657e-01],
                                 [ 1.12159389e-02,  1.11429739e-01,  2.86388193e-02],
                                 [ 1.64059236e-01,  6.85791237e-03, -2.32137604e-02],
                                 [ 9.93740665e-01, -6.16501403e-03, -1.46096578e-02],
                                 [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                 [ 1.12301876e-01,  1.77099784e-03,  1.45382457e-01],
                                 [ 4.41204876e-02,  4.84059652e-02,  1.05454822e+00],
                                 [ 9.89348409e-01, -1.31692358e-03,  9.88291764e-01],
                                 [ 9.99959923e-01, -5.32665972e-03, -1.53685870e-02]])
    elif p==1 and r==1:
        expected_pos = np.array([[9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.98033031e-01, 4.97392619e-03, 3.07726256e-03],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05],
                                 [9.90438476e-01, 2.50379538e-03, 1.87405987e-05]])
    elif p==1 and r==0:
        expected_pos = np.array([[ 9.98033031e-01,  4.97392619e-03,  3.07726256e-03],
                                 [ 1.00665809e+00,  4.22504014e-02,  9.84334657e-01],
                                 [ 1.12159389e-02,  1.11429739e-01,  2.86388193e-02],
                                 [ 1.64059236e-01,  6.85791237e-03, -2.32137604e-02],
                                 [ 9.93740665e-01, -6.16501403e-03, -1.46096578e-02],
                                 [ 9.90438476e-01,  2.50379538e-03,  1.87405987e-05],
                                 [ 1.12301876e-01,  1.77099784e-03,  1.45382457e-01],
                                 [ 4.41204876e-02,  4.84059652e-02,  1.05454822e+00],
                                 [ 9.89348409e-01, -1.31692358e-03,  9.88291764e-01],
                                 [ 9.99959923e-01, -5.32665972e-03, -1.53685870e-02]])
    expected_cost = 1.0002528364353296
    assert cost == pytest.approx(expected_cost)
    assert pos == pytest.approx(expected_pos)


@pytest.mark.parametrize("clamp", [None, (0, 1), (-1, 1)])
def test_compute_velocity_return_values(swarm, clamp):
    """Test if compute_velocity() gives the expected shape and range"""
    topology = VonNeumann(p=1, r=3)
    v = topology.compute_velocity(swarm, clamp)
    assert v.shape == swarm.position.shape
    if clamp is not None:
        assert (clamp[0] <= v).all() and (clamp[1] >= v).all()


@pytest.mark.parametrize(
    "bounds",
    [None, ([-5, -5, -5], [5, 5, 5]), ([-10, -10, -10], [10, 10, 10])],
)
def test_compute_position_return_values(swarm, bounds):
    """Test if compute_position() gives the expected shape and range"""
    topology = VonNeumann(p=1, r=2)
    p = topology.compute_position(swarm, bounds)
    assert p.shape == swarm.velocity.shape
    if bounds is not None:
        assert (bounds[0] <= p).all() and (bounds[1] >= p).all()


@pytest.mark.parametrize("r", [0, 1])
@pytest.mark.parametrize("p", [1, 2])
def test_neighbor_idx(swarm, p, r):
    """Test if the neighbor_idx attribute is assigned"""
    topology = VonNeumann(p=p, r=r)
    topology.compute_gbest(swarm)
    assert topology.neighbor_idx is not None


@pytest.mark.parametrize("m", [i for i in range(9)])
@pytest.mark.parametrize("n", [i for i in range(10)])
def test_delannoy_numbers(m, n):
    # fmt: off
    expected_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25,
                                41, 61, 63, 85, 113, 129, 145, 181, 231,
                                321, 377, 575, 681, 833, 1159, 1289,
                                1683, 2241, 3649, 3653, 5641, 7183,
                                8989, 13073, 19825, 40081, 48639, 75517,
                                108545, 22363, 224143, 265729, 598417])
    # fmt: on
    assert VonNeumann.delannoy(m, n) in expected_values
