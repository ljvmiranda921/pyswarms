#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import modules
import numpy as np
import pytest

# Import from pyswarms
from pyswarms.backend.topology import VonNeumann
from .abc_test_topology import ABCTestTopology

np.random.seed(4135157)


class TestVonNeumannTopology(ABCTestTopology):
    @pytest.fixture
    def topology(self):
        return VonNeumann

    @pytest.fixture
    def options(self):
        return {"p": 1, "r": 1}

    @pytest.mark.parametrize("r", [0, 1])
    @pytest.mark.parametrize("p", [1, 2])
    def test_update_gbest_neighborhood(self, swarm, topology, p, r):
        """Test if update_gbest_neighborhood gives the expected return values"""
        topo = topology()
        pos, cost = topo.compute_gbest(swarm, p=p, r=r)
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

    @pytest.mark.parametrize("m", [i for i in range(3)])
    @pytest.mark.parametrize("n", [i for i in range(3)])
    def test_delannoy_numbers(self, m, n):
        expected_values = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])
        assert VonNeumann.delannoy(m, n) in expected_values
