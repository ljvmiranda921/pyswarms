#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import standard library
# Import standard libraries
import re
from typing import TYPE_CHECKING, Any, Dict, Tuple, Type

# Import modules
import pytest

# Import from pyswarms
from pyswarms.backend.topology import Star
from pyswarms.base.single import SwarmOptimizer
from pyswarms.single import GeneralOptimizerPSO, GlobalBestPSO, LocalBestPSO
from pyswarms.utils.functions import single_obj as fx

# Instantiate optimizers
optimizers = [GlobalBestPSO, LocalBestPSO, GeneralOptimizerPSO]
options = {"c1": 2, "c2": 2, "w": 0.7, "k": 3, "p": 2}
parameters = dict(
    n_particles=20,
    dimensions=10,
    options=options,
)


if TYPE_CHECKING:
    class FixtureRequest:
        param: Type[SwarmOptimizer]
else:
    FixtureRequest = Any


class TestToleranceOptions:
    @pytest.fixture(params=optimizers)
    def optimizer(self, request: FixtureRequest):
        global parameters
        if request.param.__name__ == "GeneralOptimizerPSO":
            return request.param, {**parameters, **{"topology": Star()}}
        return request.param, parameters

    def test_verbose(self, optimizer: Tuple[Type[SwarmOptimizer], Dict[str, Any]], capsys: pytest.CaptureFixture[str]):
        """Test verbose run"""
        optm, params = optimizer
        opt = optm(**params)
        opt.optimize(fx.sphere, iters=100)
        out = capsys.readouterr().err
        count = len(re.findall(r"pyswarms", out))
        assert count > 0

    def test_silent(self, optimizer: Tuple[Type[SwarmOptimizer], Dict[str, Any]], capsys: pytest.CaptureFixture[str]):
        """Test silent run"""
        optm, params = optimizer
        opt = optm(**params)
        opt.optimize(fx.sphere, iters=100, verbose=False)
        out = capsys.readouterr()
        assert out.err == ""
        assert out.out == ""
