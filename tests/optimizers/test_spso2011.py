# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyswarms.single import SPSO2011
from pyswarms.utils.functions import single_obj as fx


@pytest.fixture
def spso_options():
    return {
        "w": 1 / (2 * np.log(2)),
        "c1": 0.5 + np.log(2),
        "c2": 0.5 + np.log(2),
        "k": 3,
    }


@pytest.fixture
def bounds():
    dimensions = 2
    return -10 * np.ones(dimensions), 10 * np.ones(dimensions)


def test_spso2011_init(spso_options, bounds):
    optimizer = SPSO2011(
        n_particles=40,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )

    assert optimizer.n_particles == 40
    assert optimizer.dimensions == 2
    assert optimizer.options == spso_options
    assert optimizer.seed == 42


@pytest.mark.parametrize("missing_option", ["w", "c1", "c2", "k"])
def test_spso2011_missing_option_raises(spso_options, missing_option):
    options = spso_options.copy()
    del options[missing_option]

    with pytest.raises(KeyError):
        SPSO2011(
            n_particles=40,
            dimensions=2,
            options=options,
            seed=42,
        )


def test_spso2011_nonpositive_k_raises(spso_options):
    options = spso_options.copy()
    options["k"] = 0

    with pytest.raises(ValueError):
        SPSO2011(
            n_particles=40,
            dimensions=2,
            options=options,
            seed=42,
        )


def test_generate_informants_shape_and_self_informants(spso_options, bounds):
    optimizer = SPSO2011(
        n_particles=40,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )

    optimizer._generate_informants()

    assert optimizer.informants.shape == (40, 40)
    assert optimizer.informants.dtype == bool
    assert np.all(np.diag(optimizer.informants))


def test_hypersphere_samples_inside_radius(spso_options, bounds):
    optimizer = SPSO2011(
        n_particles=40,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )

    centers = np.zeros((40, 2))
    radii = np.ones(40)

    samples = optimizer._sample_hypersphere(centers, radii)
    distances = np.linalg.norm(samples - centers, axis=1)

    assert samples.shape == centers.shape
    assert np.all(distances <= radii + 1e-12)


def test_absorbing_bounds_clip_position_and_zero_velocity(spso_options, bounds):
    optimizer = SPSO2011(
        n_particles=3,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )

    optimizer.swarm.position = np.array(
        [
            [11.0, 0.5],
            [-0.5, -12.0],
            [0.2, 0.3],
        ]
    )
    optimizer.swarm.velocity = np.array(
        [
            [0.7, 0.8],
            [0.9, -0.4],
            [0.1, 0.2],
        ]
    )

    optimizer._apply_absorbing_bounds()

    assert np.allclose(
        optimizer.swarm.position,
        np.array(
            [
                [10.0, 0.5],
                [-0.5, -10.0],
                [0.2, 0.3],
            ]
        ),
    )
    assert np.allclose(
        optimizer.swarm.velocity,
        np.array(
            [
                [0.0, 0.8],
                [0.9, 0.0],
                [0.1, 0.2],
            ]
        ),
    )


def test_spso2011_optimizes_sphere(spso_options, bounds):
    optimizer = SPSO2011(
        n_particles=40,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )

    cost, pos = optimizer.optimize(fx.sphere, iters=100, verbose=False)

    assert np.isfinite(cost)
    assert pos.shape == (2,)
    assert cost < 1.0


def test_spso2011_seed_reproducibility(spso_options, bounds):
    optimizer_1 = SPSO2011(
        n_particles=40,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )
    cost_1, pos_1 = optimizer_1.optimize(fx.sphere, iters=100, verbose=False)

    optimizer_2 = SPSO2011(
        n_particles=40,
        dimensions=2,
        options=spso_options,
        bounds=bounds,
        seed=42,
    )
    cost_2, pos_2 = optimizer_2.optimize(fx.sphere, iters=100, verbose=False)

    assert np.allclose(cost_1, cost_2)
    assert np.allclose(pos_1, pos_2)
