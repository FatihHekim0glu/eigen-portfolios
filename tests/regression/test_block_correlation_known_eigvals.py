"""Regression: a block-diagonal correlation matrix has K large eigenvalues."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenportfolios.pca import PCA


def _synthesise(n_blocks: int, n_per_block: int, n_periods: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    n_assets = n_blocks * n_per_block
    factors = rng.standard_normal((n_periods, n_blocks))
    loadings = np.zeros((n_assets, n_blocks))
    for b in range(n_blocks):
        loadings[b * n_per_block : (b + 1) * n_per_block, b] = 1.0
    idio = noise * rng.standard_normal((n_periods, n_assets))
    returns = factors @ loadings.T + idio
    return pd.DataFrame(returns, columns=[f"A{i:03d}" for i in range(n_assets)])


@pytest.mark.regression
@pytest.mark.parametrize("n_blocks", [2, 3, 4])
def test_block_count_recovered(n_blocks):
    df = _synthesise(n_blocks=n_blocks, n_per_block=12, n_periods=1500, noise=0.3, seed=0)
    result = PCA().fit(df)
    # The top n_blocks eigenvalues should be >> the rest
    top = result.eigvals[:n_blocks]
    tail = result.eigvals[n_blocks:]
    assert top.min() > 3.0 * tail.max()


@pytest.mark.regression
def test_block_eigval_magnitudes_known():
    """For K blocks of size m with loadings of 1 and idio variance s^2,
    the dominant eigenvalues of corr(R) scale like m / (1 + s^2)."""
    n_per_block = 10
    n_blocks = 3
    df = _synthesise(
        n_blocks=n_blocks,
        n_per_block=n_per_block,
        n_periods=4000,
        noise=0.3,
        seed=42,
    )
    result = PCA().fit(df)
    # Each block factor explains roughly m * (1 / (1 + 0.09)) ~ 9.17 in correlation units
    for k in range(n_blocks):
        assert result.eigvals[k] > 7.0
        assert result.eigvals[k] < 11.0


@pytest.mark.regression
def test_noise_floor_close_to_one():
    """The non-signal eigenvalues should cluster near 1 in a noise-only setup."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        rng.standard_normal((2000, 30)),
        columns=[f"X{i}" for i in range(30)],
    )
    result = PCA().fit(df)
    # bulk eigenvalues should have mean ~ 1
    mean_bulk = result.eigvals.mean()
    assert abs(mean_bulk - 1.0) < 0.05
