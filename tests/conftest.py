"""Shared fixtures for the eigen-portfolios test-suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic default RNG."""
    return np.random.default_rng(0xE16E)


@pytest.fixture
def synthetic_one_factor() -> pd.DataFrame:
    """A T x N return matrix with a single dominant common factor.

    Each asset return is a positive loading on a common factor plus
    idiosyncratic Gaussian noise with variance 0.5.  This is the
    canonical setup in which PCA recovers a market mode.
    """
    rng = np.random.default_rng(42)
    n_assets = 30
    n_periods = 600
    factor = rng.standard_normal(n_periods)
    loadings = rng.uniform(0.6, 1.2, size=n_assets)
    idio = 0.5 * rng.standard_normal((n_periods, n_assets))
    returns = factor[:, None] * loadings[None, :] + idio
    columns = [f"A{i:02d}" for i in range(n_assets)]
    index = pd.RangeIndex(n_periods, name="t")
    return pd.DataFrame(returns, index=index, columns=columns)


@pytest.fixture
def synthetic_block_correlation() -> pd.DataFrame:
    """A return matrix with three orthogonal block factors.

    Three disjoint blocks of 10 assets each load on three independent
    common factors.  PCA should recover three dominant eigenvalues.
    """
    rng = np.random.default_rng(7)
    n_per_block = 10
    n_blocks = 3
    n_assets = n_per_block * n_blocks
    n_periods = 800
    factors = rng.standard_normal((n_periods, n_blocks))
    idio = 0.4 * rng.standard_normal((n_periods, n_assets))
    loadings = np.zeros((n_assets, n_blocks))
    for b in range(n_blocks):
        loadings[b * n_per_block : (b + 1) * n_per_block, b] = rng.uniform(
            0.8, 1.2, size=n_per_block
        )
    returns = factors @ loadings.T + idio
    columns = [f"B{b}_A{i:02d}" for b in range(n_blocks) for i in range(n_per_block)]
    index = pd.RangeIndex(n_periods, name="t")
    return pd.DataFrame(returns, index=index, columns=columns)


@pytest.fixture
def synthetic_multi_factor() -> pd.DataFrame:
    """A return matrix with three dense common factors plus noise."""
    rng = np.random.default_rng(2024)
    n_assets = 40
    n_periods = 1000
    n_factors = 3
    factors = rng.standard_normal((n_periods, n_factors))
    loadings = rng.uniform(-1.0, 1.0, size=(n_assets, n_factors))
    idio = 0.5 * rng.standard_normal((n_periods, n_assets))
    returns = factors @ loadings.T + idio
    columns = [f"M{i:02d}" for i in range(n_assets)]
    index = pd.RangeIndex(n_periods, name="t")
    return pd.DataFrame(returns, index=index, columns=columns)


@pytest.fixture
def pure_noise_returns() -> pd.DataFrame:
    """A T x N matrix of i.i.d. unit-variance Gaussian noise."""
    rng = np.random.default_rng(123)
    n_assets = 50
    n_periods = 300
    data = rng.standard_normal((n_periods, n_assets))
    columns = [f"N{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(data, index=pd.RangeIndex(n_periods, name="t"), columns=columns)
