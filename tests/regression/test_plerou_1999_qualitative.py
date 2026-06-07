"""Regression: qualitative reproduction of Plerou et al. (1999).

In their analysis of S&P 500 daily returns (~422 stocks, 1962-1996, T~6500)
the bulk of the empirical eigenvalue spectrum is well described by the
Marchenko-Pastur prediction, with only a small number of eigenvalues
deviating from the bulk (interpreted as economically meaningful
factors).  We reproduce this qualitative fact on a synthetic universe
with realistic q = N / T and a small number of injected factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenportfolios.pca import PCA
from eigenportfolios.rmt import separate_signal_noise


@pytest.mark.regression
def test_majority_of_eigvals_inside_bulk():
    """Most empirical eigenvalues fall below the upper MP edge."""
    rng = np.random.default_rng(2025)
    n_assets = 100
    n_periods = 1500
    n_factors = 5
    factors = rng.standard_normal((n_periods, n_factors))
    loadings = rng.uniform(-0.6, 0.6, size=(n_assets, n_factors))
    idio = rng.standard_normal((n_periods, n_assets))
    returns = factors @ loadings.T + idio
    df = pd.DataFrame(
        returns,
        columns=[f"A{i:03d}" for i in range(n_assets)],
    )
    result = PCA().fit(df)
    split = separate_signal_noise(result.eigvals, q=n_assets / n_periods)
    fraction_noise = split.noise_indices.size / n_assets
    assert fraction_noise > 0.75
    # at least one factor should survive as signal
    assert split.signal_indices.size >= 1


@pytest.mark.regression
def test_market_mode_is_largest():
    """When all loadings are positive (market mode), it dominates."""
    rng = np.random.default_rng(1999)
    n_assets = 80
    n_periods = 1200
    market = rng.standard_normal(n_periods)
    loadings = rng.uniform(0.5, 1.5, size=n_assets)
    idio = 0.8 * rng.standard_normal((n_periods, n_assets))
    returns = market[:, None] * loadings[None, :] + idio
    df = pd.DataFrame(returns, columns=[f"A{i:03d}" for i in range(n_assets)])
    result = PCA().fit(df)
    # The top eigenvalue must be a clear outlier
    assert result.eigvals[0] > 3.0 * result.eigvals[1]


@pytest.mark.regression
def test_bulk_centred_near_one():
    """Mean of noise eigenvalues is close to one in correlation units."""
    rng = np.random.default_rng(444)
    n_assets = 80
    n_periods = 2000
    # 2 factors + dense noise
    f = rng.standard_normal((n_periods, 2))
    loadings = rng.uniform(-0.4, 0.4, size=(n_assets, 2))
    idio = rng.standard_normal((n_periods, n_assets))
    returns = f @ loadings.T + idio
    df = pd.DataFrame(returns, columns=[f"A{i:03d}" for i in range(n_assets)])
    result = PCA().fit(df)
    split = separate_signal_noise(result.eigvals, q=n_assets / n_periods)
    noise_eigs = result.eigvals[split.noise_indices]
    assert abs(float(noise_eigs.mean()) - 1.0) < 0.15
