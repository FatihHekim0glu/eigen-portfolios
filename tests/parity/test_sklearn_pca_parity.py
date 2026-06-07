"""Parity tests against ``sklearn.decomposition.PCA``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA as SkPCA  # noqa: N811

from eigenportfolios.pca import PCA


def _eigvals_from_sklearn_corr(df: pd.DataFrame) -> np.ndarray:
    """Eigenvalues of the correlation matrix via sklearn singular values."""
    standardised = (df - df.mean()) / df.std(ddof=1)
    sk = SkPCA(n_components=df.shape[1])
    sk.fit(standardised.to_numpy())
    # singular values are sqrt((T-1) * lambda) where lambda is corr eigval
    sing = sk.singular_values_
    t = df.shape[0]
    return np.sort((sing**2) / (t - 1))[::-1]


@pytest.mark.parity
def test_eigvalues_match_sklearn_one_factor(synthetic_one_factor):
    ours = PCA().fit(synthetic_one_factor).eigvals
    sk = _eigvals_from_sklearn_corr(synthetic_one_factor)
    np.testing.assert_allclose(ours, sk, atol=1e-10)


@pytest.mark.parity
def test_eigvalues_match_sklearn_blocks(synthetic_block_correlation):
    ours = PCA().fit(synthetic_block_correlation).eigvals
    sk = _eigvals_from_sklearn_corr(synthetic_block_correlation)
    np.testing.assert_allclose(ours, sk, atol=1e-10)


@pytest.mark.parity
def test_eigvalues_match_sklearn_multi_factor(synthetic_multi_factor):
    ours = PCA().fit(synthetic_multi_factor).eigvals
    sk = _eigvals_from_sklearn_corr(synthetic_multi_factor)
    np.testing.assert_allclose(ours, sk, atol=1e-10)


@pytest.mark.parity
def test_eigvalues_match_sklearn_pure_noise(pure_noise_returns):
    ours = PCA().fit(pure_noise_returns).eigvals
    sk = _eigvals_from_sklearn_corr(pure_noise_returns)
    np.testing.assert_allclose(ours, sk, atol=1e-10)


@pytest.mark.parity
def test_max_deviation_under_threshold(synthetic_multi_factor):
    ours = PCA().fit(synthetic_multi_factor).eigvals
    sk = _eigvals_from_sklearn_corr(synthetic_multi_factor)
    max_dev = float(np.max(np.abs(ours - sk)))
    assert max_dev < 1e-10
