"""Hypothesis-based invariants for the PCA decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from eigenportfolios.pca import PCA


def _random_returns(seed: int, n_assets: int, n_periods: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_periods, n_assets))
    columns = [f"A{i:03d}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=columns)


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_assets=st.integers(min_value=4, max_value=20),
    n_periods=st.integers(min_value=50, max_value=200),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_eigvals_sorted_desc(seed, n_assets, n_periods):
    df = _random_returns(seed=seed, n_assets=n_assets, n_periods=n_periods)
    result = PCA().fit(df)
    diffs = np.diff(result.eigvals)
    assert (diffs <= 1e-10).all()


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_assets=st.integers(min_value=4, max_value=20),
    n_periods=st.integers(min_value=50, max_value=200),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_eigvals_sum_equals_trace(seed, n_assets, n_periods):
    df = _random_returns(seed=seed, n_assets=n_assets, n_periods=n_periods)
    result = PCA().fit(df)
    assert result.eigvals.sum() == pytest.approx(n_assets, abs=1e-8)


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_assets=st.integers(min_value=4, max_value=15),
    n_periods=st.integers(min_value=50, max_value=150),
)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_eigvecs_orthonormal(seed, n_assets, n_periods):
    df = _random_returns(seed=seed, n_assets=n_assets, n_periods=n_periods)
    result = PCA().fit(df)
    gram = result.eigvecs.T @ result.eigvecs
    np.testing.assert_allclose(gram, np.eye(n_assets), atol=1e-9)


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n_assets=st.integers(min_value=4, max_value=12),
    n_periods=st.integers(min_value=50, max_value=200),
)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_eigvals_permutation_invariant(seed, n_assets, n_periods):
    """Permuting the columns must leave the spectrum unchanged."""
    rng = np.random.default_rng(seed)
    df = _random_returns(seed=seed, n_assets=n_assets, n_periods=n_periods)
    perm = rng.permutation(n_assets)
    permuted = df.iloc[:, perm]
    a = PCA().fit(df)
    b = PCA().fit(permuted)
    np.testing.assert_allclose(a.eigvals, b.eigvals, atol=1e-9)
