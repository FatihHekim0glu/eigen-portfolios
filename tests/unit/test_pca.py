"""Unit tests for :mod:`eigenportfolios.pca`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenportfolios.pca import PCA, EigenResult


@pytest.mark.unit
def test_fit_returns_eigenresult(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert isinstance(result, EigenResult)


@pytest.mark.unit
def test_fit_eigvals_descending(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    diffs = np.diff(result.eigvals)
    assert (diffs <= 1e-12).all()


@pytest.mark.unit
def test_fit_eigvals_nonnegative(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert (result.eigvals >= -1e-12).all()


@pytest.mark.unit
def test_fit_eigvals_sum_equals_trace(synthetic_one_factor):
    n = synthetic_one_factor.shape[1]
    result = PCA().fit(synthetic_one_factor)
    assert result.eigvals.sum() == pytest.approx(n, abs=1e-9)


@pytest.mark.unit
def test_fit_eigvecs_orthonormal(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    n = result.eigvecs.shape[0]
    gram = result.eigvecs.T @ result.eigvecs
    np.testing.assert_allclose(gram, np.eye(n), atol=1e-10)


@pytest.mark.unit
def test_fit_eigvecs_shape(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    n = synthetic_one_factor.shape[1]
    assert result.eigvecs.shape == (n, n)


@pytest.mark.unit
def test_fit_factor_returns_shape(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert result.factor_returns.shape == synthetic_one_factor.shape


@pytest.mark.unit
def test_fit_factor_returns_columns(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    n = synthetic_one_factor.shape[1]
    assert list(result.factor_returns.columns) == [f"PC{i + 1}" for i in range(n)]


@pytest.mark.unit
def test_fit_explained_var_monotone(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    diffs = np.diff(result.explained_var)
    assert (diffs >= -1e-12).all()


@pytest.mark.unit
def test_fit_explained_var_ends_at_one(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert result.explained_var[-1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_fit_one_factor_dominates(synthetic_one_factor):
    """Top eigenvalue should explain >50% of variance in a 1-factor model."""
    result = PCA().fit(synthetic_one_factor)
    n = synthetic_one_factor.shape[1]
    assert result.eigvals[0] / n > 0.5


@pytest.mark.unit
def test_fit_three_blocks_three_dominant(synthetic_block_correlation):
    result = PCA().fit(synthetic_block_correlation)
    # the three block factors should sit above the noise floor
    assert (result.eigvals[:3] > 2.0).all()
    assert (result.eigvals[3:6] < 2.0).all()


@pytest.mark.unit
def test_fit_n_features_property(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert result.n_features == synthetic_one_factor.shape[1]


@pytest.mark.unit
def test_fit_n_observations_property(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert result.n_observations == synthetic_one_factor.shape[0]


@pytest.mark.unit
def test_fit_feature_names_recorded(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    assert result.feature_names == tuple(synthetic_one_factor.columns)


@pytest.mark.unit
def test_fit_sign_convention_deterministic(synthetic_block_correlation):
    """The largest-magnitude entry of every column is non-negative."""
    result = PCA().fit(synthetic_block_correlation)
    for j in range(result.eigvecs.shape[1]):
        col = result.eigvecs[:, j]
        idx = int(np.argmax(np.abs(col)))
        assert col[idx] >= 0


@pytest.mark.unit
def test_fit_repeated_run_identical(synthetic_one_factor):
    """Fitting twice on the same data yields bit-identical results."""
    a = PCA().fit(synthetic_one_factor)
    b = PCA().fit(synthetic_one_factor)
    np.testing.assert_array_equal(a.eigvals, b.eigvals)
    np.testing.assert_array_equal(a.eigvecs, b.eigvecs)


@pytest.mark.unit
def test_fit_rejects_non_dataframe():
    with pytest.raises(TypeError):
        PCA().fit(np.zeros((10, 3)))  # type: ignore[arg-type]


@pytest.mark.unit
def test_fit_rejects_single_row():
    df = pd.DataFrame({"A": [0.1], "B": [0.2]})
    with pytest.raises(ValueError, match="at least 2 rows"):
        PCA().fit(df)


@pytest.mark.unit
def test_fit_rejects_zero_columns():
    df = pd.DataFrame(index=[0, 1, 2])
    with pytest.raises(ValueError, match="at least 1 column"):
        PCA().fit(df)


@pytest.mark.unit
def test_fit_rejects_nan():
    df = pd.DataFrame({"A": [0.1, np.nan, 0.3], "B": [0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="NaN"):
        PCA().fit(df)


@pytest.mark.unit
def test_fit_rejects_constant_column():
    df = pd.DataFrame({"A": [0.1, 0.2, 0.3], "C": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="zero-variance"):
        PCA().fit(df)


@pytest.mark.unit
def test_init_rejects_negative_ddof():
    with pytest.raises(ValueError, match="non-negative"):
        PCA(ddof=-1)


@pytest.mark.unit
def test_top_k_returns_new_result(synthetic_block_correlation):
    result = PCA().fit(synthetic_block_correlation)
    truncated = result.top_k(3)
    assert isinstance(truncated, EigenResult)
    assert truncated.eigvals.shape == (3,)
    assert truncated.eigvecs.shape[1] == 3
    assert truncated.factor_returns.shape[1] == 3


@pytest.mark.unit
def test_top_k_rejects_zero(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError):
        result.top_k(0)


@pytest.mark.unit
def test_top_k_rejects_too_large(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError):
        result.top_k(result.n_features + 1)


@pytest.mark.unit
def test_fit_index_preserved(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    pd.testing.assert_index_equal(result.factor_returns.index, synthetic_one_factor.index)


@pytest.mark.unit
def test_fit_ddof_zero_runs():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((50, 5)), columns=list("ABCDE"))
    result = PCA(ddof=0).fit(df)
    assert result.eigvals.shape == (5,)
