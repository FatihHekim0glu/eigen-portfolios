"""Regression: PCA eigenvectors are exactly orthonormal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenportfolios.pca import PCA


@pytest.mark.regression
@pytest.mark.parametrize("n, t, seed", [(20, 300, 0), (50, 500, 1), (100, 1000, 2)])
def test_eigvecs_orthonormal(n, t, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.standard_normal((t, n)),
        columns=[f"A{i:03d}" for i in range(n)],
    )
    result = PCA().fit(df)
    gram = result.eigvecs.T @ result.eigvecs
    np.testing.assert_allclose(gram, np.eye(n), atol=1e-9)


@pytest.mark.regression
def test_eigvecs_norm_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((400, 25)), columns=[f"X{i}" for i in range(25)])
    result = PCA().fit(df)
    norms = np.linalg.norm(result.eigvecs, axis=0)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


@pytest.mark.regression
def test_eigvecs_pairwise_orthogonal():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.standard_normal((500, 15)), columns=[f"Y{i}" for i in range(15)])
    result = PCA().fit(df)
    n = result.eigvecs.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            dot = float(result.eigvecs[:, i] @ result.eigvecs[:, j])
            assert abs(dot) < 1e-9
