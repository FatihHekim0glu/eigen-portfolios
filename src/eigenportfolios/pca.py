"""Principal component analysis of a returns correlation matrix.

The decomposition is performed on the sample correlation matrix
:math:`\\mathbf{C} = \\mathrm{corr}(\\mathbf{R})` of a :math:`T \\times N`
return matrix.  Eigenvalues are returned in descending order, eigenvectors
are sign-canonicalised so that the entry with maximum absolute value is
positive, and the projected factor returns are computed as
:math:`\\mathbf{F} = \\mathbf{R}_{\\text{std}} \\mathbf{V}`, where
:math:`\\mathbf{R}_{\\text{std}}` is the standardised return matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class EigenResult:
    """Immutable result of :class:`PCA.fit`.

    Attributes
    ----------
    eigvals:
        Eigenvalues of the correlation matrix, sorted in descending order.
        Shape ``(N,)``.
    eigvecs:
        Eigenvectors as columns, aligned with ``eigvals`` and
        sign-canonicalised.  Shape ``(N, N)``.
    factor_returns:
        Standardised-return projections onto the eigenvectors, a
        ``DataFrame`` of shape ``(T, N)`` with columns ``PC1`` ... ``PCN``.
    explained_var:
        Cumulative fraction of variance explained.  Shape ``(N,)``.
    feature_names:
        Names of the original features (asset tickers) in input order.
    """

    eigvals: NDArray[np.float64]
    eigvecs: NDArray[np.float64]
    factor_returns: pd.DataFrame
    explained_var: NDArray[np.float64]
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_features(self) -> int:
        """Number of original features (assets)."""
        return int(self.eigvals.shape[0])

    @property
    def n_observations(self) -> int:
        """Number of time-series observations used in the fit."""
        return int(self.factor_returns.shape[0])

    def top_k(self, k: int) -> EigenResult:
        """Return a new :class:`EigenResult` keeping only the top ``k`` modes.

        Parameters
        ----------
        k:
            Number of leading modes to retain.  Must satisfy
            ``1 <= k <= n_features``.
        """
        if k < 1 or k > self.n_features:
            raise ValueError(f"k must be in [1, {self.n_features}], got {k}")
        return EigenResult(
            eigvals=self.eigvals[:k].copy(),
            eigvecs=self.eigvecs[:, :k].copy(),
            factor_returns=self.factor_returns.iloc[:, :k].copy(),
            explained_var=self.explained_var[:k].copy(),
            feature_names=self.feature_names,
        )


class PCA:
    """Correlation-matrix PCA for a returns matrix.

    Parameters
    ----------
    ddof:
        Delta degrees of freedom used to compute the sample standard
        deviation.  Defaults to ``1`` (unbiased estimator) which matches
        ``pandas`` and ``numpy.std(..., ddof=1)``.
    """

    def __init__(self, ddof: int = 1) -> None:
        if ddof < 0:
            raise ValueError(f"ddof must be non-negative, got {ddof}")
        self.ddof = ddof

    def fit(self, returns: pd.DataFrame) -> EigenResult:
        """Fit a correlation-matrix PCA on ``returns``.

        Parameters
        ----------
        returns:
            ``T x N`` matrix of asset returns.  Must contain at least two
            observations and at least one column.  Constant columns
            (zero-variance) are not supported because the correlation
            matrix is undefined.

        Returns
        -------
        EigenResult
            Eigenvalues, eigenvectors, factor returns, and cumulative
            explained variance.
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if returns.shape[0] < 2:
            raise ValueError(f"returns must have at least 2 rows, got {returns.shape[0]}")
        if returns.shape[1] < 1:
            raise ValueError("returns must have at least 1 column")
        if returns.isna().any().any():
            raise ValueError("returns must not contain NaN values")

        std = returns.std(ddof=self.ddof)
        if (std <= 0).any():
            bad = std.index[std <= 0].tolist()
            raise ValueError(f"zero-variance columns not allowed: {bad}")

        standardised = (returns - returns.mean()) / std
        corr = np.corrcoef(standardised.to_numpy(), rowvar=False)

        # symmetrise to absorb floating-point asymmetry from corrcoef
        corr = 0.5 * (corr + corr.T)

        eigvals_raw, eigvecs_raw = np.linalg.eigh(corr)
        # eigh returns ascending; reverse for descending order
        order = np.argsort(eigvals_raw)[::-1]
        eigvals = eigvals_raw[order].astype(np.float64, copy=True)
        eigvecs = eigvecs_raw[:, order].astype(np.float64, copy=True)

        # numerical floor: eigh can produce tiny negative values for PSD
        eigvals = np.where(np.abs(eigvals) < 1e-12, 0.0, eigvals)

        eigvecs = _canonicalise_signs(eigvecs)

        factor_array = standardised.to_numpy() @ eigvecs
        columns = [f"PC{i + 1}" for i in range(eigvecs.shape[1])]
        factor_returns = pd.DataFrame(factor_array, index=returns.index, columns=columns)

        total = float(eigvals.sum())
        explained = np.zeros_like(eigvals) if total <= 0 else np.cumsum(eigvals) / total

        return EigenResult(
            eigvals=eigvals,
            eigvecs=eigvecs,
            factor_returns=factor_returns,
            explained_var=explained.astype(np.float64),
            feature_names=tuple(returns.columns.astype(str)),
        )


def _canonicalise_signs(eigvecs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip each column so the entry with maximum absolute value is positive.

    This produces a deterministic sign convention that matches the behaviour
    used by :func:`sklearn.utils.extmath.svd_flip` (component-wise) and makes
    cross-platform reproducibility straightforward.
    """
    out: NDArray[np.float64] = eigvecs.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = int(np.argmax(np.abs(col)))
        if col[idx] < 0:
            out[:, j] = -col
    return out
