"""Marchenko-Pastur random matrix theory utilities.

For an :math:`N \\times T` matrix of i.i.d. entries with mean zero and
variance :math:`\\sigma^2`, the empirical spectral distribution of
:math:`\\mathbf{C} = \\frac{1}{T} \\mathbf{X} \\mathbf{X}^{\\top}` converges
in the limit :math:`N, T \\to \\infty` with ``q = N / T -> q`` fixed and
``q in (0, 1]`` to the Marchenko-Pastur law with density

.. math::

   f(\\lambda) =
   \\frac{1}{2 \\pi \\sigma^{2}}
   \\frac{\\sqrt{(\\lambda_{+} - \\lambda)(\\lambda - \\lambda_{-})}}
        {q \\lambda}
   \\mathbf{1}_{[\\lambda_{-},\\,\\lambda_{+}]}(\\lambda),

with edges
:math:`\\lambda_{\\pm} = \\sigma^{2} (1 \\pm \\sqrt{q})^{2}`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class MPEdges:
    """Closed-form Marchenko-Pastur spectral edges."""

    lambda_minus: float
    lambda_plus: float
    q: float
    sigma2: float


@dataclass(frozen=True)
class SignalNoise:
    """Result of splitting the empirical spectrum by the upper MP edge."""

    signal_indices: NDArray[np.intp]
    noise_indices: NDArray[np.intp]
    lambda_plus: float
    sigma2: float


def marchenko_pastur_edges(q: float, sigma: float = 1.0) -> MPEdges:
    """Return the closed-form upper and lower Marchenko-Pastur edges.

    Parameters
    ----------
    q:
        Aspect ratio :math:`q = N / T`.  Must satisfy ``0 < q <= 1`` so the
        spectrum is supported on :math:`[\\lambda_{-},\\lambda_{+}]` without
        a Dirac mass at zero.
    sigma:
        Standard deviation of the underlying i.i.d. entries.  Defaults
        to ``1``.
    """
    _validate_q(q)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    sigma2 = float(sigma) ** 2
    sqrt_q = np.sqrt(q)
    lam_minus = sigma2 * (1.0 - sqrt_q) ** 2
    lam_plus = sigma2 * (1.0 + sqrt_q) ** 2
    return MPEdges(
        lambda_minus=float(lam_minus),
        lambda_plus=float(lam_plus),
        q=float(q),
        sigma2=float(sigma2),
    )


def marchenko_pastur_bulk(
    lam: NDArray[np.float64] | float,
    q: float,
    sigma: float = 1.0,
) -> NDArray[np.float64]:
    """Marchenko-Pastur density evaluated at ``lam``.

    Values outside ``[lambda_-, lambda_+]`` are returned as zero.

    Parameters
    ----------
    lam:
        Eigenvalue(s) at which to evaluate the density.
    q:
        Aspect ratio :math:`q = N / T`, ``0 < q <= 1``.
    sigma:
        Standard deviation of the underlying i.i.d. entries.
    """
    _validate_q(q)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    lam_arr = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    edges = marchenko_pastur_edges(q=q, sigma=sigma)
    lam_minus, lam_plus = edges.lambda_minus, edges.lambda_plus
    sigma2 = edges.sigma2

    density = np.zeros_like(lam_arr)
    inside = (lam_arr >= lam_minus) & (lam_arr <= lam_plus) & (lam_arr > 0.0)
    if not inside.any():
        return density
    lam_in = lam_arr[inside]
    numerator = np.sqrt((lam_plus - lam_in) * (lam_in - lam_minus))
    density[inside] = numerator / (2.0 * np.pi * sigma2 * q * lam_in)
    return density


def fit_sigma(
    eigvals: NDArray[np.float64],
    q: float,
    sigma_bounds: tuple[float, float] = (0.05, 1.5),
) -> float:
    """Estimate the noise standard deviation by least-squares spectrum fit.

    The objective is the L2 distance between the empirical eigenvalue
    histogram (restricted to the bulk region) and the Marchenko-Pastur
    density.  Only eigenvalues that *could* lie inside an MP bulk for
    some ``sigma`` in ``sigma_bounds`` are used in the fit; signal
    outliers (large eigenvalues from genuine factor structure) do not
    contaminate the estimate.

    Parameters
    ----------
    eigvals:
        Vector of empirical eigenvalues.
    q:
        Aspect ratio :math:`q = N / T`, ``0 < q <= 1``.
    sigma_bounds:
        ``(low, high)`` bounds on the sigma search range.
    """
    _validate_q(q)
    if not (sigma_bounds[0] > 0 and sigma_bounds[1] > sigma_bounds[0]):
        raise ValueError(f"invalid sigma_bounds: {sigma_bounds}")
    eig_arr = np.asarray(eigvals, dtype=np.float64).ravel()
    if eig_arr.size == 0:
        raise ValueError("eigvals must not be empty")
    if (eig_arr < 0).any():
        raise ValueError("eigvals must be non-negative")

    # Restrict to the candidate bulk: eigenvalues that lie inside the MP
    # support for the maximum sigma considered.
    max_edge = marchenko_pastur_edges(q=q, sigma=sigma_bounds[1]).lambda_plus
    bulk_mask = eig_arr <= max_edge
    if not bulk_mask.any():
        raise ValueError("no eigenvalues fall inside the MP bulk for the given sigma_bounds")
    bulk_eig = eig_arr[bulk_mask]

    def objective(sigma: float) -> float:
        edges = marchenko_pastur_edges(q=q, sigma=sigma)
        # Build a fine grid in the support and integrate squared error.
        grid = np.linspace(edges.lambda_minus, edges.lambda_plus, 200)
        density_theory = marchenko_pastur_bulk(grid, q=q, sigma=sigma)
        # Empirical density via Gaussian KDE approximation on the bulk.
        bw = 0.05 * max(edges.lambda_plus - edges.lambda_minus, 1e-3)
        diffs = (grid[:, None] - bulk_eig[None, :]) / bw
        kernel = np.exp(-0.5 * diffs**2) / (bw * np.sqrt(2.0 * np.pi))
        density_emp = kernel.mean(axis=1)
        return float(np.trapezoid((density_theory - density_emp) ** 2, grid))

    result = minimize_scalar(
        objective,
        bounds=sigma_bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


def separate_signal_noise(
    eigvals: NDArray[np.float64],
    q: float,
    sigma: float | None = None,
) -> SignalNoise:
    """Split eigenvalues into signal and noise modes.

    Eigenvalues strictly greater than the upper MP edge are labelled as
    signal; the remainder are labelled as noise.

    Parameters
    ----------
    eigvals:
        Vector of empirical eigenvalues (any order; original indices are
        preserved in the returned arrays).
    q:
        Aspect ratio :math:`q = N / T`.
    sigma:
        Noise standard deviation.  When ``None`` the value is estimated
        via :func:`fit_sigma`.
    """
    _validate_q(q)
    eig_arr = np.asarray(eigvals, dtype=np.float64).ravel()
    if eig_arr.size == 0:
        raise ValueError("eigvals must not be empty")
    sigma_used = fit_sigma(eig_arr, q=q) if sigma is None else float(sigma)
    if sigma_used <= 0:
        raise ValueError(f"sigma must be positive, got {sigma_used}")
    edges = marchenko_pastur_edges(q=q, sigma=sigma_used)
    signal_mask = eig_arr > edges.lambda_plus
    signal_idx = np.flatnonzero(signal_mask).astype(np.intp)
    noise_idx = np.flatnonzero(~signal_mask).astype(np.intp)
    return SignalNoise(
        signal_indices=signal_idx,
        noise_indices=noise_idx,
        lambda_plus=edges.lambda_plus,
        sigma2=edges.sigma2,
    )


def _validate_q(q: float) -> None:
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    if q <= 0 or q > 1:
        raise ValueError(f"q must lie in (0, 1], got {q}")
