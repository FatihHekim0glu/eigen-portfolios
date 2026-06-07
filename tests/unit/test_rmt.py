"""Unit tests for :mod:`eigenportfolios.rmt`."""

from __future__ import annotations

import numpy as np
import pytest

from eigenportfolios.rmt import (
    MPEdges,
    fit_sigma,
    marchenko_pastur_bulk,
    marchenko_pastur_edges,
    separate_signal_noise,
)


@pytest.mark.unit
def test_edges_returns_dataclass():
    edges = marchenko_pastur_edges(q=0.5, sigma=1.0)
    assert isinstance(edges, MPEdges)


@pytest.mark.unit
def test_edges_q_half_sigma_one():
    edges = marchenko_pastur_edges(q=0.5, sigma=1.0)
    expected_plus = (1.0 + np.sqrt(0.5)) ** 2
    expected_minus = (1.0 - np.sqrt(0.5)) ** 2
    assert edges.lambda_plus == pytest.approx(expected_plus, abs=1e-12)
    assert edges.lambda_minus == pytest.approx(expected_minus, abs=1e-12)


@pytest.mark.unit
def test_edges_q_one_lower_zero():
    edges = marchenko_pastur_edges(q=1.0, sigma=1.0)
    assert edges.lambda_minus == pytest.approx(0.0, abs=1e-12)
    assert edges.lambda_plus == pytest.approx(4.0, abs=1e-12)


@pytest.mark.unit
def test_edges_scales_with_sigma_squared():
    e1 = marchenko_pastur_edges(q=0.25, sigma=1.0)
    e2 = marchenko_pastur_edges(q=0.25, sigma=2.0)
    assert e2.lambda_plus == pytest.approx(4.0 * e1.lambda_plus, abs=1e-12)


@pytest.mark.unit
def test_edges_records_q_and_sigma2():
    edges = marchenko_pastur_edges(q=0.3, sigma=1.5)
    assert edges.q == 0.3
    assert edges.sigma2 == pytest.approx(2.25, abs=1e-12)


@pytest.mark.unit
def test_edges_rejects_zero_q():
    with pytest.raises(ValueError):
        marchenko_pastur_edges(q=0.0)


@pytest.mark.unit
def test_edges_rejects_q_above_one():
    with pytest.raises(ValueError):
        marchenko_pastur_edges(q=1.5)


@pytest.mark.unit
def test_edges_rejects_negative_sigma():
    with pytest.raises(ValueError):
        marchenko_pastur_edges(q=0.5, sigma=-0.1)


@pytest.mark.unit
def test_edges_rejects_nan_q():
    with pytest.raises(ValueError):
        marchenko_pastur_edges(q=float("nan"))


@pytest.mark.unit
def test_bulk_outside_support_zero():
    edges = marchenko_pastur_edges(q=0.5, sigma=1.0)
    lam = np.array([0.0, edges.lambda_minus / 2, edges.lambda_plus * 2])
    density = marchenko_pastur_bulk(lam, q=0.5, sigma=1.0)
    assert density[0] == 0.0
    assert density[1] == 0.0
    assert density[2] == 0.0


@pytest.mark.unit
def test_bulk_density_positive_inside():
    edges = marchenko_pastur_edges(q=0.5, sigma=1.0)
    grid = np.linspace(edges.lambda_minus + 1e-3, edges.lambda_plus - 1e-3, 50)
    density = marchenko_pastur_bulk(grid, q=0.5, sigma=1.0)
    assert (density > 0).all()


@pytest.mark.unit
def test_bulk_integrates_to_one():
    edges = marchenko_pastur_edges(q=0.5, sigma=1.0)
    grid = np.linspace(edges.lambda_minus, edges.lambda_plus, 4000)
    density = marchenko_pastur_bulk(grid, q=0.5, sigma=1.0)
    mass = float(np.trapezoid(density, grid))
    assert mass == pytest.approx(1.0, abs=5e-3)


@pytest.mark.unit
def test_bulk_handles_scalar_input():
    edges = marchenko_pastur_edges(q=0.5, sigma=1.0)
    val = marchenko_pastur_bulk(edges.lambda_minus + 0.1, q=0.5, sigma=1.0)
    assert val.shape == (1,)
    assert val[0] > 0


@pytest.mark.unit
def test_bulk_rejects_bad_q():
    with pytest.raises(ValueError):
        marchenko_pastur_bulk(1.0, q=0.0)


@pytest.mark.unit
def test_bulk_rejects_bad_sigma():
    with pytest.raises(ValueError):
        marchenko_pastur_bulk(1.0, q=0.5, sigma=0.0)


@pytest.mark.unit
def test_fit_sigma_recovers_unit_noise():
    rng = np.random.default_rng(0)
    n, t = 100, 400
    x = rng.standard_normal((n, t))
    corr = (x @ x.T) / t
    eigvals = np.linalg.eigvalsh(corr)
    sigma_hat = fit_sigma(eigvals, q=n / t)
    assert sigma_hat == pytest.approx(1.0, abs=0.1)


@pytest.mark.unit
def test_fit_sigma_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        fit_sigma(np.array([]), q=0.5)


@pytest.mark.unit
def test_fit_sigma_rejects_negative_eigvals():
    with pytest.raises(ValueError, match="non-negative"):
        fit_sigma(np.array([-0.1, 0.5, 1.0]), q=0.5)


@pytest.mark.unit
def test_fit_sigma_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="sigma_bounds"):
        fit_sigma(np.array([0.5, 1.0]), q=0.5, sigma_bounds=(1.0, 0.5))


@pytest.mark.unit
def test_fit_sigma_no_bulk_eigvals():
    with pytest.raises(ValueError, match="MP bulk"):
        fit_sigma(np.array([100.0, 200.0]), q=0.5)


@pytest.mark.unit
def test_separate_with_explicit_sigma():
    eigvals = np.array([10.0, 8.0, 1.5, 1.2, 0.5, 0.3])
    split = separate_signal_noise(eigvals, q=0.5, sigma=1.0)
    assert split.signal_indices.tolist() == [0, 1]
    assert split.noise_indices.tolist() == [2, 3, 4, 5]


@pytest.mark.unit
def test_separate_records_lambda_plus():
    eigvals = np.array([5.0, 0.5])
    split = separate_signal_noise(eigvals, q=0.5, sigma=1.0)
    expected = (1.0 + np.sqrt(0.5)) ** 2
    assert split.lambda_plus == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
def test_separate_with_estimated_sigma():
    rng = np.random.default_rng(0)
    n, t = 80, 300
    x = rng.standard_normal((n, t))
    corr = (x @ x.T) / t
    eigvals = np.linalg.eigvalsh(corr)
    split = separate_signal_noise(eigvals, q=n / t)
    # pure noise: most modes should be inside bulk
    assert split.noise_indices.size >= int(0.8 * n)


@pytest.mark.unit
def test_separate_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        separate_signal_noise(np.array([]), q=0.5, sigma=1.0)


@pytest.mark.unit
def test_separate_rejects_nonpositive_sigma():
    with pytest.raises(ValueError, match="sigma"):
        separate_signal_noise(np.array([1.0, 0.5]), q=0.5, sigma=0.0)


@pytest.mark.unit
def test_separate_q_one_supports_lower_zero():
    eigvals = np.array([5.0, 2.0, 0.1])
    split = separate_signal_noise(eigvals, q=1.0, sigma=1.0)
    assert split.lambda_plus == pytest.approx(4.0, abs=1e-12)
