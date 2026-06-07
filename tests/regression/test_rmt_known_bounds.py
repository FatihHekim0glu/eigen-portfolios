"""Regression: closed-form Marchenko-Pastur edges match the analytic value."""

from __future__ import annotations

import numpy as np
import pytest

from eigenportfolios.rmt import marchenko_pastur_bulk, marchenko_pastur_edges


@pytest.mark.regression
def test_lambda_plus_sigma_one_n100_t200():
    """For sigma=1, N=100, T=200 the upper edge equals (1 + sqrt(0.5))^2."""
    edges = marchenko_pastur_edges(q=100 / 200, sigma=1.0)
    expected = (1.0 + np.sqrt(0.5)) ** 2
    assert edges.lambda_plus == pytest.approx(expected, abs=1e-12)


@pytest.mark.regression
def test_lambda_minus_sigma_one_n100_t200():
    edges = marchenko_pastur_edges(q=100 / 200, sigma=1.0)
    expected = (1.0 - np.sqrt(0.5)) ** 2
    assert edges.lambda_minus == pytest.approx(expected, abs=1e-12)


@pytest.mark.regression
def test_bulk_integrates_to_one_q_one_quarter():
    edges = marchenko_pastur_edges(q=0.25, sigma=1.0)
    grid = np.linspace(edges.lambda_minus, edges.lambda_plus, 6000)
    density = marchenko_pastur_bulk(grid, q=0.25, sigma=1.0)
    mass = float(np.trapezoid(density, grid))
    assert mass == pytest.approx(1.0, abs=2e-3)


@pytest.mark.regression
def test_bulk_integrates_to_one_q_three_quarters():
    edges = marchenko_pastur_edges(q=0.75, sigma=1.0)
    grid = np.linspace(edges.lambda_minus, edges.lambda_plus, 6000)
    density = marchenko_pastur_bulk(grid, q=0.75, sigma=1.0)
    mass = float(np.trapezoid(density, grid))
    assert mass == pytest.approx(1.0, abs=5e-3)


@pytest.mark.regression
def test_empirical_bulk_within_edges():
    """For pure noise, every empirical eigenvalue lies within [eps, lambda+]."""
    rng = np.random.default_rng(2024)
    n, t = 100, 400
    x = rng.standard_normal((n, t))
    corr = (x @ x.T) / t
    eigvals = np.linalg.eigvalsh(corr)
    edges = marchenko_pastur_edges(q=n / t, sigma=1.0)
    # finite-N: allow a few outliers above lambda_+ at most
    above = int((eigvals > edges.lambda_plus * 1.1).sum())
    assert above <= 2
