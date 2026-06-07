"""Unit tests for :mod:`eigenportfolios.plots`."""

from __future__ import annotations

import numpy as np
import pytest

from eigenportfolios.pca import PCA
from eigenportfolios.plots import (
    factor_returns_figure,
    spectrum_figure,
    weights_heatmap_figure,
)


@pytest.mark.unit
def test_spectrum_figure_structure(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    fig = spectrum_figure(result, q=0.05)
    assert set(fig) == {"data", "layout"}
    assert len(fig["data"]) == 2
    assert fig["data"][0]["type"] == "histogram"
    assert fig["data"][1]["type"] == "scatter"
    assert "shapes" in fig["layout"]


@pytest.mark.unit
def test_spectrum_figure_rejects_bad_n_points(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError):
        spectrum_figure(result, q=0.05, n_bulk_points=1)


@pytest.mark.unit
def test_factor_returns_figure_structure(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    fig = factor_returns_figure(result, k=3)
    assert len(fig["data"]) == 3
    assert all(tr["type"] == "scatter" for tr in fig["data"])


@pytest.mark.unit
def test_factor_returns_figure_rejects_bad_k(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError):
        factor_returns_figure(result, k=0)


@pytest.mark.unit
def test_factor_returns_figure_rejects_too_large_k(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError):
        factor_returns_figure(result, k=result.eigvecs.shape[1] + 1)


@pytest.mark.unit
def test_weights_heatmap_structure():
    eigvecs = np.eye(5)
    tickers = list("ABCDE")
    fig = weights_heatmap_figure(eigvecs, tickers, k=3)
    assert fig["data"][0]["type"] == "heatmap"
    assert fig["data"][0]["x"] == ["PC1", "PC2", "PC3"]
    assert fig["data"][0]["y"] == tickers


@pytest.mark.unit
def test_weights_heatmap_rejects_mismatched_tickers():
    eigvecs = np.eye(5)
    with pytest.raises(ValueError, match="length"):
        weights_heatmap_figure(eigvecs, ["A", "B", "C"], k=2)


@pytest.mark.unit
def test_weights_heatmap_rejects_bad_k():
    eigvecs = np.eye(5)
    tickers = list("ABCDE")
    with pytest.raises(ValueError):
        weights_heatmap_figure(eigvecs, tickers, k=0)
