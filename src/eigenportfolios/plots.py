"""Plotly figure builders for the eigen-portfolios pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from eigenportfolios.rmt import marchenko_pastur_bulk, marchenko_pastur_edges

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from eigenportfolios.pca import EigenResult


def spectrum_figure(
    result: EigenResult,
    q: float,
    sigma: float = 1.0,
    n_bulk_points: int = 200,
) -> dict[str, Any]:
    """Build the eigenvalue-spectrum figure with the MP overlay.

    Returns a Plotly figure as a plain ``dict`` so the module remains
    importable without Plotly being installed.
    """
    if n_bulk_points < 2:
        raise ValueError(f"n_bulk_points must be >= 2, got {n_bulk_points}")
    edges = marchenko_pastur_edges(q=q, sigma=sigma)
    grid = np.linspace(edges.lambda_minus, edges.lambda_plus, n_bulk_points)
    density = marchenko_pastur_bulk(grid, q=q, sigma=sigma)
    eig = result.eigvals

    histogram_trace = {
        "type": "histogram",
        "x": eig.tolist(),
        "name": "empirical eigenvalues",
        "histnorm": "probability density",
        "opacity": 0.65,
    }
    mp_trace = {
        "type": "scatter",
        "x": grid.tolist(),
        "y": density.tolist(),
        "mode": "lines",
        "name": "Marchenko-Pastur",
    }
    layout = {
        "title": "Empirical spectrum vs Marchenko-Pastur bulk",
        "xaxis": {"title": "eigenvalue"},
        "yaxis": {"title": "density"},
        "shapes": [
            {
                "type": "line",
                "x0": edges.lambda_plus,
                "x1": edges.lambda_plus,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"dash": "dash"},
            }
        ],
    }
    return {"data": [histogram_trace, mp_trace], "layout": layout}


def factor_returns_figure(result: EigenResult, k: int = 3) -> dict[str, Any]:
    """Build a time-series figure of the leading ``k`` factor returns."""
    if k < 1 or k > result.eigvecs.shape[1]:
        raise ValueError(f"k must be in [1, {result.eigvecs.shape[1]}], got {k}")
    fr = result.factor_returns.iloc[:, :k]
    traces = [
        {
            "type": "scatter",
            "x": fr.index.astype(str).tolist(),
            "y": fr[col].to_numpy().tolist(),
            "mode": "lines",
            "name": col,
        }
        for col in fr.columns
    ]
    layout = {
        "title": f"Leading {k} factor returns",
        "xaxis": {"title": "date"},
        "yaxis": {"title": "factor return"},
    }
    return {"data": traces, "layout": layout}


def weights_heatmap_figure(
    eigvecs: NDArray[np.float64],
    tickers: list[str] | tuple[str, ...],
    k: int = 5,
) -> dict[str, Any]:
    """Build a heatmap of the loadings of the leading ``k`` eigenvectors."""
    if k < 1 or k > eigvecs.shape[1]:
        raise ValueError(f"k must be in [1, {eigvecs.shape[1]}], got {k}")
    if len(tickers) != eigvecs.shape[0]:
        raise ValueError(f"tickers length {len(tickers)} != eigvec rows {eigvecs.shape[0]}")
    sub = eigvecs[:, :k]
    trace = {
        "type": "heatmap",
        "z": sub.tolist(),
        "x": [f"PC{i + 1}" for i in range(k)],
        "y": list(tickers),
        "colorscale": "RdBu",
        "zmid": 0,
    }
    layout = {
        "title": "Eigenvector loadings",
        "xaxis": {"title": "factor"},
        "yaxis": {"title": "asset"},
    }
    return {"data": [trace], "layout": layout}
