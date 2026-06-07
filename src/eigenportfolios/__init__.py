"""Eigen-portfolios: PCA + Marchenko-Pastur RMT for financial returns."""

from __future__ import annotations

from eigenportfolios.interpretation import label_factors
from eigenportfolios.pca import PCA, EigenResult
from eigenportfolios.rmt import (
    fit_sigma,
    marchenko_pastur_bulk,
    marchenko_pastur_edges,
    separate_signal_noise,
)

__all__ = [
    "PCA",
    "EigenResult",
    "fit_sigma",
    "label_factors",
    "marchenko_pastur_bulk",
    "marchenko_pastur_edges",
    "separate_signal_noise",
]

__version__ = "0.1.0"
