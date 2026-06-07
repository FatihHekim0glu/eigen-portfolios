# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-07

### Added
- `eigenportfolios.pca` module with deterministic PCA on the correlation matrix,
  returning an immutable `EigenResult` dataclass with eigenvalues sorted in
  descending order, sign-canonicalised eigenvectors, factor returns, and the
  cumulative explained-variance vector.
- `eigenportfolios.rmt` module implementing the Marchenko-Pastur bulk density,
  the closed-form spectral edges, a maximum-likelihood noise-variance estimator
  (`fit_sigma`), and a signal/noise separator that splits eigenmodes by the
  upper Marchenko-Pastur edge.
- `eigenportfolios.interpretation` heuristic that assigns sector labels to
  signal eigenmodes from a user-supplied sector lookup callable.
- `eigenportfolios.universe` builder for a point-in-time S&P 500 ticker
  universe with an optional Polygon hook.
- `eigenportfolios.plots` helpers for Plotly figures (eigenvalue spectrum
  with Marchenko-Pastur overlay, factor-return time series, weight heatmap).
- `eigenportfolios.data` price loader with Polygon and yfinance backends and
  a disk cache.
- `eigenportfolios.cli` Typer command-line entry point exposed as
  `eigenportfolios run`.
- Streamlit demo at `app.py`.
- Test suite covering unit (>= 65 tests), regression (4 tests including
  Plerou 1999 qualitative reproduction), property (Hypothesis invariants),
  and parity (scikit-learn PCA, max abs deviation < 1e-10).
