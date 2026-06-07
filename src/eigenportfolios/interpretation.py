"""Factor-interpretation heuristic for signal eigenmodes.

Each signal eigenvector is mapped to a human-readable label by examining
the sector breakdown of its largest-magnitude entries.  If a single
sector dominates the top-weight basket the factor is labelled
``"<Sector>"``; otherwise it is labelled ``"Mixed"`` (or ``"Market"`` for
the leading mode whose loadings are uniformly same-signed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from eigenportfolios.pca import EigenResult


def label_factors(
    result: EigenResult,
    sector_lookup: Callable[[str], str],
    *,
    top_k_assets: int = 10,
    sector_dominance: float = 0.5,
    market_uniformity: float = 0.85,
    n_factors: int | None = None,
) -> list[str]:
    """Assign sector labels to the leading eigenmodes of ``result``.

    Parameters
    ----------
    result:
        Output of :class:`eigenportfolios.pca.PCA.fit`.
    sector_lookup:
        Callable mapping a ticker to a sector string.  Missing tickers
        should return ``"Unknown"``.
    top_k_assets:
        Number of largest-magnitude loadings inspected per factor.
    sector_dominance:
        Threshold for the share of top-weight assets that must belong to
        the same sector for the factor to be labelled with that sector.
    market_uniformity:
        Threshold on the fraction of same-signed loadings used to detect
        the market mode.
    n_factors:
        Number of leading factors to label.  Defaults to all factors.
    """
    if top_k_assets < 1:
        raise ValueError(f"top_k_assets must be >= 1, got {top_k_assets}")
    if not 0.0 < sector_dominance <= 1.0:
        raise ValueError(f"sector_dominance must be in (0, 1], got {sector_dominance}")
    if not 0.5 <= market_uniformity <= 1.0:
        raise ValueError(f"market_uniformity must be in [0.5, 1], got {market_uniformity}")

    n_total = result.eigvecs.shape[1]
    n_use = n_total if n_factors is None else min(n_factors, n_total)
    labels: list[str] = []
    tickers = result.feature_names or tuple(f"X{i + 1}" for i in range(result.n_features))

    for j in range(n_use):
        vec = result.eigvecs[:, j]
        if _is_market_mode(vec, market_uniformity):
            labels.append("Market")
            continue
        labels.append(
            _sector_label(
                vec=vec,
                tickers=tickers,
                sector_lookup=sector_lookup,
                top_k_assets=min(top_k_assets, vec.shape[0]),
                sector_dominance=sector_dominance,
            )
        )
    return labels


def _is_market_mode(vec: NDArray[np.float64], uniformity: float) -> bool:
    """Return ``True`` when the eigenvector points consistently in one sign."""
    if vec.size == 0:
        return False
    same_sign = max(float((vec >= 0).mean()), float((vec <= 0).mean()))
    return same_sign >= uniformity


def _sector_label(
    vec: NDArray[np.float64],
    tickers: tuple[str, ...],
    sector_lookup: Callable[[str], str],
    top_k_assets: int,
    sector_dominance: float,
) -> str:
    """Find the dominant sector among the top-weight assets."""
    idx = np.argsort(np.abs(vec))[::-1][:top_k_assets]
    sectors = [sector_lookup(tickers[i]) for i in idx]
    counts: dict[str, int] = {}
    for sec in sectors:
        counts[sec] = counts.get(sec, 0) + 1
    if not counts:
        return "Mixed"
    best_sector, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_sector == "Unknown":
        # If "Unknown" dominates, fall back to the next-best non-Unknown.
        filtered = [(s, c) for s, c in counts.items() if s != "Unknown"]
        if not filtered:
            return "Unknown"
        best_sector, best_count = max(filtered, key=lambda kv: kv[1])
    share = best_count / top_k_assets
    if share >= sector_dominance:
        return best_sector
    return "Mixed"
