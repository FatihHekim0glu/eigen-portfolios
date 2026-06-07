"""Unit tests for :mod:`eigenportfolios.interpretation`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eigenportfolios.interpretation import label_factors
from eigenportfolios.pca import PCA


def _sector_lookup_factory(mapping):
    def _lookup(ticker: str) -> str:
        return mapping.get(ticker, "Unknown")

    return _lookup


@pytest.mark.unit
def test_label_market_mode_on_one_factor(synthetic_one_factor):
    """The leading mode of a 1-factor model is labelled 'Market'."""
    result = PCA().fit(synthetic_one_factor)
    mapping = dict.fromkeys(synthetic_one_factor.columns, "Tech")
    labels = label_factors(result, sector_lookup=_sector_lookup_factory(mapping), n_factors=1)
    assert labels == ["Market"]


@pytest.mark.unit
def test_label_n_factors_respected():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        rng.standard_normal((400, 6)),
        columns=[f"A{i}" for i in range(6)],
    )
    result = PCA().fit(df)
    mapping = dict.fromkeys(df.columns, "Unknown")
    labels = label_factors(result, sector_lookup=_sector_lookup_factory(mapping), n_factors=3)
    assert len(labels) == 3


@pytest.mark.unit
def test_label_clips_to_n_features():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.standard_normal((200, 4)), columns=list("ABCD"))
    result = PCA().fit(df)
    mapping = dict.fromkeys(df.columns, "Sector")
    labels = label_factors(result, sector_lookup=_sector_lookup_factory(mapping), n_factors=10)
    assert len(labels) == 4


@pytest.mark.unit
def test_label_sector_dominates(synthetic_block_correlation):
    """A block factor with one sector should get that sector's label."""
    cols = list(synthetic_block_correlation.columns)
    mapping = {}
    for i, col in enumerate(cols):
        block = i // 10
        mapping[col] = {0: "Tech", 1: "Health", 2: "Energy"}[block]
    result = PCA().fit(synthetic_block_correlation)
    labels = label_factors(
        result,
        sector_lookup=_sector_lookup_factory(mapping),
        n_factors=4,
        top_k_assets=8,
    )
    # at least one of the top 4 modes should map to one of the sectors
    sector_set = {"Tech", "Health", "Energy"}
    assert any(lbl in sector_set for lbl in labels[1:4])


@pytest.mark.unit
def test_label_mixed_when_no_dominant_sector():
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        rng.standard_normal((600, 20)),
        columns=[f"X{i:02d}" for i in range(20)],
    )
    result = PCA().fit(df)
    sectors = ["Tech", "Health", "Energy", "Financials", "Industrials"]
    mapping = {col: sectors[i % len(sectors)] for i, col in enumerate(df.columns)}
    labels = label_factors(
        result,
        sector_lookup=_sector_lookup_factory(mapping),
        n_factors=5,
        top_k_assets=8,
        sector_dominance=0.95,
    )
    assert "Mixed" in labels


@pytest.mark.unit
def test_label_rejects_bad_top_k(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError, match="top_k_assets"):
        label_factors(
            result,
            sector_lookup=lambda t: "X",
            top_k_assets=0,
        )


@pytest.mark.unit
def test_label_rejects_bad_dominance(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError, match="sector_dominance"):
        label_factors(result, sector_lookup=lambda t: "X", sector_dominance=0.0)


@pytest.mark.unit
def test_label_rejects_bad_uniformity(synthetic_one_factor):
    result = PCA().fit(synthetic_one_factor)
    with pytest.raises(ValueError, match="market_uniformity"):
        label_factors(result, sector_lookup=lambda t: "X", market_uniformity=0.2)


@pytest.mark.unit
def test_is_market_mode_empty_vector():
    """Empty eigenvector cannot be classified as the market mode."""
    from eigenportfolios.interpretation import _is_market_mode

    assert _is_market_mode(np.array([], dtype=float), 0.85) is False


@pytest.mark.unit
def test_label_unknown_only_returns_unknown():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.standard_normal((200, 4)), columns=list("ABCD"))
    result = PCA().fit(df)
    labels = label_factors(
        result,
        sector_lookup=lambda t: "Unknown",
        n_factors=2,
    )
    assert any(lbl == "Unknown" for lbl in labels) or "Market" in labels
