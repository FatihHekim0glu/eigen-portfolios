"""Unit tests for :mod:`eigenportfolios.universe`."""

from __future__ import annotations

from datetime import date

import pytest

from eigenportfolios.universe import (
    STATIC_UNIVERSE,
    UniverseEntry,
    build_universe,
    sector_lookup,
)


@pytest.mark.unit
def test_static_universe_nonempty():
    assert len(STATIC_UNIVERSE) > 50


@pytest.mark.unit
def test_sector_lookup_known():
    assert sector_lookup("AAPL") == "Information Technology"


@pytest.mark.unit
def test_sector_lookup_unknown():
    assert sector_lookup("ZZZZNOTREAL") == "Unknown"


@pytest.mark.unit
def test_build_universe_offline_returns_entries():
    entries = build_universe(as_of=date(2024, 1, 1))
    assert all(isinstance(e, UniverseEntry) for e in entries)
    assert len(entries) > 50


@pytest.mark.unit
def test_build_universe_filters_future_listings():
    # META first traded 2012-05-18; before that date it must be filtered out
    entries = build_universe(as_of=date(2010, 1, 1))
    tickers = [e.ticker for e in entries]
    assert "META" not in tickers


@pytest.mark.unit
def test_build_universe_filters_delisted():
    fake = {
        "OLD": ("Industrials", date(1990, 1, 1), date(2000, 1, 1)),
    }

    # patch by passing a hook that emulates a delisted ticker
    def hook(as_of: date) -> list[UniverseEntry]:
        return [
            UniverseEntry("OLD", *fake["OLD"]),
        ]

    entries = build_universe(as_of=date(2024, 1, 1), polygon_hook=hook)
    assert entries[0].ticker == "OLD"


@pytest.mark.unit
def test_build_universe_skips_delisted_via_static():
    """A static-universe entry with last_traded < as_of must be excluded."""
    from eigenportfolios import universe as uni

    custom = {
        "OLD": ("Industrials", date(1990, 1, 1), date(2000, 1, 1)),
        "NEW": ("Industrials", date(1990, 1, 1), None),
    }
    original = uni.STATIC_UNIVERSE.copy()
    try:
        uni.STATIC_UNIVERSE.clear()
        uni.STATIC_UNIVERSE.update(custom)
        entries = uni.build_universe(as_of=date(2024, 1, 1))
        tickers = [e.ticker for e in entries]
        assert "NEW" in tickers
        assert "OLD" not in tickers
    finally:
        uni.STATIC_UNIVERSE.clear()
        uni.STATIC_UNIVERSE.update(original)


@pytest.mark.unit
def test_build_universe_polygon_hook_overrides():
    def hook(as_of: date) -> list[UniverseEntry]:
        return [UniverseEntry("XYZ", "Energy", date(2020, 1, 1), None)]

    entries = build_universe(as_of=date(2024, 1, 1), polygon_hook=hook)
    assert [e.ticker for e in entries] == ["XYZ"]
