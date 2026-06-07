"""Unit tests for :mod:`eigenportfolios.data` (network calls mocked)."""

from __future__ import annotations

from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from eigenportfolios.data import PriceRequest, load_prices


@pytest.mark.unit
def test_price_request_cache_key_stable():
    req = PriceRequest(
        tickers=("AAPL", "MSFT"),
        start=date(2020, 1, 1),
        end=date(2024, 1, 1),
    )
    key = req.cache_key("polygon")
    assert "AAPL,MSFT" in key
    assert "polygon" in key
    assert "2020-01-01" in key


@pytest.mark.unit
def test_price_request_cache_key_orders_tickers():
    req = PriceRequest(
        tickers=("MSFT", "AAPL"),
        start=date(2020, 1, 1),
        end=date(2024, 1, 1),
    )
    key = req.cache_key("yfinance")
    # tickers should be sorted in the cache key for stability
    assert "AAPL,MSFT" in key


@pytest.mark.unit
def test_load_prices_rejects_empty_tickers(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        load_prices(
            tickers=[],
            start=date(2020, 1, 1),
            end=date(2020, 6, 1),
            cache_dir=tmp_path,
        )


@pytest.mark.unit
def test_load_prices_rejects_end_before_start(tmp_path):
    with pytest.raises(ValueError, match="strictly greater"):
        load_prices(
            tickers=["AAPL"],
            start=date(2020, 6, 1),
            end=date(2020, 1, 1),
            cache_dir=tmp_path,
        )


@pytest.mark.unit
def test_load_prices_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="unknown source"):
        load_prices(
            tickers=["AAPL"],
            start=date(2020, 1, 1),
            end=date(2020, 6, 1),
            source="bloomberg",
            cache_dir=tmp_path,
        )


@pytest.mark.unit
def test_load_prices_string_dates_accepted(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_polygon(req, api_key):
        captured["start"] = req.start
        captured["end"] = req.end
        idx = pd.date_range(req.start, periods=3, freq="D")
        return pd.DataFrame({"AAPL": [1.0, 2.0, 3.0]}, index=idx)

    monkeypatch.setattr("eigenportfolios.data._load_polygon", fake_polygon)
    frame = load_prices(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-01-05",
        source="polygon",
        polygon_api_key="dummy",
        cache_dir=tmp_path,
    )
    assert captured["start"] == date(2020, 1, 1)
    assert captured["end"] == date(2020, 1, 5)
    assert "AAPL" in frame.columns


@pytest.mark.unit
def test_load_prices_polygon_mocked(tmp_path):
    with respx.mock(base_url="https://api.polygon.io") as router:
        router.get("/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"t": 1704067200000, "c": 100.0},
                        {"t": 1704153600000, "c": 101.0},
                        {"t": 1704240000000, "c": 102.0},
                    ]
                },
            )
        )
        frame = load_prices(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-05",
            source="polygon",
            polygon_api_key="dummy",
            cache_dir=tmp_path,
        )
    assert list(frame.columns) == ["AAPL"]
    assert frame.shape[0] == 3


@pytest.mark.unit
def test_load_prices_polygon_caches_second_call(tmp_path):
    with respx.mock(base_url="https://api.polygon.io") as router:
        route = router.get("/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05").mock(
            return_value=httpx.Response(
                200,
                json={"results": [{"t": 1704067200000, "c": 100.0}]},
            )
        )
        first = load_prices(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-05",
            source="polygon",
            polygon_api_key="dummy",
            cache_dir=tmp_path,
        )
        # second call should hit the cache rather than the network
        second = load_prices(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-05",
            source="polygon",
            polygon_api_key="dummy",
            cache_dir=tmp_path,
        )
    assert route.call_count == 1
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.unit
def test_load_prices_polygon_falls_back_to_yfinance(tmp_path, monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    called: dict[str, bool] = {"yf": False}

    def fake_yf(req):
        called["yf"] = True
        idx = pd.date_range(req.start, periods=2, freq="D")
        return pd.DataFrame({"AAPL": [1.0, 2.0]}, index=idx)

    monkeypatch.setattr("eigenportfolios.data._load_yfinance", fake_yf)
    frame = load_prices(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-01-05",
        source="polygon",
        cache_dir=tmp_path,
    )
    assert called["yf"] is True
    assert "AAPL" in frame.columns


@pytest.mark.unit
def test_load_prices_polygon_empty_raises(tmp_path):
    with respx.mock(base_url="https://api.polygon.io") as router:
        router.get("/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        with pytest.raises(RuntimeError, match="no data"):
            load_prices(
                tickers=["AAPL"],
                start="2024-01-01",
                end="2024-01-05",
                source="polygon",
                polygon_api_key="dummy",
                cache_dir=tmp_path,
            )


@pytest.mark.unit
def test_load_prices_yfinance_mocked(tmp_path, monkeypatch):
    def fake_yf(req):
        idx = pd.date_range(req.start, periods=4, freq="D")
        return pd.DataFrame(
            np.arange(8, dtype=float).reshape(4, 2),
            index=idx,
            columns=["AAPL", "MSFT"],
        )

    monkeypatch.setattr("eigenportfolios.data._load_yfinance", fake_yf)
    frame = load_prices(
        tickers=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-05",
        source="yfinance",
        cache_dir=tmp_path,
    )
    assert list(frame.columns) == ["AAPL", "MSFT"]
    assert frame.shape == (4, 2)


@pytest.mark.unit
def test_coerce_date_accepts_datetime(tmp_path, monkeypatch):
    from datetime import datetime as dt

    captured: dict[str, date] = {}

    def fake_yf(req):
        captured["start"] = req.start
        captured["end"] = req.end
        idx = pd.date_range(req.start, periods=2, freq="D")
        return pd.DataFrame({"AAPL": [1.0, 2.0]}, index=idx)

    monkeypatch.setattr("eigenportfolios.data._load_yfinance", fake_yf)
    load_prices(
        tickers=["AAPL"],
        start=dt(2020, 1, 1, 12, 0, 0),
        end=dt(2020, 6, 1, 12, 0, 0),
        source="yfinance",
        cache_dir=tmp_path,
    )
    assert captured["start"] == date(2020, 1, 1)
    assert captured["end"] == date(2020, 6, 1)


@pytest.mark.unit
def test_coerce_date_rejects_bad_type(tmp_path):
    with pytest.raises(TypeError, match="unsupported date type"):
        load_prices(
            tickers=["AAPL"],
            start=12345,  # type: ignore[arg-type]
            end="2020-06-01",
            cache_dir=tmp_path,
        )


@pytest.mark.unit
def test_open_cache_creates_directory(tmp_path):
    """The disk cache should create the target directory if it does not exist."""
    from eigenportfolios.data import _open_cache

    target = tmp_path / "nested" / "cache"
    cache = _open_cache(target)
    assert cache is not None
    assert target.exists()


@pytest.mark.unit
def test_load_yfinance_missing_module(monkeypatch, tmp_path):
    """When yfinance is not importable, _load_yfinance raises RuntimeError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("no yfinance")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="yfinance backend"):
        load_prices(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-05",
            source="yfinance",
            cache_dir=tmp_path,
        )


@pytest.mark.unit
def test_load_yfinance_calls_real_backend(monkeypatch, tmp_path):
    """End-to-end: _load_yfinance is invoked and reshapes the MultiIndex frame."""
    import types

    n_periods = 6
    idx = pd.date_range("2024-01-01", periods=n_periods, freq="B")
    multi = pd.DataFrame(
        np.arange(n_periods * 4, dtype=float).reshape(n_periods, 4),
        index=idx,
        columns=pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "Close"]]),
    )

    def fake_download(*args, **kwargs):
        return multi

    fake_yf = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    frame = load_prices(
        tickers=["AAPL", "MSFT"],
        start="2024-01-01",
        end="2024-01-10",
        source="yfinance",
        cache_dir=tmp_path,
    )
    assert set(frame.columns) == {"AAPL", "MSFT"}
    assert frame.shape[0] == n_periods


@pytest.mark.unit
def test_load_yfinance_empty_raises(monkeypatch, tmp_path):
    import types

    def fake_download(*args, **kwargs):
        return pd.DataFrame(columns=pd.MultiIndex.from_product([["AAPL"], ["Close"]]))

    fake_yf = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    with pytest.raises(RuntimeError, match="no data"):
        load_prices(
            tickers=["AAPL"],
            start="2024-01-01",
            end="2024-01-10",
            source="yfinance",
            cache_dir=tmp_path,
        )


@pytest.mark.unit
def test_load_yfinance_single_ticker(monkeypatch, tmp_path):
    """Single-ticker yfinance returns a flat-column frame; we handle the path."""
    import types

    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    flat = pd.DataFrame(
        {"Open": [1.0, 2.0, 3.0], "Close": [10.0, 11.0, 12.0]},
        index=idx,
    )

    def fake_download(*args, **kwargs):
        return flat

    fake_yf = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    frame = load_prices(
        tickers=["AAPL"],
        start="2024-01-01",
        end="2024-01-10",
        source="yfinance",
        cache_dir=tmp_path,
    )
    assert list(frame.columns) == ["AAPL"]


@pytest.mark.unit
def test_load_prices_uses_default_cache_when_none(monkeypatch, tmp_path):
    """Passing cache_dir=None invokes DEFAULT_CACHE_DIR creation."""
    from eigenportfolios import data as data_mod

    monkeypatch.setattr(data_mod, "DEFAULT_CACHE_DIR", tmp_path / "default")

    def fake_yf(req):
        return pd.DataFrame(
            {"AAPL": [1.0]},
            index=pd.date_range(req.start, periods=1, freq="D"),
        )

    monkeypatch.setattr("eigenportfolios.data._load_yfinance", fake_yf)
    frame = load_prices(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-01-05",
        source="yfinance",
    )
    assert "AAPL" in frame.columns


@pytest.mark.unit
def test_load_prices_polygon_env_key(monkeypatch, tmp_path):
    """POLYGON_API_KEY env var triggers the polygon path."""
    monkeypatch.setenv("POLYGON_API_KEY", "envkey")
    captured: dict[str, str] = {}

    def fake_polygon(req, api_key):
        captured["key"] = api_key
        idx = pd.date_range(req.start, periods=2, freq="D")
        return pd.DataFrame({"AAPL": [1.0, 2.0]}, index=idx)

    monkeypatch.setattr("eigenportfolios.data._load_polygon", fake_polygon)
    load_prices(
        tickers=["AAPL"],
        start="2020-01-01",
        end="2020-01-05",
        source="polygon",
        cache_dir=tmp_path,
    )
    assert captured["key"] == "envkey"
