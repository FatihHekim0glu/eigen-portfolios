"""Price loader with optional Polygon and yfinance backends and disk cache."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "eigenportfolios"


@dataclass(frozen=True)
class PriceRequest:
    """A normalised request for daily close prices."""

    tickers: tuple[str, ...]
    start: date
    end: date

    def cache_key(self, source: str) -> str:
        """Stable string key for the request."""
        tickers = ",".join(sorted(self.tickers))
        return f"{source}|{tickers}|{self.start.isoformat()}|{self.end.isoformat()}"


def load_prices(
    tickers: Iterable[str],
    start: date | str,
    end: date | str,
    source: str = "yfinance",
    cache_dir: Path | str | None = None,
    polygon_api_key: str | None = None,
) -> pd.DataFrame:
    """Load a wide ``T x N`` daily close-price ``DataFrame``.

    Parameters
    ----------
    tickers:
        Iterable of ticker strings.
    start, end:
        Inclusive date range (``date`` or ISO string).
    source:
        ``"polygon"`` or ``"yfinance"``.  When ``"polygon"`` is requested
        but ``polygon_api_key`` is ``None`` the function falls back to
        ``"yfinance"``.
    cache_dir:
        Directory used for the disk cache.  ``None`` falls back to
        :data:`DEFAULT_CACHE_DIR`.
    polygon_api_key:
        Polygon API key.  When ``None`` the ``POLYGON_API_KEY``
        environment variable is checked.
    """
    req = PriceRequest(
        tickers=tuple(sorted(set(tickers))),
        start=_coerce_date(start),
        end=_coerce_date(end),
    )
    if not req.tickers:
        raise ValueError("tickers must be non-empty")
    if req.end <= req.start:
        raise ValueError(f"end ({req.end}) must be strictly greater than start ({req.start})")

    cache = _open_cache(cache_dir)
    key = req.cache_key(source)
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return _decode_frame(cached)

    if source == "polygon":
        api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")
        if not api_key:
            source = "yfinance"
        else:
            frame = _load_polygon(req, api_key=api_key)
            if cache is not None:
                cache.set(key, _encode_frame(frame))
            return frame

    if source == "yfinance":
        frame = _load_yfinance(req)
        if cache is not None:
            cache.set(key, _encode_frame(frame))
        return frame

    raise ValueError(f"unknown source: {source!r}")


def _coerce_date(value: date | str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"unsupported date type: {type(value).__name__}")


def _open_cache(cache_dir: Path | str | None) -> Any | None:
    """Open the disk cache; return ``None`` when ``diskcache`` is missing."""
    try:
        import diskcache
    except ImportError:
        return None
    target = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(str(target))


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "index": frame.index.astype(str).tolist(),
        "index_dtype": str(frame.index.dtype),
        "columns": list(frame.columns),
        "values": frame.to_numpy().tolist(),
    }


def _decode_frame(payload: dict[str, Any]) -> pd.DataFrame:
    index = pd.to_datetime(payload["index"])
    target_dtype = payload.get("index_dtype")
    if target_dtype is not None and str(index.dtype) != target_dtype:
        with contextlib.suppress(TypeError, ValueError):
            index = index.astype(target_dtype)
    return pd.DataFrame(
        np.asarray(payload["values"], dtype=float),
        index=index,
        columns=payload["columns"],
    )


def _load_polygon(req: PriceRequest, api_key: str) -> pd.DataFrame:
    """Fetch daily closes from Polygon using ``httpx``."""
    import httpx

    frames: list[pd.DataFrame] = []
    base = "https://api.polygon.io/v2/aggs/ticker"
    with httpx.Client(timeout=30.0) as client:
        for ticker in req.tickers:
            url = f"{base}/{ticker}/range/1/day/{req.start.isoformat()}/{req.end.isoformat()}"
            params = {"adjusted": "true", "sort": "asc", "apiKey": api_key}
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results") or []
            if not rows:
                continue
            ts = pd.to_datetime([r["t"] for r in rows], unit="ms")
            closes = [float(r["c"]) for r in rows]
            frames.append(pd.DataFrame({ticker: closes}, index=ts))
    if not frames:
        raise RuntimeError("polygon returned no data for requested universe")
    merged = pd.concat(frames, axis=1).sort_index()
    result: pd.DataFrame = merged.dropna(how="all")
    return result


def _load_yfinance(req: PriceRequest) -> pd.DataFrame:
    """Fetch daily closes via the yfinance backend."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance backend requested but yfinance is not installed") from exc

    raw = yf.download(
        list(req.tickers),
        start=req.start.isoformat(),
        end=req.end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    closes = _extract_closes_from_yfinance(raw, req.tickers)
    closes = closes.sort_index().dropna(how="all")
    if closes.empty:
        raise RuntimeError("yfinance returned no data for requested universe")
    out: pd.DataFrame = closes
    return out


def _extract_closes_from_yfinance(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        closes: dict[str, pd.Series] = {}
        for ticker in tickers:
            if ticker in raw.columns.get_level_values(0):
                closes[ticker] = raw[ticker]["Close"]
        return pd.DataFrame(closes)
    # Single ticker case.
    return pd.DataFrame({tickers[0]: raw["Close"]})
