"""Point-in-time S&P 500 universe builder.

A short static list of large-cap members is shipped so that the
universe builder is fully offline by default; when a Polygon hook is
provided the universe is refined using exchange membership and the
ticker's first/last-traded dates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

# Static fallback universe: a representative cross-sector basket large
# enough to drive a Marchenko-Pastur analysis (q ~ N/T well inside the
# bulk-regime).  Sector tags follow GICS top-level naming.
STATIC_UNIVERSE: dict[str, tuple[str, date, date | None]] = {
    "AAPL": ("Information Technology", date(1980, 12, 12), None),
    "MSFT": ("Information Technology", date(1986, 3, 13), None),
    "NVDA": ("Information Technology", date(1999, 1, 22), None),
    "GOOGL": ("Communication Services", date(2004, 8, 19), None),
    "AMZN": ("Consumer Discretionary", date(1997, 5, 15), None),
    "META": ("Communication Services", date(2012, 5, 18), None),
    "TSLA": ("Consumer Discretionary", date(2010, 6, 29), None),
    "AVGO": ("Information Technology", date(2009, 8, 6), None),
    "ORCL": ("Information Technology", date(1986, 3, 12), None),
    "CRM": ("Information Technology", date(2004, 6, 23), None),
    "JPM": ("Financials", date(1980, 1, 2), None),
    "BAC": ("Financials", date(1980, 1, 2), None),
    "WFC": ("Financials", date(1980, 1, 2), None),
    "GS": ("Financials", date(1999, 5, 4), None),
    "MS": ("Financials", date(1993, 2, 23), None),
    "BLK": ("Financials", date(1999, 10, 1), None),
    "BRK-B": ("Financials", date(1996, 5, 9), None),
    "V": ("Financials", date(2008, 3, 19), None),
    "MA": ("Financials", date(2006, 5, 25), None),
    "AXP": ("Financials", date(1980, 1, 2), None),
    "JNJ": ("Health Care", date(1980, 1, 2), None),
    "UNH": ("Health Care", date(1984, 10, 17), None),
    "PFE": ("Health Care", date(1980, 1, 2), None),
    "ABBV": ("Health Care", date(2013, 1, 2), None),
    "LLY": ("Health Care", date(1980, 1, 2), None),
    "MRK": ("Health Care", date(1980, 1, 2), None),
    "TMO": ("Health Care", date(1980, 1, 2), None),
    "ABT": ("Health Care", date(1980, 1, 2), None),
    "BMY": ("Health Care", date(1980, 1, 2), None),
    "AMGN": ("Health Care", date(1983, 6, 17), None),
    "XOM": ("Energy", date(1980, 1, 2), None),
    "CVX": ("Energy", date(1980, 1, 2), None),
    "COP": ("Energy", date(1981, 11, 19), None),
    "EOG": ("Energy", date(1989, 10, 1), None),
    "SLB": ("Energy", date(1980, 1, 2), None),
    "WMT": ("Consumer Staples", date(1980, 1, 2), None),
    "PG": ("Consumer Staples", date(1980, 1, 2), None),
    "KO": ("Consumer Staples", date(1980, 1, 2), None),
    "PEP": ("Consumer Staples", date(1980, 1, 2), None),
    "COST": ("Consumer Staples", date(1985, 12, 5), None),
    "MCD": ("Consumer Discretionary", date(1980, 1, 2), None),
    "HD": ("Consumer Discretionary", date(1981, 9, 22), None),
    "NKE": ("Consumer Discretionary", date(1980, 12, 2), None),
    "SBUX": ("Consumer Discretionary", date(1992, 6, 26), None),
    "LOW": ("Consumer Discretionary", date(1980, 1, 2), None),
    "BA": ("Industrials", date(1980, 1, 2), None),
    "CAT": ("Industrials", date(1980, 1, 2), None),
    "GE": ("Industrials", date(1980, 1, 2), None),
    "HON": ("Industrials", date(1980, 1, 2), None),
    "UNP": ("Industrials", date(1980, 1, 2), None),
    "UPS": ("Industrials", date(1999, 11, 10), None),
    "RTX": ("Industrials", date(1980, 1, 2), None),
    "LMT": ("Industrials", date(1995, 3, 16), None),
    "DE": ("Industrials", date(1980, 1, 2), None),
    "MMM": ("Industrials", date(1980, 1, 2), None),
    "NEE": ("Utilities", date(1980, 1, 2), None),
    "SO": ("Utilities", date(1980, 1, 2), None),
    "DUK": ("Utilities", date(1980, 1, 2), None),
    "AEP": ("Utilities", date(1980, 1, 2), None),
    "D": ("Utilities", date(1980, 1, 2), None),
    "T": ("Communication Services", date(1980, 1, 2), None),
    "VZ": ("Communication Services", date(1983, 11, 21), None),
    "NFLX": ("Communication Services", date(2002, 5, 23), None),
    "DIS": ("Communication Services", date(1980, 1, 2), None),
    "CMCSA": ("Communication Services", date(1980, 1, 2), None),
    "LIN": ("Materials", date(1980, 1, 2), None),
    "APD": ("Materials", date(1980, 1, 2), None),
    "SHW": ("Materials", date(1980, 1, 2), None),
    "FCX": ("Materials", date(1995, 7, 6), None),
    "NEM": ("Materials", date(1980, 1, 2), None),
    "PLD": ("Real Estate", date(1997, 11, 21), None),
    "AMT": ("Real Estate", date(1998, 6, 5), None),
    "EQIX": ("Real Estate", date(2000, 8, 11), None),
    "CCI": ("Real Estate", date(1998, 8, 18), None),
    "PSA": ("Real Estate", date(1980, 11, 14), None),
}


@dataclass(frozen=True)
class UniverseEntry:
    """A single point-in-time universe row."""

    ticker: str
    sector: str
    first_traded: date
    last_traded: date | None


def sector_lookup(ticker: str) -> str:
    """Return the static sector label for ``ticker``.

    Tickers absent from :data:`STATIC_UNIVERSE` are mapped to ``"Unknown"``.
    """
    entry = STATIC_UNIVERSE.get(ticker)
    return entry[0] if entry is not None else "Unknown"


def build_universe(
    as_of: date,
    polygon_hook: Callable[[date], list[UniverseEntry]] | None = None,
) -> list[UniverseEntry]:
    """Return the point-in-time universe valid on ``as_of``.

    Parameters
    ----------
    as_of:
        The date for which the universe is requested.  Only tickers whose
        ``first_traded <= as_of`` and ``last_traded is None or as_of <=
        last_traded`` are included.
    polygon_hook:
        Optional callable that returns a refined list of entries from a
        live Polygon snapshot.  When provided, its output is returned
        unchanged.
    """
    if polygon_hook is not None:
        return list(polygon_hook(as_of))

    entries: list[UniverseEntry] = []
    for ticker, (sector, first, last) in STATIC_UNIVERSE.items():
        if first > as_of:
            continue
        if last is not None and last < as_of:
            continue
        entries.append(
            UniverseEntry(
                ticker=ticker,
                sector=sector,
                first_traded=first,
                last_traded=last,
            )
        )
    return entries
