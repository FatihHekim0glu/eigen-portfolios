"""Typer CLI for the eigen-portfolios pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from eigenportfolios.data import load_prices
from eigenportfolios.interpretation import label_factors
from eigenportfolios.pca import PCA
from eigenportfolios.rmt import separate_signal_noise
from eigenportfolios.universe import sector_lookup

app = typer.Typer(
    add_completion=False,
    help="PCA + Marchenko-Pastur signal/noise separation on financial returns.",
)

_TICKERS_OPT = typer.Option(
    ...,
    "--tickers",
    help="Comma-separated list of tickers, e.g. AAPL,MSFT,NVDA.",
)
_START_OPT = typer.Option(..., "--start", help="Start date (YYYY-MM-DD).")
_END_OPT = typer.Option(..., "--end", help="End date (YYYY-MM-DD).")
_SOURCE_OPT = typer.Option("yfinance", "--source", help="Price source: 'yfinance' or 'polygon'.")
_NFACT_OPT = typer.Option(5, "--n-factors", help="Number of leading factors to label.")
_OUTPUT_OPT = typer.Option(None, "--output", help="Optional CSV path for eigenvalues.")


@app.command()
def run(
    tickers: str = _TICKERS_OPT,
    start: str = _START_OPT,
    end: str = _END_OPT,
    source: str = _SOURCE_OPT,
    n_factors: int = _NFACT_OPT,
    output: Path | None = _OUTPUT_OPT,
) -> None:
    """Run the end-to-end pipeline on the requested universe."""
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise typer.BadParameter("at least one ticker is required")
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    prices = load_prices(
        tickers=ticker_list,
        start=start_date,
        end=end_date,
        source=source,
    )
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    result = run_pipeline(returns=returns, n_factors=n_factors)

    typer.echo(f"N = {result['n']}  T = {result['t']}  q = {result['q']:.4f}")
    typer.echo(f"sigma (fitted) = {result['sigma']:.4f}")
    typer.echo(f"upper MP edge  = {result['lambda_plus']:.4f}")
    typer.echo(f"# signal modes = {result['n_signal']}")
    labels_obj = result["labels"]
    assert isinstance(labels_obj, list)
    typer.echo("labels: " + ", ".join(labels_obj))
    if output is not None:
        pd.Series(result["eigvals"]).to_csv(output, index=False)
        typer.echo(f"wrote {output}")


def run_pipeline(returns: pd.DataFrame, n_factors: int) -> dict[str, object]:
    """Execute PCA + RMT split on ``returns`` and return a summary dict."""
    n = int(returns.shape[1])
    t = int(returns.shape[0])
    if t < 2 or n < 1:
        raise ValueError(f"insufficient data: T={t}, N={n}")
    q = n / t
    if q >= 1.0:
        raise ValueError(f"q = N/T = {q:.3f} must be < 1; need more rows than columns")
    eig_result = PCA().fit(returns)
    split = separate_signal_noise(eig_result.eigvals, q=q)
    labels = label_factors(
        eig_result,
        sector_lookup=sector_lookup,
        n_factors=min(n_factors, eig_result.n_features),
    )
    return {
        "n": n,
        "t": t,
        "q": q,
        "sigma": float(np.sqrt(split.sigma2)),
        "lambda_plus": float(split.lambda_plus),
        "n_signal": int(split.signal_indices.size),
        "labels": labels,
        "eigvals": eig_result.eigvals,
    }


if __name__ == "__main__":  # pragma: no cover
    app()
