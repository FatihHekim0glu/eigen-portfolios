"""Unit tests for :mod:`eigenportfolios.cli`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from eigenportfolios.cli import app, run_pipeline


@pytest.mark.unit
def test_run_pipeline_summary_keys(synthetic_one_factor):
    summary = run_pipeline(returns=synthetic_one_factor, n_factors=3)
    assert {"n", "t", "q", "sigma", "lambda_plus", "n_signal", "labels", "eigvals"} <= set(summary)


@pytest.mark.unit
def test_run_pipeline_signal_count_one_factor(synthetic_one_factor):
    summary = run_pipeline(returns=synthetic_one_factor, n_factors=5)
    assert summary["n_signal"] >= 1


@pytest.mark.unit
def test_run_pipeline_rejects_q_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((5, 10)), columns=[f"A{i}" for i in range(10)])
    with pytest.raises(ValueError, match="q"):
        run_pipeline(returns=df, n_factors=2)


@pytest.mark.unit
def test_run_pipeline_rejects_empty():
    df = pd.DataFrame(columns=["A"])
    with pytest.raises(ValueError, match="insufficient"):
        run_pipeline(returns=df, n_factors=1)


@pytest.mark.unit
def test_cli_help_runs():
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "tickers" in res.stdout.lower()


@pytest.mark.unit
def test_cli_run_with_mocked_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_assets, n_periods = 10, 200
    idx = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    prices = pd.DataFrame(
        100.0 + np.cumsum(rng.standard_normal((n_periods, n_assets)), axis=0),
        index=idx,
        columns=[f"T{i:02d}" for i in range(n_assets)],
    )

    def fake_load(tickers, start, end, source):
        return prices

    monkeypatch.setattr("eigenportfolios.cli.load_prices", fake_load)
    runner = CliRunner()
    out = tmp_path / "eig.csv"
    res = runner.invoke(
        app,
        [
            "--tickers",
            "T00,T01,T02,T03,T04,T05,T06,T07,T08,T09",
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
            "--n-factors",
            "3",
            "--output",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out.exists()


@pytest.mark.unit
def test_cli_rejects_empty_tickers():
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--tickers",
            "",
            "--start",
            "2020-01-01",
            "--end",
            "2020-12-31",
        ],
    )
    assert res.exit_code != 0
