"""Streamlit demo for the eigen-portfolios pipeline."""

from __future__ import annotations

from datetime import date

import streamlit as st

from eigenportfolios.cli import run_pipeline
from eigenportfolios.data import load_prices
from eigenportfolios.pca import PCA
from eigenportfolios.plots import (
    factor_returns_figure,
    spectrum_figure,
    weights_heatmap_figure,
)
from eigenportfolios.universe import STATIC_UNIVERSE

st.set_page_config(page_title="Eigen-Portfolios", layout="wide")

st.title("Eigen-Portfolios")
st.caption(
    "PCA on financial returns with Marchenko-Pastur signal/noise separation."
)

with st.sidebar:
    st.header("Universe")
    default_tickers = list(STATIC_UNIVERSE.keys())[:20]
    tickers = st.multiselect("Tickers", default_tickers, default=default_tickers)
    start = st.date_input("Start", value=date(2020, 1, 1))
    end = st.date_input("End", value=date(2024, 12, 31))
    n_factors = st.slider("Factors to label", 1, 10, 5)
    submit = st.button("Run")

if submit:
    prices = load_prices(tickers=tickers, start=start, end=end)
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    summary = run_pipeline(returns=returns, n_factors=n_factors)
    st.subheader("Summary")
    st.write(
        {
            "N": summary["n"],
            "T": summary["t"],
            "q": summary["q"],
            "sigma": summary["sigma"],
            "lambda+": summary["lambda_plus"],
            "signal modes": summary["n_signal"],
            "labels": summary["labels"],
        }
    )

    result = PCA().fit(returns)
    st.plotly_chart(
        spectrum_figure(result, q=summary["q"], sigma=summary["sigma"]),
        use_container_width=True,
    )
    st.plotly_chart(
        factor_returns_figure(result, k=min(3, result.eigvecs.shape[1])),
        use_container_width=True,
    )
    st.plotly_chart(
        weights_heatmap_figure(
            result.eigvecs,
            tickers=list(returns.columns),
            k=min(5, result.eigvecs.shape[1]),
        ),
        use_container_width=True,
    )
