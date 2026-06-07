# eigen-portfolios

> **Now live as an interactive web tool at https://fatihhekimoglu-platform.vercel.app/tools/eigen-portfolios** — part of the fatihhekimoglu.com quantitative-tools platform.

[![ci](https://github.com/FatihHekim0glu/eigen-portfolios/actions/workflows/ci.yml/badge.svg)](https://github.com/FatihHekim0glu/eigen-portfolios/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen.svg)](#validation)
[![type-checked](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)

A from-scratch, research-grade Python toolkit that

1. estimates the empirical spectrum of a return correlation matrix via
   principal component analysis,
2. fits the Marchenko-Pastur (MP) noise spectrum and separates signal
   eigenmodes from noise, and
3. interprets the surviving signal modes by sector composition.

The package is intentionally compact (one numerical kernel per concern)
and is validated against scikit-learn PCA, the closed-form Marchenko-Pastur
edges, and the qualitative spectrum reported by Plerou et al. (1999).

## Method

For a `T x N` return matrix `R`, let `R_std` be the column-standardised
returns and `C = corr(R)` the sample correlation matrix. PCA solves the
symmetric eigenproblem

$$\mathbf{C}\,\mathbf{v}_{k} = \lambda_{k}\,\mathbf{v}_{k},
\qquad \lambda_{1} \geq \lambda_{2} \geq \cdots \geq \lambda_{N} \geq 0,$$

with orthonormal eigenvectors $\mathbf{v}_{k}$. Factor returns are

$$\mathbf{F} = \mathbf{R}_{\mathrm{std}}\,\mathbf{V}.$$

For an i.i.d. noise matrix with variance $\sigma^{2}$ and aspect ratio
$q = N / T \in (0, 1]$, the Marchenko-Pastur (1967) bulk has density

$$f(\lambda) =
\frac{1}{2\pi\sigma^{2}}\,
\frac{\sqrt{(\lambda_{+}-\lambda)(\lambda - \lambda_{-})}}{q\,\lambda},
\quad \lambda \in [\lambda_{-}, \lambda_{+}],$$

with closed-form edges

$$\lambda_{\pm} = \sigma^{2}\,(1 \pm \sqrt{q})^{2}.$$

`eigen-portfolios` estimates $\sigma$ by a bounded least-squares fit of
the MP density to the empirical eigenvalue histogram (restricted to the
candidate bulk), and labels every eigenmode with
$\lambda_{k} > \lambda_{+}$ as "signal". This is the workflow used by
Laloux, Cizeau, Bouchaud, and Potters (1999) on the S&P 500.

## Install

```bash
uv sync --all-extras
```

## Local development

```bash
# tests
uv run pytest -q --cov=eigenportfolios --cov-report=term

# lint + types
uv run ruff check src
uv run mypy --strict src
```

## CLI

```bash
uv run eigenportfolios run \
  --tickers AAPL,MSFT,NVDA,JPM,XOM,JNJ,WMT,BA,NEE \
  --start 2020-01-01 --end 2024-12-31
```

## Streamlit demo

```bash
uv run streamlit run app.py
```

## Validation

| Test                                                                | Expected                                                          | Tolerance      |
| ------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------- |
| MP closed-form edge ($\sigma=1$, $N=100$, $T=200$)                  | $\lambda_{+} = (1 + \sqrt{0.5})^{2} \approx 2.9142$               | $< 10^{-12}$   |
| MP bulk density integrates to one on $[\lambda_{-}, \lambda_{+}]$   | $1.0$                                                             | $< 10^{-3}$    |
| scikit-learn PCA parity (singular values, invariants)               | identical up to sign and column order                             | $< 10^{-10}$   |
| Block-diagonal correlation with $K$ blocks                          | $K$ eigenvalues $\gg 1$, rest $\approx 0$                         | exact          |
| Plerou et al. (1999) qualitative reproduction                       | majority of empirical eigenvalues fall inside the MP bulk         | qualitative    |

## Citations

```bibtex
@article{marchenko1967,
  author  = {Marchenko, V. A. and Pastur, L. A.},
  title   = {Distribution of eigenvalues for some sets of random matrices},
  journal = {Matematicheskii Sbornik},
  volume  = {72},
  number  = {4},
  pages   = {507--536},
  year    = {1967}
}

@article{laloux1999noise,
  author  = {Laloux, Laurent and Cizeau, Pierre and Bouchaud, Jean-Philippe
             and Potters, Marc},
  title   = {Noise dressing of financial correlation matrices},
  journal = {Physical Review Letters},
  volume  = {83},
  number  = {7},
  pages   = {1467--1470},
  year    = {1999}
}

@article{plerou1999universal,
  author  = {Plerou, Vasiliki and Gopikrishnan, Parameswaran and Rosenow,
             Bernd and Amaral, Luis A. Nunes and Stanley, H. Eugene},
  title   = {Universal and nonuniversal properties of cross correlations in
             financial time series},
  journal = {Physical Review Letters},
  volume  = {83},
  number  = {7},
  pages   = {1471--1474},
  year    = {1999}
}
```

## License

MIT. See [LICENSE](LICENSE).
