# RMLM-Audit — Reproducing and Stress-Testing a Max-Linear Causal Graph of Extreme Industry Losses

Python implementation, reproduction and audit of

> Klüppelberg, C. and Krali, M. (2025). *Causal analysis of extreme risk in a network of industry portfolios.* arXiv:2504.00523 (v2); to appear in the Canadian Journal of Statistics.

The base paper estimates a **directed acyclic graph (DAG)** that describes how extreme daily losses propagate among 30 US industry portfolios, using a *recursive max-linear model* (RMLM). It returns **one** graph. This project (1) re-implements the method, (2) checks the implementation against the authors' R code, and (3) adds an **audit layer ("RMLM-Audit")** that asks how far that single graph can be trusted.

This README is written so that someone new to the project can understand every file, every concept and every result. Read sections 1–3 first; the rest is reference.

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Main findings](#2-main-findings)
3. [Background: the concepts you need](#3-background-the-concepts-you-need)
4. [Repository layout](#4-repository-layout)
5. [Setup and quick start](#5-setup-and-quick-start)
6. [Data](#6-data)
7. [Core library (`python/rmlm/`)](#7-core-library-pythonrmlm)
8. [Scripts reference (`python/*.py`)](#8-scripts-reference-pythonpy)
9. [The audit methodology (components A–E)](#9-the-audit-methodology-components-ae)
10. [Results folder guide](#10-results-folder-guide)
11. [How to reproduce everything, in order](#11-how-to-reproduce-everything-in-order)
12. [Parameters and defaults](#12-parameters-and-defaults)
13. [Tests](#13-tests)
14. [Known issues, caveats and limitations](#14-known-issues-caveats-and-limitations)
15. [FAQ](#15-faq)
16. [References](#16-references)

---

## 1. The project in one page

**Question of the base paper.** When one industry has a very large loss, which other industries are dragged along, and in which direction?

**Data.** Daily returns of 30 US industry portfolios (Kenneth French data library), 1 June 1989 – 15 June 1998 (n = 2285 days, d = 30 industries). Daily *losses* are `X = max(−return, 0)`.

**Model.** Recursive max-linear model on a DAG: the extreme loss of an industry is the **maximum** of its own fresh shock and scaled shocks passed on from its ancestors,

```
X_i = max_{j in An(i)} a_ij · Z_j ,     Z_j independent, Fréchet(α = 2)
```

where `A = (a_ij)` has unit row norms and `a_ij ≠ 0` only if `j` is an ancestor of `i` (or `j = i`).

**Estimator (three steps).**
1. *Causal order* (Algorithm 1): sort the industries into "generations", sources first.
2. *Coefficients* (Algorithm 2): estimate the matrix `A` given the order.
3. *Thresholding* (`δ`): keep an edge `j→i` only if it beats the best indirect path by more than `δ`.

**What we added.** A stress-test ("audit") of that estimator:

| Audit question | Tool | Script |
|---|---|---|
| Is the graph reproducible under resampling? | block bootstrap, split-half, rolling windows | `compare_methods.py`, `audit_extra.py split` |
| Does the instability come from the *order* or from the *coefficients*? | full vs fixed-order bootstrap | `boot_fixed_order.py` |
| How do the coefficients change in every resample and every window? | coefficient audit with figures | `coef_audit.py` |
| Does the model's structure fit the data? | tail-dependence goodness-of-fit with parametric bootstrap | `gof_check.py` |
| Can single edges be tested? | permutation test, conditional test, BH correction | `perm_edges.py`, `edge_test.py` |
| What does one added variable do? | proxy test with shuffled-column null, dilution test | `audit_extra.py proxy`, `null_proxy.py`, `dilution_real.py` |
| Is the causal order supported by an independent method? | causal tail coefficient | `baseline_order.py` |
| What does one extreme day do to the graph? | event study | `event_study.py` |
| Do these tools work when the model is *true*? | simulations E1–E8 | `experiments_all.py`, `ctrl30.py` |

---

## 2. Main findings

All real-data numbers come from the repository's own scripts on the 1989–1998 data (30 industries, n = 2285). Simulation results use d = 30 and few replications; treat them as indicative.

| # | Finding | Evidence |
|---|---|---|
| 1 | The single-run graph has **142 edges**, but only **3–5** reappear in ≥ 70 % of block-bootstrap resamples (3 with B = 200, 5 with B = 100; depends on B and seed). | `edges_bootstrap.csv`, `boot_edge_coef.csv` |
| 2 | Most of this instability comes from the **ordering step** (Algorithm 1): with the order held fixed, 57 edges (instead of 5) reach frequency ≥ 0.7. Mean Kendall τ between bootstrap and original order is only 0.29. Mines is a source in 90 % and Coal in 50 % of resamples. | `boot_fixed_order.py` |
| 3 | Coefficients jump between 0 and ≈ 0.7 from resample to resample; with the rule "freq ≥ 0.7 and 2.5 % quantile of a_ij > 0.05" **no** edge is "meaningful". | `coef_audit.py`, `fig_boot_coef_*.png` |
| 4 | Rolling 3-year windows: number of edges 84–129, nSHD to previous window 0.66–0.81, Kendall τ of the window order vs the full-sample order 0.17–0.43. Mines is the only source until 1996, then Coal; the Mines–Coal direction flips. | `rolling_features.csv`, `fig_roll_*.png` |
| 5 | The finite-atom structure of the model is **rejected** (statistic T = 0.188 vs 95th percentile 0.113 of the reference). It tests "≤ d atoms", **not** "max vs sum" (Remark 1 of the paper). | `gof_check.py` |
| 6 | Time-shift permutation and conditional edge tests flag almost all edges as significant (132/142 raw, 142/142 after BH; 128 and 133 for the conditional test). Because the null model does not fit, this is **not** read as "all edges are real". | `edges_permutation.csv`, `edges_conditional_test.csv` |
| 7 | Adding any extra column (even a shuffled one) makes it the only source and removes Mines and Coal as sources. | `null_proxy.csv` |
| 8 | The causal-tail-coefficient order agrees with Algorithm 1 not at all on real data (Kendall τ ≈ −0.006). | `baseline_order.py` |
| 9 | The real-data instability "fingerprint" (full vs fixed-order bootstrap) resembles a **misspecified** simulation (multivariate t) more than a correct max-linear one. This is suggestive, not proof. | `decomp` experiment in the proposal |
| 10 | GARCH(1,1)-filtered data (clustering removed) show the **same** picture: full bootstrap `[34, 7, 0]` vs fixed order `[117, 67, 23]` edges at frequency ≥ 0.5/0.7/0.9 (unfiltered, same seed: `[30, 4, 1]` vs `[103, 51, 16]`), Kendall τ 0.28 vs 0.32, goodness-of-fit T = 0.199 (95th pct 0.112) vs 0.187 (0.116). Univariate volatility clustering is therefore not the main source. | `garch_filter.py` |
| 11 | Relations are essentially **instantaneous**: for the 15 most frequent edges χ at lag 0 is 0.29 vs 0.07 averaged over lags 1…5 (chance 0.05); the peak is at lag 0 for 15/15 edges; no lead of parent over child (asymmetry −0.014; positive for 3/15). Small excess above chance at lags 1–2 (≈ 0.08–0.10) points to residual serial dependence. | `lag_check.py` |
| 12 | **Adjacency is more stable than direction**: 68 pairs are adjacent in ≥ 50 % of resamples but only 30 directed edges reach 50 %; 13 vs 4 at 70 %; mean direction consistency 0.78 (28 of 68 pairs below 0.75). Consecutive rolling windows: directed nSHD 0.79 vs skeleton nSHD 0.67. | `graph_audit.py` |
| 13 | The paper's rule (centroid over k) is almost as close to the bootstrap cloud as the best candidate (bootstrap medoid): mean nSHD 0.659 (k = 94) vs 0.640 (medoid) vs 0.655 (single run), but all are far (bootstrap graphs differ from each other by 0.71 on average). Window graphs are farther (0.77–0.81). | `graph_audit.py` |
| 14 | In simulation, thresholding at bootstrap frequency ≥ 0.5 raises precision (e.g. 0.27 → 0.73 at edge density 0.05) with about the same recall. A "calibrated" threshold brings no gain. | `proposed_exp.py` results in the proposal |

---

## 3. Background: the concepts you need

**DAG, parent, ancestor.** An edge `j→i` means "j is a direct cause of i". `An(i)` is `i` plus all its ancestors.

**Recursive max-linear model (RMLM).** `X = A ×_max Z`, i.e. `X_i = max_j a_ij Z_j`. Heavy-tailed shocks make *the largest* shock dominate on extreme days, which is why max (and not sum) is natural for extremes.

**Fréchet margins.** Each column is transformed by ranks to a standard Fréchet(2) margin:

```
X̃_{t,i} = [ −ln( rank_t,i / (n+1) ) ]^(−1/2)
```

This is why the **tail index is fixed at α = 2** and never estimated from the data.

**Radius and extreme days.** `R_t = ‖X̃_t‖₂`. Estimation uses only the `k` days with the largest radius.

**Scaling of a max-projection.** For a set of nodes `I`, `M_I = max_{k∈I} X_k` has scaling `σ²_{M_I} = Σ_ℓ max_{k∈I} a_kℓ²`. It is estimated from the `k` extreme days. All three estimation steps are built from such scalings.

**Algorithm 1 (causal order).** For a node `j` not yet ordered and `O` the set already ordered, compare scalings of `max(X_i, a·X_j, a·X_O)` with those of `max(X_i, X_j, X_O)`. By Theorem 5 of the paper, `j` has no ancestor outside `O` iff a certain difference equals `(a² − 1)` times a scaling, otherwise it is smaller. Nodes whose score is within `ε` of the best form the next *generation*. Output: list of generations; **generation 0 = sources**. Parameters: `a > 1` (default 1.3), `ε` (0.1 in the paper), `k` (≈ 250 = 0.11 n).

**Algorithm 2 (coefficients).** With the order known, `A²` is a linear function of scalings (Proposition 2). Negative values are set to 0, rows renormalised. Parameter `k` (≈ 90 = 0.04 n).

**Hard thresholding and the minimum DAG.** `A` is identifiable, the original edge weights are not; what is identifiable is the *minimum max-linear DAG* `D^A`. We keep `j→i` iff

```
a_ij > max_{k ≠ i,j}  a_ik · a_kj / a_kk  +  δ          (δ = 0.1)
```

**nSHD.** Structural Hamming distance between two edge sets, divided by the total number of edges in both: `nSHD = |E1 Δ E2| / (|E1| + |E2|)`. 0 = identical, 1 = no edge in common.

**Block bootstrap (moving block, Künsch 1989).** Rebuild a series of length n by gluing together ≈ n / ℓ blocks of ℓ = 20 consecutive days whose starting points are drawn uniformly at random. Preserves short-term serial dependence, which day-by-day resampling would destroy.

**Rolling window.** Re-run the whole pipeline on a window of 3 years (756 days), then slide by 1 year → 7 windows. Used to see whether the graph is stable in time. There is no train/test split: the model is a structure-estimation method, not a predictor.

**Permutation test.** Break the structure under test (here: circularly shift one column in time), recompute the statistic many times, and report `p = (1 + #{null ≥ observed}) / (P + 1)`. **Benjamini–Hochberg (BH)** controls the false discovery rate over many edges.

**Instability index.** `ρ_τ = 1 − N_τ^full / N_τ^fixed`, where `N_τ` is the number of edges with bootstrap frequency ≥ τ under the full (order re-estimated) or fixed-order bootstrap. Near 0 → coefficients are to blame; near 1 → the order is. Descriptive, not a test.

---

## 4. Repository layout

```
rmlm-industry-portfolios/
├── README.md                      this file (replaces the short original)
├── README.txt                     note about coef_audit.py
├── environment/
│   ├── requirements.txt           pip: numpy, pandas, pytest, networkx, matplotlib
│   └── environment.yml            conda env "rmlm" (python 3.11)
├── R/original/
│   └── CJS_functions_code.R       authors' R code (Mario Krali), unmodified
├── data/
│   ├── raw/30_Industry_Portfolios_Daily.csv   Ken French download
│   └── processed/                 X_losses.csv, X_frechet.csv (generated by data_prep)
├── python/
│   ├── rmlm/                      importable package (section 7)
│   │   ├── __init__.py            explains the two implementations
│   │   ├── validated_lib.py       from-scratch implementation, unit-tested
│   │   ├── original_port.py       line-by-line port of the authors' R code
│   │   ├── fast_lib.py            fast self-contained library used by all audits
│   │   ├── data_prep.py           raw csv → losses → Fréchet margins
│   │   └── thresholding.py        hard thresholding, SHD, nSHD, centroid graph
│   ├── run_pipeline.py            raw data → published-style graph + comparison to paper
│   ├── compare_methods.py         single run vs bootstrap vs rolling windows
│   ├── coef_audit.py              coefficient audit with figures (bootstrap + rolling)
│   ├── boot_fixed_order.py        order vs coefficient instability decomposition
│   ├── graph_audit.py             window graph drawings, 'which graph is closest to all others', directed vs undirected
│   ├── lag_check.py               lagged tail dependence: are relations instantaneous or lead-lag?
│   ├── garch_filter.py            GARCH(1,1) filter (volatility clustering removed), then re-run the audits
│   ├── audit_extra.py             split-half reproducibility and proxy test
│   ├── gof_check.py               tail-dependence goodness-of-fit
│   ├── perm_edges.py              permutation test per edge + BH/Bonferroni
│   ├── edge_test.py               conditional (parametric-bootstrap) edge test + BH
│   ├── baseline_order.py          causal tail coefficient ordering baseline
│   ├── event_study.py             extreme-day study
│   ├── make_proxy.py              downloads/builds the external proxy column
│   ├── null_proxy.py              real proxy vs shuffled proxies
│   ├── dilution_real.py           why does one extra column wreck the graph?
│   ├── ctrl30.py                  simulated control run at d = 30
│   ├── experiments_all.py         simulation experiments E1, E2, E4, E5, E8
│   └── tests/                     pytest suite (7 tests)
├── results/
│   ├── tables/                    csv outputs of the audits
│   ├── coef_audit/                figures and csv from coef_audit.py
│   └── figures/                   (empty placeholder)
├── notebooks/                     (empty placeholder)
└── docs/project_structure_and_testing_fa.md   Persian walk-through of structure, R code and testing
```

Scripts expect to be **run from inside `python/`** (they use relative paths such as `../data/processed/X_losses.csv`).

---

## 5. Setup and quick start

```bash
git clone https://github.com/kiashakooshan/rmlm-industry-portfolios.git
cd rmlm-industry-portfolios
pip install -r environment/requirements.txt
cd python
pytest tests/ -v                                       # 7 tests, < 1 s
python -m rmlm.data_prep                               # builds data/processed/X_losses.csv and X_frechet.csv
python run_pipeline.py                                 # published-style graph, compared with the paper
python compare_methods.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100
```

 Tested here with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, matplotlib 3.11.

---

## 6. Data

* **Source:** Kenneth R. French data library, *30 Industry Portfolios [Daily]*, value-weighted block (Fama–French style industry portfolios). Saved at `data/raw/30_Industry_Portfolios_Daily.csv`. The file is free to download from the French library.
* **Window:** `1989-06-01` to `1998-06-15` (as in the paper). 2285 trading days, 30 columns.
* **Losses:** returns are in percent; `X = max(−return/100, 0)`. Many entries are exactly 0 (days with a gain), which is why the Fréchet transformation uses ranks (ties get distinct ranks through `argsort`).
* **Processed files:** `X_losses.csv` (what every audit script reads) and `X_frechet.csv` (Fréchet(2) margins). Both have a date index and one column per industry.
* **Industry names (30):** Food, Beer, Smoke, Games, Books, Hshld, Clths, Hlth, Chems, Txtls, Cnstr, Steel, FabPr, ElcEq, Autos, Carry, Mines, Coal, Oil, Util, Telcm, Servs, BusEq, Paper, Trans, Whlsl, Rtail, Meals, Fin, Other.
* **External proxy (optional):** `make_proxy.py dgs10` or `wti` builds `data/raw/dgs10_proxy.csv` or `wti_proxy.csv` (columns `date, value ≥ 0`) from FRED series DGS10 (10-year Treasury yield) or DCOILWTICO (WTI crude); value = absolute daily change / loss. If the download fails, download the csv from FRED manually and put it in `data/raw/`. These files are not in the repository.

---

## 7. Core library (`python/rmlm/`)

The package keeps **three** implementations side by side on purpose.

### 7.1 `validated_lib.py` — the from-scratch reference
Written directly from the paper's formulas and unit-tested. Uses **one shared global radius** `R` and one set of extreme days for every max-projection. Orientation: arrays are `(n, d)`.

| Function | What it does |
|---|---|
| `simulate_rmlm(d, n, edge_density, alpha, seed)` | Random ground-truth max-linear model: random upper-triangular `C`, recursive `A`, simulated data. |
| `scaling(Mvals, R, k, dim_const)` | Empirical scaling estimator (formulas 28/30), generalised to any max-projection. |
| `build_M(X, cols, weight)` | Row-wise max over chosen columns (the projection `M_I`). |
| `combine(*parts)` | Row-wise max of several projections. |
| `algorithm1(X, a, eps, k)` | Causal order as a list of generations, sources first. |
| `algorithm2(X, order_groups, k)` | Estimate `A` given the order. |

### 7.2 `original_port.py` — faithful port of the authors' R code
Arrays have the **R orientation `(d, n)`**. Each function is a line-by-line port of `R/original/CJS_functions_code.R`:

| Python | R function | Purpose |
|---|---|---|
| `margins_to_frechet` | `margins_to_frechet` | Fréchet transform (formula 29). |
| `omega_A`, `_topk_cols` | `omega_A` | Angular components of the k most extreme observations. |
| `sigma_M_I` | `sigma_M_I` | Scaling of a max-projection over a set `I`. |
| `A_generations` | `A_generations` | Algorithm 2 given generations. |
| `mwp_gr_eps` | `mwp_gr_eps` | Hard thresholding (section 7.1 of the paper). |
| `omega_caus_order`, `sigma_M`, `pair_sigma_diff`, `sigma_i_aj_aI_diff` | same | Building blocks of Algorithm 1. |
| `causord_eps_a` | `causord_eps_a` | Algorithm 1 (causal order estimate). |

**Difference from `validated_lib`:** the authors' code uses a **local** radius per max-projection, ours a global one. On synthetic data the port shows more estimation variance (correlation with truth ≈ 0.86–0.88 vs ≈ 0.97–0.98 for the validated version). Both are kept: the port as a reference, the validated version as default.

### 7.3 `fast_lib.py` — the library behind all audits
Self-contained and optimised so that one full pipeline run takes ≈ 0.2 s at `d = 30, n = 2285`; this makes bootstrap and permutation tests feasible. Convention: `X` is `(n, d)`; `A[i, j] ≠ 0` means `j` is an ancestor of `i`; an edge is a tuple `(parent, child)`.

| Function | What it does |
|---|---|
| `A_from_C(C)`, `random_C(d, density, rng)` | Build a coefficient matrix `A` (unit rows) from a random weight matrix `C`. |
| `sample_from_A(A, n, rng, alpha)` | Simulate `n` days from a given `A` (Fréchet innovations). |
| `simulate_rmlm(d, n, edge_density, alpha, seed)` | Returns `(X, A, C)`. |
| `to_frechet(X, alpha)` | Rank-based Fréchet transform. |
| `top_idx(X, k)` | Indices and radii of the `k` most extreme days. The module variable `RADIUS_COLS` (default `None`) can restrict the radius to the first `m` columns (used by `dilution_real.py`). |
| `scaling(...)` | Scaling of a max-projection. |
| `algorithm1(X, a, eps, k)` | Causal order (generations, sources first). |
| `algorithm2(X, groups, k)` | `A` given the order (original index orientation). |
| `hard_threshold_dag(A, delta)` | Edge set `{(j, i)}` after thresholding. |
| `reach_edges(A)` | All positive off-diagonal entries (reachability graph). |
| `shd`, `nshd`, `prec_rec` | Graph distances and precision/recall. |
| `pipeline(X, a, eps, k_order, k_A, delta, standardise)` | Full pipeline: Fréchet → order → `A` → thresholded DAG. Returns `(edges, A, groups)`. |

Every audit script does `from rmlm.fast_lib import *`.

### 7.4 `data_prep.py`
`load_raw` (parse the first table of the French csv, missing codes → NaN), `to_losses(df, start, end)` (losses in the paper's window), `to_frechet(X, alpha)`. Run `python -m rmlm.data_prep` to write `data/processed/X_losses.csv` and `X_frechet.csv`.

### 7.5 `thresholding.py`
`hard_threshold_dag(A, delta)`, `shd`, `nshd`, and `centroid_graph(graphs)` (the graph with the smallest total nSHD to all others — the paper's way to choose among several `k`).

### 7.6 `run_pipeline.py`
Reproduces the paper's workflow: Fréchet margins → causal order with `k = 250` (estimated **once**) → for `k ∈ {90, 92, 94, 96, 98}` estimate `A`, threshold with `δ = 0.1` → choose the centroid graph → print edges and Kendall's τ against the paper's published order (`PAPER_ORDER`).

---

## 8. Scripts reference (`python/*.py`)

Common arguments for the real-data scripts: `--csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15`. Defaults for the estimator: `a = 1.3`, `ε = 0.1`, `δ = 0.1`, `k_order = 0.11 n`, `k_A = 0.04 n`.

### `compare_methods.py` — single run vs bootstrap vs rolling windows
* **Does:** (1) one pipeline run on all data; (2) `--boot` block-bootstrap resamples (block `--block 20`) and edge frequencies; (3) rolling windows (`--window-years 3 --step-years 1`).
* **Run:** `python compare_methods.py --csv ... --start ... --end ... --boot 100 --window-years 3 --step-years 1`
* **Writes** to `../results/tables`: `edges_single.csv` (parent, child), `edges_bootstrap.csv` (parent, child, freq), `rolling_windows.csv` (start, end, n_edges, nshd_to_previous, nshd_to_full_sample).
* **Look for:** how many single-run edges have `freq ≥ 0.7`.

### `boot_fixed_order.py` — order or coefficients?
* **Does:** two block bootstraps. FULL re-runs everything (as in the paper); FIXED keeps the causal order of the original data and re-estimates only `A` and the threshold.
* **Prints:** number of edges with frequency ≥ 0.5 / 0.7 / 0.9 for both, how often each node is in generation 0, mean Kendall τ of bootstrap orders vs the original order.
* **Real-data result (B = 100):** full `[32, 5, 1]`, fixed `[105, 57, 16]`, τ = 0.286, Mines in gen 0 in 90 %, Coal in 50 %.

### `coef_audit.py` — coefficients in every resample and every window
* **Does:** Part 1: for each of `--boot` bootstrap resamples store the coefficient matrix, the thresholding quantity `S_ij = a_ij − max_k a_ik a_kj / a_kk` and the selected edges. Part 2: for each rolling window run the pipeline plus `--win-boot` within-window bootstraps for the 8 key edges.
* **Run:** `python coef_audit.py --csv ... --start ... --end ... --boot 100 --win-boot 30 --topk 15`
* **Writes** to `../results/coef_audit/`: see [section 10](#10-results-folder-guide).
* **"Meaningful" edge (descriptive rule):** bootstrap frequency ≥ 0.7 **and** 2.5 % quantile of `a_ij` > 0.05. This is not a formal test, the thresholds are arbitrary, and the intervals come from a model the goodness-of-fit test rejects.

### `audit_extra.py` — two modes
* `split`: cut the series into blocks, compare odd vs even blocks (nSHD), repeated over several block lengths; compared with a bootstrap-filtered version (freq ≥ 0.7). Writes `split_half.csv`. Note: serial dependence means odd and even blocks are not fully independent.
* `proxy`: append an external column (csv `date,value`), compare the graph with and without it (edges removed, which nodes change generation). Add `--shuffle` for a time-shuffled control column. Writes `proxy_removed_edges.csv`.

### `null_proxy.py` — real proxy vs shuffled proxies
Compares the real proxy column against `--n` shuffled copies on: edges removed from the 30-column graph, edges with the proxy as endpoint, size of generation 0, whether Mines and Coal stay sources. Writes `null_proxy.csv`.

### `dilution_real.py` — why does one extra column wreck the graph?
Tests the *dilution hypothesis* (the extra column changes which days are selected as extreme because the radius uses all 31 columns). Four variants: shuffled proxy with radius from 31 columns (paper), shuffled proxy with radius from the original 30, iid Fréchet(2) column with 31, same with 30. **Not yet run on real data.**

### `gof_check.py` — tail-dependence goodness of fit
* **Does:** fit the RMLM, simulate from it, compare empirical pairwise upper-tail dependence coefficients `χ̂_ij(q)`, `q = 0.95`, between data and model. Statistic `T = mean_{i<j} |χ̂_ij^data − χ̂_ij^model|`. The reference distribution comes from `--reps` (default 30) datasets simulated from the fitted model, **refitting the whole pipeline** on each.
* **Run:** `python gof_check.py --csv ... --start ... --end ... --reps 30`
* **Real-data result:** `T = 0.188`, reference mean 0.093, 95th percentile 0.113 → rejected.
* **Scope:** the max-linear model and its additive counterpart share the same limiting angular measure (Remark 1 of the paper). `T` therefore tests "at most d atoms", not "max versus sum".

### `graph_audit.py` — window graphs, "which graph is the best centre?", directed vs undirected
* **Does:** (1) draws the graph of every rolling window (rows = generations, top = sources); (2) builds the candidate graphs — single run, the paper's rule (centroid = minimum total nSHD among the graphs for k = 90…98 with the order estimated once), bootstrap medoid, consensus graphs (frequency ≥ 0.5 / 0.7) and every window graph — and measures each candidate's mean nSHD to the cloud of bootstrap graphs (lower = closer to all others = smaller expected error); (3) separates *adjacency* (undirected skeleton) from *direction*.
* **Run:** `python graph_audit.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100`
* **Writes** to `../results/graph_audit/`: `fig_window_graphs.png`, `fig_candidate_distance.png`, `fig_skeleton_vs_directed.png`, `candidate_distances.csv`, `pair_freq.csv`, `window_pairwise_nshd.csv`, `summary.txt`.
* **Caution:** nSHD divides by |E1|+|E2|, so a *sparse* consensus graph is always "far" from dense graphs; the criterion favours dense graphs and should be read together with precision/recall-style criteria.

### `lag_check.py` — with what lag are two nodes related?
* **Does:** for the most frequent edges (parent j → child i) computes χ_h = P(U_i(t+h) > q | U_j(t) > q), q = 0.95, for h = −5…+5 (h > 0: parent leads), for the whole sample and for each rolling window, plus a lead–lag asymmetry. The RMLM is instantaneous, so this checks that assumption. Chance level is 1 − q = 0.05.
* **Run:** `python lag_check.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --edges ../results/tables/edges_bootstrap.csv --top 15`
* **Writes** to `../results/lag_check/`: `lag_table.csv`, `lag_by_window.csv`, `fig_lag.png`, `summary.txt`.

### `garch_filter.py` — remove volatility clustering, then re-run the audits
* **Does:** fits a Gaussian-QMLE GARCH(1,1) (own implementation, only numpy/scipy) to each industry's returns in the window, takes standardised residuals z and defines filtered losses `max(−z, 0)`. Writes a csv that can be passed to any script with `--csv`.
* **Run:** `python garch_filter.py --raw ../data/raw/30_Industry_Portfolios_Daily.csv --start 1989-06-01 --end 1998-06-15 --out ../data/processed/X_losses_garch.csv`, then e.g. `python boot_fixed_order.py --csv ../data/processed/X_losses_garch.csv ...` and `python gof_check.py --csv ../data/processed/X_losses_garch.csv ...`.
* **Result:** the filter removes the clustering (lag-1 autocorrelation of squares 0.136 → 0.021; mean α+β ≈ 0.94) but the instability and the rejection of the model remain (see section 2).

### `perm_edges.py` — permutation test per edge
H0 for `j→i`: "column j has no extremal dependence on the rest". Null: circularly shift column `j` in time (keeps its margin and autocorrelation, breaks cross-dependence), keep the causal order fixed, re-estimate `A`, record `A[i, j]`. Writes `edges_permutation.csv` (`parent, child, a_hat, p, q_BH, p_bonf, sig_BH`). With 200 permutations, Bonferroni is practically impossible (smallest p is `1/(P+1)`).

### `edge_test.py` — conditional edge test
H0: `j→i` is **not** in the minimum DAG, i.e. `a_ij` equals its best indirect path (`S_ij = 0`). Null model `A0` = `A_hat` with `a_ij` replaced by the indirect-path value, rows renormalised; simulate from `A0`, keep the order fixed, re-estimate, record `S_ij`; BH correction. Writes `edges_conditional_test.csv`. Candidates and `A_hat` come from the same data, so p-values are optimistic (post-selection).

### `baseline_order.py` — independent ordering baseline
Causal tail coefficient (Gnecco, Meinshausen, Peters, Engelke 2021): `Γ_ij` = mean over the `k` days where `X_i` is largest of the rank-uniform value of `X_j`. Node score = `mean_j (Γ_ij − Γ_ji)`; sources score highest. Prints top/bottom nodes and Kendall τ vs Algorithm 1. **The definition was implemented from memory and should be verified against arXiv:1908.05097.**

### `event_study.py` — one extreme day
For each event date (default: the three days with the largest radius): (a) *influence* — nSHD between the full-data graph and the graph with ±10 days around the event removed, versus the same with 10 days removed at random places (50 draws); (b) *before/after* — nSHD between the graphs before and after the event versus random split points. Options: `--dates D1 D2 …` (explicit event dates), `--h` (days removed on each side, default 10), `--null` (random draws, default 50). Writes `event_study.csv`. Result: effects are not clearly separable from ordinary instability (p = 0.08–0.18, uncorrected).

### `make_proxy.py` — builds the proxy csv
`python make_proxy.py dgs10` or `wti`. Tries FRED, falls back to a local `../data/raw/DGS10.csv` / `WTI.csv`.

### `ctrl30.py` — simulated control run
Simulates three `d = 30` max-linear datasets with known truth and runs the pipeline and bootstrap, to compare with real-data behaviour. (See the import issue in section 14.)

### `experiments_all.py` — simulation experiments
Run `python experiments_all.py [E1 E2 E4 E5 E8]` (default: all). Writes `results.json` and `fig_*.png` in the working directory.

| Experiment | Question |
|---|---|
| E1 | Why does `a² − 1` appear, and how should `a` be chosen? |
| E2 | Why do max-projections matter for extremes (max vs sum)? |
| E8 | Single run vs bootstrap vs rolling window on simulated data with known truth. |
| E4 | Effect of a hidden confounder (violating causal sufficiency). |
| E5 | Goodness-of-fit statistic on max-linear, additive-linear and multivariate-t data. Also defines `chi_matrix` and `gof_stat` used by `gof_check.py`. |

---

## 9. The audit methodology (components A–E)

**A. Order versus coefficient decomposition** (`boot_fixed_order.py`, `coef_audit.py`). Edge frequencies

```
π_full_ij = (1/B) Σ_b 1{ (j→i) ∈ Φ(X*_b) }                 order re-estimated in every resample
π_fix_ij  = (1/B) Σ_b 1{ (j→i) ∈ Ψ(X*_b ; Ô) }             order Ô of the original data held fixed
```

with `Φ` the full pipeline and `Ψ` Algorithm 2 plus thresholding given the order. Also: the instability index `ρ_τ`, the mean Kendall τ of generation ranks, and the generation-0 frequency of every node.

**B. Fit audit** (`gof_check.py`). Statistic `T(X)` above, reference distribution by parametric bootstrap with refitting. Reject at 5 % if `T` exceeds the 95th reference percentile.

**C. Edge-level inference** (`perm_edges.py`, `edge_test.py`). Two nulls (time-shift permutation; conditional boundary model), p-values `(1 + #{null ≥ obs}) / (P + 1)`, Benjamini–Hochberg q-values. **Caveat:** on this data both tests mark almost all edges significant; the null does not fit the data, so this is a negative result about the validity of such tests under misspecification, not evidence for the edges.

**D. Added-variable sensitivity** (`audit_extra.py proxy`, `null_proxy.py`, `dilution_real.py`). Compare a real external column with 30 time-shuffled copies on: edges removed, edges incident to the new column, whether it is the only source, whether Mines and Coal remain sources.

**E. Independent order baseline** (`baseline_order.py`). Causal tail coefficient; compared with Algorithm 1 by Kendall τ and, in simulation, by the fraction of ancestor–descendant pairs ordered correctly (0.94 for Algorithm 1 vs 0.98 for the tail-coefficient order on simulated max-linear data).

**Evaluation protocol in simulation.** `d = 30`, `n = 2285`, `C_ii ~ U(0.5, 1.3)`, edge `k→i` with probability *density* and weight `C_ik ~ U(0.3, 1.4)`, `A` from the recursion with unit rows, Fréchet(2) innovations. Truth = the minimum DAG of `A`. Misspecified alternative: one-factor multivariate t (3 d.f.). Metrics: precision, recall, F1, SHD, nSHD.

---

## 10. Results folder guide

### `results/tables/`
| File | Produced by | Columns / content |
|---|---|---|
| `edges_single.csv` | `compare_methods.py` | `parent, child` — the 142 single-run edges. |
| `edges_bootstrap.csv` | `compare_methods.py` | `parent, child, freq` — bootstrap frequency of each edge. |
| `rolling_windows.csv` | `compare_methods.py` | `start, end, n_edges, nshd_to_previous, nshd_to_full_sample` per window. |
| `split_half.csv` | `audit_extra.py split` | per block length: edge counts and nSHD raw vs bootstrap-filtered. |
| `proxy_removed_edges.csv` | `audit_extra.py proxy` | edges (`A->B`) removed after adding the proxy. |
| `null_proxy.csv` | `null_proxy.py` | one row per run: `removed, n_proxy_edges, proxy_out, gen0_size, proxy_in_gen0, mines_coal_still_sources`. |
| `edges_permutation.csv` | `perm_edges.py` | `parent, child, a_hat, p, q_BH, p_bonf, sig_BH`. |
| `edges_conditional_test.csv` | `edge_test.py` | `parent, child, a_hat, S_ij, p, q_BH, sig_BH`. |
| `event_study.csv` | `event_study.py` | `date, radius, nshd_remove_event, null_remove_mean, null_remove_95, p_remove, nshd_before_after, null_split_mean`. |

### `results/coef_audit/` (from `coef_audit.py`)
| File | Content |
|---|---|
| `boot_edge_coef.csv` | Per edge: `parent, child, in_full_sample_graph, full_sample_a, boot_freq, a_mean, a_sd, a_q025, a_q975, S_mean, S_q025, meaningful`. |
| `fig_boot_coef_box.png` | Box plot of `a_ij` over the bootstrap resamples for the top-15 edges (green = meaningful, orange = not; black tick = full-sample value; right: frequency). |
| `fig_boot_coef_trace.png` | `a_ij` in each resample b = 1…B for the six most frequent edges. |
| `fig_boot_heatmaps.png` | Three 30 × 30 maps: mean `a_ij`, sd of `a_ij`, edge frequency. Rows = child, columns = parent, ordered by generation. |
| `rolling_features.csv` | Per window: `n_edges, n_generations, sources, strongest_edges, mean_coef_of_edges, nshd_to_previous, nshd_to_full_sample, kendall_tau_order_vs_full, edges_stable_in_window_boot_ge07`. |
| `rolling_edge_coef.csv` | Per window and key edge: `a_window, boot_q025, boot_q975, boot_freq, edge_in_window_graph`. |
| `fig_roll_coef.png` | Coefficient of the eight key edges in every window with within-window bootstrap 95 % intervals (dashed = full sample). |
| `fig_roll_generation.png` | Generation of every industry in every window (0 = source). |
| `fig_roll_summary.png` | Edges per window, nSHD, Kendall τ over time. |

**How the within-window bootstrap works.** For each window, the pipeline is run on the window's ≈ 756 days; then `--win-boot` (default 30) moving-block resamples of that window are drawn and the pipeline is re-run on each. The bars in `fig_roll_coef.png` are the 2.5 % and 97.5 % quantiles of the key edge's coefficient over those resamples. With only 30 resamples the quantiles are noisy.

---

## 11. How to reproduce everything, in order

From `python/` (≈ minutes each on a laptop):

```bash
python -m rmlm.data_prep
pytest tests/ -v
python run_pipeline.py
python compare_methods.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 200 --window-years 3 --step-years 1
python boot_fixed_order.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100
python coef_audit.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100 --win-boot 30 --topk 15
python audit_extra.py split --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --reps 10 --boot 50
python gof_check.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --reps 30
python perm_edges.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --perms 200
python edge_test.py  --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15
python baseline_order.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15
python event_study.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15
python graph_audit.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100
python lag_check.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --edges ../results/tables/edges_bootstrap.csv --top 15
python garch_filter.py --raw ../data/raw/30_Industry_Portfolios_Daily.csv --start 1989-06-01 --end 1998-06-15 --out ../data/processed/X_losses_garch.csv
python make_proxy.py dgs10
python audit_extra.py proxy --csv ../data/processed/X_losses.csv --proxy ../data/raw/dgs10_proxy.csv --start 1989-06-01 --end 1998-06-15
python null_proxy.py --csv ../data/processed/X_losses.csv --proxy ../data/raw/dgs10_proxy.csv --start 1989-06-01 --end 1998-06-15 --n 30
python experiments_all.py            # simulations; writes results.json and fig_*.png
```

Check each script's `--help`/docstring for options; bootstrap counts can be lowered for a quick test. Different seeds or `B` change edge counts slightly (e.g. 3 vs 5 edges at frequency ≥ 0.7).

---

## 12. Parameters and defaults

| Parameter | Default | Meaning |
|---|---|---|
| `a` | 1.3 | Scaling factor in Algorithm 1 (must be > 1). |
| `eps` (ε) | 0.1 | Tolerance defining a generation. |
| `delta` (δ) | 0.1 | Thresholding margin for edges. |
| `k_order` | 0.11 · n (≈ 250) | Number of extreme days in Algorithm 1. |
| `k_A` | 0.04 · n (≈ 90) | Number of extreme days in Algorithm 2. |
| block length | 20 days | Moving-block bootstrap. |
| `--boot` | 100 | Bootstrap resamples (main). |
| `--win-boot` | 30 | Bootstrap resamples inside each rolling window. |
| window / step | 3 years (756 d) / 1 year (252 d) | Rolling windows. |
| `q` | 0.95 | Level for tail-dependence coefficients in the fit test. |
| tail index α | 2 (fixed) | Fréchet shape; never estimated. |

---

## 13. Tests

`pytest tests/ -v` — **7 tests, all passing** (checked on Python 3.13).

* `test_validated_lib.py`: a single-node scaling equals 1; Algorithm 1 returns a valid topological order; Algorithm 2 recovers a synthetic `A` reasonably.
* `test_original_port.py`: Fréchet transform is positive; causal order is a valid topological order; `A_generations` is accurate to a loose tolerance; `mwp_gr_eps` removes redundant edges.

The audit scripts (`coef_audit.py`, `boot_fixed_order.py`, …) have **no unit tests**; they were checked on simulated data and by running them on the real data.

---

## 14. Known issues, caveats and limitations

**Code issues found while documenting (not yet fixed in the repository):**
* *(fixed)* `ctrl30.py` imported a non-existent `rmlm_lib`; it now imports `rmlm.fast_lib`.
* *(fixed)* `scipy` was missing from `requirements.txt` and `environment.yml`; added.
* *(fixed)* `data_prep.py` pointed to `...Daily.CSV` while the file is `.csv` (failed on case-sensitive systems); now `.csv`.
* Several files contain compiled `__pycache__` folders in git; `.gitignore` already lists them, so they can be removed with `git rm -r --cached`.
* `.gitignore` ignores `data/processed/*.csv`, yet `X_losses.csv` and `X_frechet.csv` are in the repository.
* `docs/project_structure_and_testing_fa.md` refers to other docs (`01_…`, `02_…`) that are not in the repository.

**Scientific limitations:**
* One 9-year period; no second period yet.
* Simulation studies use few replications (3–12) and a single family of random graphs.
* One misspecified alternative (multivariate t) was compared; serial dependence, volatility clustering and hidden factors are not ruled out as causes of the instability.
* The goodness-of-fit reference distribution uses only 30 replications, and its generating fit (ε = 0.1, k = 0.11 n) differs slightly from the fit inside the statistic.
* Edge-test p-values are post-selection (optimistic); the permutation/conditional nulls rely on a model the fit test rejects.
* Block length 20 is a fixed choice; sensitivity to block length, window length, `a`, `ε`, `k`, `δ` is not yet studied.
* The tail index is assumed to be 2 for every industry; it is not estimated or tested.
* No expanding-window or out-of-sample analysis.
* The GARCH filter is univariate Gaussian GARCH(1,1) fitted once on the whole window (not refitted inside each bootstrap resample); GARCH-t, GJR/EGARCH and a common-volatility-factor filter were not tried.
* The "dilution" explanation for the added-variable effect is a hypothesis; `dilution_real.py` has not been run on the real data.
* The causal tail coefficient was implemented from memory of Gnecco et al. (2021) and must be verified.
* The within-window bootstrap uses only 30 resamples per window.

**Planned work:** second period (e.g. 2003–2007), dilution test, goodness-of-fit with ≥ 200 reference replications and separation of misfit causes (GARCH-filtered margins, t-copula and factor alternatives), sample-size growth analysis, an order-stabilised estimator (consensus of bootstrap generations), a fixed-order version of `coef_audit.py`, resampling-based nulls for edge tests, regime analysis (calm vs crisis days).

---

## 15. FAQ

**Is there a train/test split?** No. The method estimates a structure; it does not predict. Stability is assessed with block bootstrap, split-half, and rolling windows. An out-of-sample consistency check is planned.

**Why are so many coefficients exactly 0 in the bootstrap?** Algorithm 2 sets negative squared coefficients to 0, and a coefficient is also 0 when, in that resample, the two nodes are placed in the same generation or in the opposite order.

**Why do the permutation tests mark nearly every edge significant, while the bootstrap finds almost none stable?** They answer different questions. The permutation null ("j has no extremal dependence on the rest") is easily rejected in financial data; the bootstrap asks whether the same edge, with the same direction and weight, reappears in new samples.

**Why are there two implementations (`validated_lib` and `original_port`)?** The port makes sure our results do not depend on a misreading of the paper's R code; the validated library is the more accurate and tested default; `fast_lib` is the speed-optimised version for resampling.

**Can I use another dataset?** Yes: provide a csv of non-negative losses (date index, one column per node) and pass it with `--csv`. The scripts do not depend on the industry names, except `run_pipeline.py` (published order) and the Mines/Coal checks in the proxy scripts.

---

## 16. References

1. Klüppelberg, C. and Krali, M. (2025). Causal analysis of extreme risk in a network of industry portfolios. arXiv:2504.00523.
2. Klüppelberg, C. and Krali, M. (2021). Estimating an extreme Bayesian network via scalings. *J. Multivariate Anal.* 181.
3. Gissibl, N. and Klüppelberg, C. (2018). Max-linear models on directed acyclic graphs. *Bernoulli* 24(4A).
4. Tran, N. M., Buck, J. and Klüppelberg, C. (2024). Estimating a directed tree for extremes. *J. R. Stat. Soc. B* 86(3).
5. Krali, M., Davison, A. C. and Klüppelberg, C. (2023). Heavy-tailed max-linear structural equation models in networks with hidden nodes. arXiv:2306.15356.
6. Gnecco, N., Meinshausen, N., Peters, J. and Engelke, S. (2021). Causal discovery in heavy-tailed models. *Ann. Statist.*; arXiv:1908.05097.
7. Runge, J. et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. *Sci. Adv.* 5(11).
8. Saggioro, E. et al. (2020). Reconstructing regime-dependent causal relationships from observational time series. *Chaos*; arXiv:2007.00267.
9. Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Ann. Statist.* 17.
10. Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. *J. R. Stat. Soc. B* 57.
11. Resnick, S. I. (2007). *Heavy-Tail Phenomena*. Springer.

*Items 1, 5, 6 and 8 were located on arXiv; the other bibliographic details should be checked against the sources before citing.*

The authors' original R code is by Mario Krali (https://github.com/mariokrali) and is included unmodified under `R/original/`. No licence file has been added to this repository yet; add one before sharing the code publicly.
