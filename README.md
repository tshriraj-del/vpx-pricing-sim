# VPX — Vehicle Pricing Simulator & Optimizer

A pricing-strategy tool for vehicle OEMs. Configure pricing / incentive /
finance / inventory levers across nameplates and regions, watch projected
business outcomes update in real time, and let a constrained optimizer
recommend the **profit-maximizing plan**. Save scenarios and compare them
side by side.

Built as an economically rigorous prototype with **zero third-party
dependencies** — the engine, the optimizer, the estimation/calibration layer,
and the web server are all pure Python standard library; the UI is vanilla JS
with CSS/SVG charts.

> **Honest scope.** Every number here comes from a **synthetic dataset with
> assumed coefficients**. The economics are internally consistent and the
> methods are real, but the magnitudes are *not* calibrated to any live market.
> Use it to reason about pricing, not to set real prices — until fit to data
> via the calibration harness (below).

---

## Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| **Engine** | Python 3.9+ **stdlib only** | Vectorless, pure functions. Full 6-vehicle × 5-region matrix solves in <50 ms. |
| **Demand model** | Nested multinomial **logit** (discrete choice) | Category size × within-segment softmax. |
| **Optimizer** | **Hooke–Jeeves** pattern search (derivative-free), feasibility-first | Constrained, multi-start. No SciPy. |
| **Calibration** | **Maximum likelihood** (Adam) + two-way **fixed-effects** regression | Hand-rolled in stdlib; no numpy. |
| **Web layer** | `http.server` (local) / **Vercel** Python serverless functions (prod) | Shared glue in `vpx_web.py`. |
| **Frontend** | Vanilla **JS** + Tailwind-ish CSS + inline **SVG** charts | Single static `public/index.html`, no build step. |
| **Storage** | **Turso** (libSQL) over its HTTP API, optional | Append-only, versioned. Falls back to browser `localStorage` if unconfigured. |
| **Deploy** | Vercel (static `public/` + `api/*.py`) | No build command, no env vars required. |

No `requirements.txt` dependencies — the file exists only so Vercel detects the
Python runtime.

---

## Modeling & methods ("the ML details")

This is a **structural choice model**, not a black-box predictor — every
coefficient is interpretable and identifiable. The pieces:

### 1. Two prices, never one
The OEM's realized price and the buyer's perceived price are tracked
separately, because conflating them mis-states revenue:

```
oem_net_price   = list_price − OEM-funded incentives        → drives REVENUE & MARGIN
consumer_price  = oem_net_price − EV credit − state rebate   → drives DEMAND only
```

Government EV credits and state rebates lower what the *buyer* perceives (so
they lift demand through the choice model) but do **not** reduce OEM revenue.

### 2. Demand — nested multinomial logit
**Stage 1 — category size** (how big the segment is), a constant-elasticity
shifter with seasonality and macro:

```
segment_volume = base_size[seg,region]
               × seasonal[month] × macro_adjustment(cpi, rate, fuel, …)
               × (segment_avg_price / reference_price) ^ segment_elasticity
```

**Stage 2 — within-segment share**, a softmax over our nameplates *and*
competitors:

```
U_j      = α_j − β_price·consumer_price_j
                + β_apr·(market_rate − apr_j)
                + β_ev·ev_infra_j
share_j  = exp(U_j) / Σ_k exp(U_k)
units_j  = segment_volume × share_j
```

Why a logit rather than ad-hoc elasticity formulas:

- **Volume and share are one model** — `share × segment_size = units` by
  construction, so they can never disagree.
- **Bounded** — `share ∈ (0,1)`, so price cuts have diminishing returns (no
  "lower price → infinite volume" degeneracy).
- **Cross-elasticity is free** — dropping one nameplate's price pulls share
  from its siblings *and* competitors through the shared denominator.
- It's the **industry-standard** approach to differentiated-product demand
  (discrete choice / BLP-style).

Other modeled effects: leasing & residual-value risk, finance subvention
(labeled approximation), continuous inventory pressure, and FX as a
consolidated-USD translation layer.

### 3. Optimizer — constrained, derivative-free
The inverse problem: instead of hand-tuning sliders, solve for the lever
vector that maximizes contribution margin:

```
maximize    Σ contribution_margin
over        price_delta[v]      ∈ [−30%, +30%]
            apr_subvention[v]   ∈ [0, market_rate]
subject to  oem_net ≥ cogs            (never below cost)
            units   ≤ capacity        (factory ceiling)
            market_share ≥ floor      (brand-presence guardrail)
            Σ finance_subsidy ≤ budget
method      Hooke–Jeeves pattern search, feasibility-first comparator,
            3-point multi-start  (≈ 870 evals, <0.1 s, no SciPy)
```

Tractable precisely because the logit demand is smooth and bounded. The output
reports the recommended plan, the contribution uplift, and **which constraint
binds** (e.g. "share floor binds; capacity slack").

### 4. Calibration — estimation, with validation
`vpx_calibrate.py` turns assumed coefficients into *fitted* ones:

- **Logit coefficients** (`β_price`, `β_apr`, `β_ev`, per-nameplate `α`):
  maximum-likelihood estimation via **Adam gradient ascent** on the grouped
  multinomial log-likelihood. The gradient is closed-form:
  `∂LL/∂θ = Σ (n_j − N·P_j) · x_j`.
- **Category elasticity**: a **two-way fixed-effects** regression of
  `log(volume)` on `log(price)` (region×segment and month×segment effects,
  via iterative demeaning). Promo-driven price variation identifies the slope;
  the fixed effects absorb seasonality and macro shocks.
- **Validation**: the default run is a **parameter-recovery test** — generate
  history from a *hidden* coefficient set, fit it blind, and confirm the
  estimator (a) recovers the truth and (b) predicts months it never saw
  (out-of-sample **MAPE** and **R²**). This proves the estimation code is
  correct *before* it's trusted on real data.

### 5. Why not deep learning / gradient-boosted trees?
A black-box model trained on synthetic data generated from its own assumed
elasticity just **recovers the generating function** — it looks impressive and
proves nothing (and SHAP on it is decoration). A calibrated structural choice
model is interpretable, identifiable, validatable against held-out actuals, and
is what real OEM pricing science uses. When real history with genuine
price/incentive **variation** is available, the same harness fits it — see the
identification note in `--make-template`.

---

## Repository layout

```
vpx_sim.py          Engine + CLI: v2 economics, optimizer, 15 invariant self-tests
vpx_web.py          Engine ↔ JSON glue + scenario endpoints (shared local/serverless)
vpx_app.py          Local dev server (http.server)
vpx_store.py        Append-only scenario store on Turso (libSQL HTTP API)
vpx_calibrate.py    Calibration + backtest harness
public/index.html   Single-page UI (vanilla JS, CSS/SVG charts)
api/*.py            Vercel serverless functions wrapping vpx_web
vercel.json         Serverless config (bundles vpx_*.py into the functions)
```

---

## Run locally

```bash
python3 vpx_app.py                 # → http://127.0.0.1:8765   (no dependencies)
```

CLI tools:

```bash
python3 vpx_sim.py optimize        # the optimizer, in the terminal
python3 vpx_sim.py --selftest      # 15 economic invariants
python3 vpx_calibrate.py           # parameter-recovery + backtest demo
python3 vpx_calibrate.py --make-template      # CSV schema for real data
python3 vpx_calibrate.py --data history.csv   # calibrate on a real panel
```

---

## Deploy on Vercel

1. Import the repo at **vercel.com/new** — no build command, no framework
   preset (it's static `public/` + Python `api/` functions). Deploy.
2. **That's it.** It runs with **zero environment variables**; saved scenarios
   persist per-browser via `localStorage`.

Optional — **shared, server-side scenarios** via Turso:

```bash
turso db create vpx
turso db show vpx --url        # → libsql://vpx-<org>.turso.io
turso db tokens create vpx
```
Set in Vercel → Settings → Environment Variables, then redeploy:
```
TURSO_DATABASE_URL = libsql://vpx-<org>.turso.io
TURSO_AUTH_TOKEN   = <token>
```
The `scenarios` table auto-creates on first save (append-only: every
save/rename/delete is a new versioned row — nothing is overwritten or
hard-deleted). The frontend switches to server mode automatically.

---

## License

Prototype / demonstration code.
