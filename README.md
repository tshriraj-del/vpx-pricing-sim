# VPX: Vehicle Pricing Simulator and Optimizer

### [▶ Live demo: vpx-pricing-sim.onrender.com](https://vpx-pricing-sim.onrender.com/)

A pricing-strategy tool for vehicle OEMs. Configure pricing, incentive,
finance and inventory levers across nameplates and regions, watch projected
business outcomes update in real time, and let a constrained optimizer
recommend the **profit-maximizing plan**. Save scenarios and compare them side
by side.

**It is a single self-contained `index.html`.** The whole engine (simulation
plus optimizer) runs client-side in vanilla JavaScript. No backend, no build
step, no dependencies. It deploys as a static site anywhere, and it even works
by opening the file directly.

> **Honest scope.** Every number comes from a **synthetic dataset with assumed
> coefficients**. The economics are internally consistent and the methods are
> real, but the magnitudes are *not* calibrated to any live market. Use it to
> reason about pricing, not to set real prices, until it is fit to data via the
> calibration harness in `python/`.

---

## Tech stack

| Layer | Choice | Notes |
|-------|--------|-------|
| **App** | Single static `index.html` | Vanilla JS, hand-rolled CSS, inline SVG charts. No framework, no build. |
| **Engine** | Client-side JS port of the Python reference | Deterministic, runs in the browser. Verified to reproduce the Python numbers exactly. |
| **Demand model** | Nested multinomial **logit** (discrete choice) | Category size times within-segment softmax. |
| **Optimizer** | **Hooke-Jeeves** pattern search, feasibility-first | Constrained, multi-start. About 870 evaluations in under 100 ms. |
| **Scenarios** | Browser `localStorage` | Per-browser save and compare. No database, no accounts. |
| **Reference + calibration** | Python 3.9+ **stdlib only** (`python/`) | The validated source of truth plus the estimation and backtest harness. |
| **Deploy** | Any static host (Render, Netlify, GitHub Pages) | Zero config, no build command, no environment variables. |

Why client-side? The simulation and optimization are pure, fast functions, so
there is nothing a server needs to do. Running them in the browser makes the
demo trivially deployable and dependency-free. The Python in `python/` remains
the **reference implementation** (with a 15-invariant self-test) that the JS
engine is validated against, plus the calibration harness for fitting to real
data.

---

## Modeling and methods

This is a **structural choice model**, not a black-box predictor. Every
coefficient is interpretable and identifiable.

### Two prices, never one

The OEM's realized price and the buyer's perceived price are tracked
separately, because conflating them mis-states revenue:

```
oem_net_price   = list_price - OEM-funded incentives        -> drives REVENUE & MARGIN
consumer_price  = oem_net_price - EV credit - state rebate  -> drives DEMAND only
```

Government EV credits and state rebates lower what the *buyer* perceives
(lifting demand via the choice model) but do **not** reduce OEM revenue.

### Demand: nested multinomial logit

**Stage 1, category size.** A constant-elasticity shifter with seasonality and
macro:

```
segment_volume = base_size[seg,region] x seasonal x macro
               x (segment_avg_price / reference_price) ^ segment_elasticity
```

**Stage 2, within-segment share.** A softmax over our nameplates *and*
competitors:

```
U_j     = a_j - B_price*consumer_price_j + B_apr*(market_rate - apr_j) + B_ev*ev_infra_j
share_j = exp(U_j) / sum_k exp(U_k)
units_j = segment_volume x share_j
```

Why a logit rather than ad-hoc elasticity formulas: volume and share are **one
model** (they cannot disagree), share is **bounded** in (0,1) so there is no
"lower price implies infinite volume", **cross-elasticity is automatic** via the
shared denominator, and it is the **industry-standard** approach to
differentiated-product demand. Other modeled effects: leasing and
residual-value risk, finance subvention, continuous inventory pressure, and FX
translation.

### Optimizer: constrained, derivative-free

```
maximize    sum of contribution_margin
over        price_delta[v] in [-30%, +30%],  apr_subvention[v] in [0, market_rate]
subject to  oem_net >= cogs;  units <= capacity;  market_share >= floor;  sum subsidy <= budget
method      Hooke-Jeeves pattern search, feasibility-first comparator, 3-point multi-start
```

This is tractable precisely because the logit demand is smooth and bounded. It
reports the recommended plan, the contribution uplift, and **which constraint
binds**.

### Calibration: estimation with validation (`python/vpx_calibrate.py`)

- **Logit coefficients** by **maximum likelihood** (Adam gradient ascent on the
  grouped multinomial log-likelihood, closed-form gradient `sum (n_j - N*P_j)*x_j`).
- **Category elasticity** by **two-way fixed-effects** regression of
  `log(volume)` on `log(price)` via iterative demeaning.
- **Validation** by a **parameter-recovery test**: fit a *hidden* coefficient
  set blind, then confirm recovery plus an out-of-sample backtest (MAPE, R²), so
  the estimation code is proven correct before it is trusted on real data.

### Why not deep learning or gradient-boosted trees?

A black box trained on synthetic data generated from its own assumed elasticity
just **recovers the generating function**, which proves nothing. A calibrated
structural choice model is interpretable, identifiable and validatable, and it
is what real OEM pricing science uses.

> **Deep dive:** what we would actually use with real data, a BLP
> random-coefficients logit with instruments for price, and where ML
> legitimately fits (DML, hierarchical Bayes, forecasting the nuisance layer),
> is in [`docs/demand-modeling-with-real-data.md`](docs/demand-modeling-with-real-data.md).

---

## Repository layout

```
index.html                  The whole app, UI plus JS engine (deploy this)
python/vpx_sim.py           Reference engine, CLI, 15 invariant self-tests
python/vpx_calibrate.py     Calibration and backtest harness
docs/                       Deep-dive notes (demand modeling with real data)
render.yaml                 Render static-site config
```

---

## Run it

**The app.** Open `index.html` in a browser, or serve the folder:

```bash
python3 -m http.server 8765      # http://127.0.0.1:8765
```

**The Python reference and calibration:**

```bash
cd python
python3 vpx_sim.py optimize                   # the optimizer, in the terminal
python3 vpx_sim.py --selftest                 # 15 economic invariants
python3 vpx_calibrate.py                      # parameter-recovery + backtest demo
python3 vpx_calibrate.py --make-template      # CSV schema for real data
python3 vpx_calibrate.py --data history.csv   # calibrate on a real panel
```

---

## Deploy

It is a static site. Import the repo into **Render** (or Netlify, GitHub Pages,
Cloudflare Pages):

- **Service type:** Static Site, not Web Service
- **Build command:** leave blank
- **Publish directory:** `.`
- **Environment variables:** none

No serverless functions, no runtime, no database. Saved scenarios live in each
visitor's browser via `localStorage`.

> Looking for the server-backed version with a Python API and shared scenario
> storage? That is the companion
> [`vpx-pricing-tool`](https://github.com/tshriraj-del/vpx-pricing-tool) repo.

---

## License

Prototype and demonstration code.
