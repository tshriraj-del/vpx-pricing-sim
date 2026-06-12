# VPX — Vehicle Pricing Simulator & Optimizer

A pricing-strategy tool for vehicle OEMs: configure pricing / incentive /
finance / inventory levers across nameplates and regions, see projected
business outcomes in real time, and have an optimizer recommend the
profit-maximizing plan under business constraints.

Built as a teaching-grade but economically rigorous prototype. **Pure Python
standard library** — no numpy, no framework, no build step.

> Honest scope: every number here comes from **synthetic data with assumed
> coefficients**. The economics are correct and internally consistent, but the
> magnitudes are not calibrated to any real market. Use it to *reason*, not to
> set real prices, until calibrated (see `vpx_calibrate.py`).

## What's in the box

| File | Role |
|------|------|
| `vpx_sim.py` | The engine + CLI. v2 economics (two-price split, nested-logit demand, leasing/residuals, FX), the optimizer, and 15 invariant self-tests. |
| `vpx_web.py` | Engine ↔ JSON glue + scenario endpoints, shared by the local server and the serverless functions. |
| `vpx_app.py` | Local dev server (`http.server`). Serves the SPA + the API on `localhost`. |
| `vpx_store.py` | Append-only scenario store on **Turso** (libSQL) over its HTTP API. |
| `vpx_calibrate.py` | Calibration + backtest harness. Fits the model to data and reports out-of-sample accuracy. |
| `public/index.html` | The single-page UI (vanilla JS, CSS/SVG charts). |
| `api/*.py` | Vercel serverless functions wrapping `vpx_web`. |

## Run locally

```bash
python3 vpx_app.py            # → http://127.0.0.1:8765
```

No dependencies. Scenarios persist in the browser (`localStorage`) unless Turso
env vars are set (below), in which case they persist server-side.

CLI tools:

```bash
python3 vpx_sim.py optimize       # the optimizer, in the terminal
python3 vpx_sim.py --selftest     # 15 economic invariants
python3 vpx_calibrate.py          # parameter-recovery + backtest demo
python3 vpx_calibrate.py --make-template   # CSV schema for real data
```

## Deploy on Vercel

The app is structured for Vercel's serverless model: `public/` is served
statically, and the compute lives in stateless Python functions under `api/`.

1. Import the repo in Vercel (no build command needed — it's static + Python
   functions).
2. **For server-side scenarios, add a [Turso](https://turso.tech) database** and
   set these Environment Variables in the Vercel project:

   ```
   TURSO_DATABASE_URL = libsql://<your-db>-<org>.turso.io
   TURSO_AUTH_TOKEN   = <turso db tokens create ...>
   ```

   The `scenarios` table is created automatically on first use (append-only:
   every save/rename/delete is a new versioned row; nothing is overwritten or
   hard-deleted).

   Without these vars the demo still works — it falls back to per-browser
   `localStorage`.

## Calibration — turning guesses into a fitted model

`vpx_calibrate.py` is the bridge to real use. Its default run is a
**parameter-recovery test**: it generates history from a *hidden* coefficient
set, fits the model without seeing them, and checks it (a) recovers them and
(b) predicts months it never saw — proving the fitting machinery is correct
before you trust it on real data. Then:

```bash
python3 vpx_calibrate.py --data your_history.csv
```

fits the same model to a real panel. See `--make-template` for the schema. The
one thing that matters most: to identify **price elasticity** you need real
**price/incentive variation** — open aggregate sales data alone under-identifies
it.

## License

Prototype / demonstration code.
