#!/usr/bin/env python3
"""
VPX v2 — Calibration & backtest harness
=======================================

Turns the simulator from "well-reasoned guesses" into "a model fit to data."
Pure standard library (no numpy/scipy) — it implements the econometrics by hand.

Two jobs:

  1. PROVE THE MACHINERY (default run).
     Generate 36 months of synthetic history from a *hidden* "true" set of
     coefficients — WITH realistic promo-driven price variation, seasonality,
     a macro cycle, and noise. The fitter is NOT told the true values. We then
     check it (a) RECOVERS them and (b) PREDICTS held-out months it never saw.
     This is a parameter-recovery test: it validates the fitting code itself,
     so that when a fit to REAL data looks bad you know it's the *model*, not a
     bug. (And it avoids the v1 trap of "ML trained on its own assumptions".)

  2. RUN ON REAL DATA.
     `python3 vpx_calibrate.py --data history.csv`  fits the same model to a
     real panel and reports the same backtest metrics. `--make-template` prints
     the exact CSV schema you need to assemble.

What gets fit:
  * Multinomial-logit demand: shared beta_price, beta_apr, beta_ev + a brand
    intercept (alpha) per nameplate, by maximum likelihood (Adam gradient
    ascent on the grouped log-likelihood).
  * Category-size elasticity: two-way fixed-effects (region x segment and
    month x segment) regression of log(volume) on log(price) — promo variation
    identifies it; the FE absorb seasonality and macro.

Honest scope: a passing recovery test proves the harness works. Whether the
model *form* matches reality is a separate question only REAL data answers —
and identifying price elasticity needs price/incentive VARIATION, which open
aggregate sales data mostly lacks (see --make-template notes).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Optional, Tuple

import vpx_sim as E

G, R, GR, B, X = E.GREEN, E.RED, E.GRAY, E.BOLD, E.RESET

# --------------------------------------------------------------------------- #
# The "true" data-generating process — HIDDEN from the fitter                  #
# --------------------------------------------------------------------------- #

TRUE = {
    "beta_price": 1.45,        # per $10k of price  (= 1.45e-4 per $)
    "beta_apr": 5.5,           # utility per unit of (market_rate - offered_apr)
    "beta_ev": 0.65,           # EV utility x charging-infra score
    "cat_elasticity": -0.70,   # category size vs price-vs-reference
}

# monthly seasonality (index by calendar month 0=Jan)
SEASON = [0.90, 0.93, 1.03, 1.08, 1.11, 1.06, 1.04, 1.07, 1.00, 0.97, 0.94, 1.12]


def options_by_segment() -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = {}
    for v in E.VEHICLES:
        by.setdefault(v.segment, []).append(
            {"key": "our:" + v.vid, "name": v.name, "price0": v.msrp,
             "ev": v.ev_eligible, "alpha": v.alpha, "ours": True})
    for c in E.COMPETITORS:
        by.setdefault(c.segment, []).append(
            {"key": "cmp:" + c.name, "name": c.name, "price0": c.price,
             "ev": c.ev_eligible, "alpha": c.alpha, "ours": False})
    return by


def gen_panel(seed: int = 7, months: int = 36) -> List[dict]:
    rng = random.Random(seed)
    opts = options_by_segment()
    # a wandering macro index ~1.0 (common shock across regions each month)
    macro = []
    m = 1.0
    for _ in range(months):
        m = 0.85 * m + 0.15 * 1.0 + rng.gauss(0, 0.03)
        macro.append(max(0.8, m))

    rows: List[dict] = []
    for t in range(months):
        mo = t % 12
        for reg in E.REGIONS:
            for seg, ol in opts.items():
                prepared = []
                for o in ol:
                    # price variation: Q4 push + idiosyncratic promo events + jitter
                    promo = -0.03 * max(0, mo - 8) / 3.0
                    if rng.random() < 0.30:
                        promo += -rng.uniform(0.02, 0.10)
                    promo += rng.gauss(0, 0.01)
                    price = o["price0"] * reg.price_mult * (1 + promo)
                    apr_diff = rng.uniform(0.02, 0.05) if (o["ours"] and rng.random() < 0.18) else 0.0
                    ev_term = reg.ev_infra if o["ev"] else 0.0
                    prepared.append((o, price, apr_diff, ev_term))

                Us = [o["alpha"] - TRUE["beta_price"] * (p / 1e4)
                      + TRUE["beta_apr"] * a + TRUE["beta_ev"] * e
                      for (o, p, a, e) in prepared]
                mx = max(Us)
                ex = [math.exp(u - mx) for u in Us]
                Z = sum(ex)
                shares = [e / Z for e in ex]

                seg_avg = sum(p for (_, p, _, _) in prepared) / len(prepared)
                ref = sum(o["price0"] for o in ol) / len(ol) * reg.price_mult
                size = (E.SEGMENT_SIZE[reg.rid][seg] * SEASON[mo] * macro[t]
                        * (seg_avg / ref) ** TRUE["cat_elasticity"]
                        * math.exp(rng.gauss(0, 0.04)))

                for i, (o, p, a, e) in enumerate(prepared):
                    units = size * shares[i] * math.exp(rng.gauss(0, 0.05))
                    rows.append({
                        "month": t, "region": reg.rid, "segment": seg, "option": o["key"],
                        "is_ours": 1 if o["ours"] else 0, "price": round(p, 2),
                        "apr_diff": round(a, 4), "ev_flag": 1 if o["ev"] else 0,
                        "ev_infra": round(reg.ev_infra, 3) if o["ev"] else 0.0,
                        "units": round(units, 2)})
    return rows


# --------------------------------------------------------------------------- #
# Cells (one softmax choice-set = month x region x segment)                    #
# --------------------------------------------------------------------------- #


def to_cells(rows: List[dict]) -> List[dict]:
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        groups.setdefault((row["month"], row["region"], row["segment"]), []).append(row)
    return [{"key": k, "rows": rl, "N": sum(x["units"] for x in rl)}
            for k, rl in groups.items()]


# --------------------------------------------------------------------------- #
# Multinomial-logit fit (Adam gradient ascent on grouped log-likelihood)       #
# --------------------------------------------------------------------------- #


def fit_logit(cells: List[dict], iters: int = 4000) -> Tuple[dict, dict, dict]:
    seg_keys: Dict[str, set] = {}
    for c in cells:
        seg = c["key"][2]
        for row in c["rows"]:
            seg_keys.setdefault(seg, set()).add(row["option"])
    ref = {seg: sorted(ks)[0] for seg, ks in seg_keys.items()}      # reference alpha = 0
    free = [(seg, k) for seg in seg_keys for k in sorted(seg_keys[seg]) if k != ref[seg]]
    aidx = {sk: i + 3 for i, sk in enumerate(free)}                 # 0,1,2 = betas
    P = 3 + len(free)

    data = []
    for c in cells:
        seg = c["key"][2]
        opts = []
        for row in c["rows"]:
            xp = row["price"] / 1e4
            xe = row["ev_infra"] if row["ev_flag"] else 0.0
            opts.append((xp, row["apr_diff"], xe, aidx.get((seg, row["option"])), row["units"]))
        opts = [o for o in opts]
        opts_N = sum(o[4] for o in opts)
        if opts_N > 0:
            data.append((opts, opts_N))

    theta = [0.0] * P
    mv = [0.0] * P
    vv = [0.0] * P
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05
    total = sum(N for _, N in data) or 1.0
    prevLL = None
    for it in range(1, iters + 1):
        g = [0.0] * P
        LL = 0.0
        for opts, N in data:
            Us = []
            for (xp, xa, xe, ai, n) in opts:
                a = theta[ai] if ai is not None else 0.0
                Us.append(-theta[0] * xp + theta[1] * xa + theta[2] * xe + a)
            mx = max(Us)
            ex = [math.exp(u - mx) for u in Us]
            Z = sum(ex)
            logZ = mx + math.log(Z)
            for idx, (xp, xa, xe, ai, n) in enumerate(opts):
                Ps = ex[idx] / Z
                if n > 0:
                    LL += n * (Us[idx] - logZ)
                resid = n - N * Ps
                g[0] += resid * (-xp)
                g[1] += resid * xa
                g[2] += resid * xe
                if ai is not None:
                    g[ai] += resid
        g = [x / total for x in g]
        for i in range(P):
            mv[i] = b1 * mv[i] + (1 - b1) * g[i]
            vv[i] = b2 * vv[i] + (1 - b2) * g[i] * g[i]
            mh = mv[i] / (1 - b1 ** it)
            vh = vv[i] / (1 - b2 ** it)
            theta[i] += lr * mh / (math.sqrt(vh) + eps)
        if it % 250 == 0:
            if prevLL is not None and abs(LL - prevLL) < 1e-4 * abs(prevLL):
                break
            prevLL = LL

    beta = {"beta_price": theta[0], "beta_apr": theta[1], "beta_ev": theta[2]}
    alpha = {sk: theta[i] for sk, i in aidx.items()}
    for seg, rk in ref.items():
        alpha[(seg, rk)] = 0.0
    return beta, alpha, ref


def predict_shares(rows: List[dict], beta: dict, alpha: dict, seg: str) -> List[float]:
    Us = []
    for row in rows:
        xe = row["ev_infra"] if row["ev_flag"] else 0.0
        a = alpha.get((seg, row["option"]), 0.0)
        Us.append(-beta["beta_price"] * (row["price"] / 1e4)
                  + beta["beta_apr"] * row["apr_diff"] + beta["beta_ev"] * xe + a)
    mx = max(Us)
    ex = [math.exp(u - mx) for u in Us]
    Z = sum(ex)
    return [e / Z for e in ex]


# --------------------------------------------------------------------------- #
# Category-size elasticity via two-way fixed effects (iterative demeaning)      #
# --------------------------------------------------------------------------- #


def fit_cat_elasticity(cells: List[dict]) -> float:
    pts = []  # [g1, g2, x=log price, y=log units]
    for c in cells:
        (t, reg, seg) = c["key"]
        if c["N"] <= 0:
            continue
        avgp = sum(x["price"] for x in c["rows"]) / len(c["rows"])
        pts.append([reg + "|" + seg, str(t) + "|" + seg, math.log(avgp), math.log(c["N"])])
    if not pts:
        return float("nan")
    for _ in range(60):
        for gcol in (0, 1):
            sx: Dict[str, float] = {}
            sy: Dict[str, float] = {}
            cn: Dict[str, int] = {}
            for p in pts:
                k = p[gcol]
                sx[k] = sx.get(k, 0.0) + p[2]
                sy[k] = sy.get(k, 0.0) + p[3]
                cn[k] = cn.get(k, 0) + 1
            for p in pts:
                k = p[gcol]
                p[2] -= sx[k] / cn[k]
                p[3] -= sy[k] / cn[k]
    sxy = sum(p[2] * p[3] for p in pts)
    sxx = sum(p[2] * p[2] for p in pts)
    return sxy / sxx if sxx else float("nan")


# --------------------------------------------------------------------------- #
# Backtest                                                                     #
# --------------------------------------------------------------------------- #


def backtest(test_cells: List[dict], beta: dict, alpha: dict) -> dict:
    acts, preds, act_sh, pred_sh = [], [], [], []
    for c in test_cells:
        seg = c["key"][2]
        N = c["N"]
        if N <= 0:
            continue
        sh = predict_shares(c["rows"], beta, alpha, seg)
        for i, row in enumerate(c["rows"]):
            if row["is_ours"]:
                acts.append(row["units"])
                preds.append(sh[i] * N)
                act_sh.append(row["units"] / N)
                pred_sh.append(sh[i])
    n = len(acts)
    mape = sum(abs(p - a) / a for a, p in zip(acts, preds) if a > 0) / max(1, n)
    share_mae = sum(abs(a - p) for a, p in zip(act_sh, pred_sh)) / max(1, n)
    ybar = sum(acts) / max(1, n)
    ss_tot = sum((a - ybar) ** 2 for a in acts) or 1.0
    ss_res = sum((a - p) ** 2 for a, p in zip(acts, preds))
    return {"n": n, "units_mape": mape, "share_mae_pp": share_mae * 100,
            "r2": 1 - ss_res / ss_tot}


# --------------------------------------------------------------------------- #
# Reporting helpers                                                            #
# --------------------------------------------------------------------------- #


def _camry_elasticity(cells: List[dict], beta_price_per10k: float) -> Tuple[float, float, float]:
    prices, shares = [], []
    for c in cells:
        if c["N"] <= 0:
            continue
        for row in c["rows"]:
            if row["option"] == "our:camry":
                prices.append(row["price"])
                shares.append(row["units"] / c["N"])
    p = sum(prices) / len(prices)
    s = sum(shares) / len(shares)
    elas = -(beta_price_per10k / 1e4) * p * (1 - s)
    return elas, p, s


def _row(label, true, hat, unit=""):
    err = abs(hat - true) / abs(true) * 100 if true else 0.0
    col = G if err < 12 else (R if err > 25 else GR)
    print("  {:<24}{:>10}{:>14}{}{:>10.1f}% off{}".format(
        label, "{:.3f}{}".format(true, unit), "{:.3f}{}".format(hat, unit), col, err, X))


# --------------------------------------------------------------------------- #
# Modes                                                                        #
# --------------------------------------------------------------------------- #


def run_recovery_demo(seed: int = 7) -> int:
    print()
    print(B + "=" * 74 + X)
    print(B + "  VPX CALIBRATION  ::  parameter-recovery + backtest (synthetic truth)" + X)
    print(B + "=" * 74 + X)
    print(GR + "  36 months x 5 regions x 4 segments, promo-driven price variation + noise." + X)
    print(GR + "  The fitter is NOT told the true coefficients below." + X)

    rows = gen_panel(seed=seed)
    cells = to_cells(rows)
    print(GR + "  panel: {:,} option-month observations in {} choice-set cells.".format(
        len(rows), len(cells)) + X)

    # ---- recovery on the full sample ----
    beta, alpha, ref = fit_logit(cells)
    cat = fit_cat_elasticity(cells)

    print()
    print(B + "  COEFFICIENT RECOVERY  (true vs. fitted)" + X)
    print("  " + "-" * 70)
    print("  {:<24}{:>10}{:>14}{:>18}".format("Parameter", "true", "recovered", "error"))
    _row("beta_price (per $10k)", TRUE["beta_price"], beta["beta_price"])
    _row("beta_apr", TRUE["beta_apr"], beta["beta_apr"])
    _row("beta_ev", TRUE["beta_ev"], beta["beta_ev"])
    _row("category elasticity", TRUE["cat_elasticity"], cat)

    e_true, p_c, s_c = _camry_elasticity(cells, TRUE["beta_price"])
    e_hat, _, _ = _camry_elasticity(cells, beta["beta_price"])
    print("  " + "-" * 70)
    print("  Implied Camry own-price elasticity (at ${:,.0f}, {:.1%} share):".format(p_c, s_c))
    err = abs(e_hat - e_true) / abs(e_true) * 100
    col = G if err < 12 else GR
    print("    true {:.2f}   recovered {}{:.2f}{}   ({:.1f}% off)".format(
        e_true, col, e_hat, X, err))

    # ---- brand-intercept recovery (relative to segment reference) ----
    true_alpha = {}
    for seg, ol in options_by_segment().items():
        for o in ol:
            true_alpha[(seg, o["key"])] = o["alpha"]
    diffs = []
    for v in E.VEHICLES:
        seg = v.segment
        rk = ref[seg]
        rel_true = true_alpha[(seg, "our:" + v.vid)] - true_alpha[(seg, rk)]
        rel_hat = alpha[(seg, "our:" + v.vid)] - alpha[(seg, rk)]
        diffs.append(abs(rel_hat - rel_true))
    print("  Brand intercepts (alpha): mean abs error across 6 nameplates = {}{:.3f}{}".format(
        G if sum(diffs) / len(diffs) < 0.2 else GR, sum(diffs) / len(diffs), X))

    # ---- backtest: fit on first 30 months, predict last 6 ----
    months = sorted(set(c["key"][0] for c in cells))
    cut = months[-6]
    train = [c for c in cells if c["key"][0] < cut]
    test = [c for c in cells if c["key"][0] >= cut]
    beta_tr, alpha_tr, _ = fit_logit(train)
    bt = backtest(test, beta_tr, alpha_tr)

    print()
    print(B + "  OUT-OF-SAMPLE BACKTEST  (train months 0-29, test 30-35, unseen)" + X)
    print("  " + "-" * 70)
    print("    units MAPE       {}{:.1f}%{}   (avg abs error vs actual unit sales)".format(
        G if bt["units_mape"] < 0.10 else R, bt["units_mape"] * 100, X))
    print("    share MAE        {:.2f} pp".format(bt["share_mae_pp"]))
    print("    units R^2        {:.3f}".format(bt["r2"]))

    # ---- verdict ----
    ok = (abs(beta["beta_price"] - TRUE["beta_price"]) / TRUE["beta_price"] < 0.15
          and abs(cat - TRUE["cat_elasticity"]) < 0.15
          and bt["units_mape"] < 0.10)
    print()
    if ok:
        print(G + B + "  VERDICT: harness recovers the truth and predicts unseen months." + X)
        print(GR + "  The fitting machinery is correct. It is ready to point at real data." + X)
        print(GR + "  (Whether the model FORM fits reality is what real data then tells you.)" + X)
    else:
        print(R + B + "  VERDICT: recovery/backtest outside tolerance — inspect before trusting." + X)
    print()
    return 0 if ok else 1


def run_on_data(path: str) -> int:
    rows = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            rows.append({
                "month": int(d["month"]), "region": d["region"], "segment": d["segment"],
                "option": d["option"], "is_ours": int(d["is_ours"]),
                "price": float(d["price"]), "apr_diff": float(d.get("apr_diff", 0) or 0),
                "ev_flag": int(d.get("ev_flag", 0) or 0),
                "ev_infra": float(d.get("ev_infra", 0) or 0), "units": float(d["units"])})
    cells = to_cells(rows)
    months = sorted(set(c["key"][0] for c in cells))
    if len(months) < 6:
        print("Need >= 6 distinct months to backtest; got {}.".format(len(months)))
        return 2
    cut = months[-max(6, len(months) // 5)]
    train = [c for c in cells if c["key"][0] < cut]
    test = [c for c in cells if c["key"][0] >= cut]

    print(B + "\n  VPX CALIBRATION on real data: {}".format(path) + X)
    print(GR + "  {:,} obs, {} cells, {} months ({} train / {} test)".format(
        len(rows), len(cells), len(months),
        len(set(c['key'][0] for c in train)), len(set(c['key'][0] for c in test))) + X)
    beta, alpha, ref = fit_logit(train)
    cat = fit_cat_elasticity(train)
    bt = backtest(test, beta, alpha)
    print()
    print("  fitted beta_price (per $10k) {:.3f}   beta_apr {:.2f}   beta_ev {:.2f}".format(
        beta["beta_price"], beta["beta_apr"], beta["beta_ev"]))
    print("  fitted category elasticity   {:.3f}".format(cat))
    e_hat, p_c, s_c = _camry_elasticity(cells, beta["beta_price"]) if any(
        r["option"] == "our:camry" for r in rows) else (float("nan"), 0, 0)
    if not math.isnan(e_hat):
        print("  implied 1st-nameplate own-price elasticity {:.2f}".format(e_hat))
    print()
    print(B + "  OUT-OF-SAMPLE BACKTEST" + X)
    print("    units MAPE   {}{:.1f}%{}".format(
        G if bt["units_mape"] < 0.10 else R, bt["units_mape"] * 100, X))
    print("    share MAE    {:.2f} pp".format(bt["share_mae_pp"]))
    print("    units R^2    {:.3f}".format(bt["r2"]))
    print()
    if bt["units_mape"] < 0.10:
        print(G + "  Backtest within 10% — the model form holds on this data." + X)
    else:
        print(R + "  Backtest > 10% — the model form needs work on this data "
                  "(re-spec demand, add features, or check data quality)." + X)
    print()
    return 0


TEMPLATE_NOTE = """\
# VPX calibration panel — one row per (option x region x month).
# Columns:
#   month     integer period index (0,1,2,... ascending in time)
#   region    region id (e.g. na, eu, jp)            -- any consistent labels
#   segment   segment id (e.g. Sedan, SUV, Truck)    -- defines the choice set
#   option    nameplate id; prefix yours "our:" and rivals "cmp:"  (our:camry, cmp:Honda Accord)
#   is_ours   1 if it's your vehicle, 0 if a competitor
#   price     average TRANSACTION price that period (USD) -- NOT MSRP if you can help it
#   apr_diff  (market_rate - offered_apr) as a fraction, 0 if none   e.g. 0.04
#   ev_flag   1 if electrified, else 0
#   ev_infra  0..1 charging-infrastructure score for that region (0 if ev_flag=0)
#   units     unit sales that period
#
# IDENTIFICATION CAVEAT (read this before assembling data):
#   To estimate PRICE elasticity the harness needs price VARIATION that moves
#   independently of demand -- i.e. real transaction-price/incentive history.
#   Open data gives you 'units' (Toyota pressroom, GoodCarBadCar) and MSRP, but
#   MSRP barely moves and per-model incentive/ATP history is largely proprietary
#   (Cox/KBB/J.D. Power). Without that variation the price coefficient is weakly
#   identified. Plan to source incentive/ATP data, or treat price coefficients
#   from open data as indicative only and lean on the share model.
month,region,segment,option,is_ours,price,apr_diff,ev_flag,ev_infra,units
0,na,Sedan,our:camry,1,27250.00,0.00,0,0.0,18342
0,na,Sedan,cmp:Honda Accord,0,27010.00,0.00,0,0.0,17120
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="VPX calibration & backtest harness")
    ap.add_argument("--data", help="CSV panel to calibrate on (see --make-template)")
    ap.add_argument("--make-template", action="store_true", help="print the CSV schema + notes")
    ap.add_argument("--seed", type=int, default=7, help="seed for the synthetic recovery demo")
    args = ap.parse_args()
    if args.make_template:
        print(TEMPLATE_NOTE, end="")
        return 0
    if args.data:
        return run_on_data(args.data)
    return run_recovery_demo(seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
