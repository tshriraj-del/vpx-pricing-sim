#!/usr/bin/env python3
"""
VPX v2 — shared web layer.

Engine <-> JSON glue + scenario endpoints, imported by BOTH the local dev
server (vpx_app.py) and the Vercel serverless functions (api/*.py). Keeping it
here means there is exactly one implementation behind localhost and production.
"""
from __future__ import annotations

import os
from typing import Dict, List

import vpx_sim as E

try:
    import vpx_store
except Exception:                      # pragma: no cover
    vpx_store = None

SEG_COLOR = {"Sedan": "#3b6ea5", "SUV": "#2e8b8b", "Truck": "#c08a2e", "Hybrid": "#3f9d6c"}

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "index.html")


def page_html() -> bytes:
    with open(_HTML_PATH, "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Engine -> JSON                                                               #
# --------------------------------------------------------------------------- #


def _params_from(d: Dict) -> E.Params:
    p = E.Params()
    for k in ("market_rate", "loyalty_base", "fleet_mix", "seg_elasticity", "beta_price"):
        if k in d and d[k] is not None:
            setattr(p, k, float(d[k]))
    return p


def _scenario_from(body: Dict) -> E.Scenario:
    params = _params_from(body.get("params", {}))
    levers: Dict[str, E.Lever] = {}
    for vid, lv in body.get("levers", {}).items():
        apr = lv.get("apr", None)
        subv = bool(lv.get("subvented", False)) and apr is not None \
            and float(apr) < params.market_rate - 1e-9
        levers[vid] = E.Lever(
            price_mult_delta=float(lv.get("price_mult_delta", 0.0)),
            cashback=float(lv.get("cashback", 0.0)),
            loyalty_bonus=float(lv.get("loyalty_bonus", 0.0)),
            fleet_discount=float(lv.get("fleet_discount", 0.0)),
            apr=(float(apr) if subv else None),
            subvented=subv,
            ev_credit=float(lv.get("ev_credit", 0.0)),
            state_incentive=float(lv.get("state_incentive", 0.0)),
            dos=float(lv.get("dos", 50.0)),
        )
    return E.Scenario(name="ui", params=params, levers=levers)


def summarize(cells: List[E.Cell], k: E.KPIs) -> Dict:
    agg: Dict[str, Dict[str, float]] = {}
    for c in cells:
        a = agg.setdefault(c.vid, {"units": 0.0, "rev": 0.0, "gm": 0.0,
                                   "net_u": 0.0, "shr_u": 0.0, "resid": 0.0})
        a["units"] += c.units
        a["rev"] += c.revenue_usd
        a["gm"] += c.gross_margin_usd
        a["net_u"] += c.oem_net * c.units
        a["shr_u"] += c.seg_share * c.units
        a["resid"] += c.residual_risk_usd
    name_by = {v.vid: v.name for v in E.VEHICLES}
    seg_by = {v.vid: v.segment for v in E.VEHICLES}
    vehicles = []
    for v in E.VEHICLES:
        a = agg[v.vid]
        u = a["units"] or 1.0
        vehicles.append({
            "vid": v.vid, "name": name_by[v.vid], "segment": seg_by[v.vid],
            "color": SEG_COLOR[seg_by[v.vid]],
            "units": a["units"], "revenue": a["rev"], "oem_net": a["net_u"] / u,
            "margin_pct": (a["gm"] / a["rev"] if a["rev"] else 0.0),
            "share": a["shr_u"] / u,
        })
    reg_rev: Dict[str, float] = {}
    for c in cells:
        reg_rev[c.rid] = reg_rev.get(c.rid, 0.0) + c.revenue_usd
    rname = {r.rid: r.name for r in E.REGIONS}
    regions = [{"rid": rid, "name": rname[rid], "revenue": rev} for rid, rev in reg_rev.items()]
    return {
        "kpis": {
            "revenue": k.revenue_usd, "gross_margin": k.gross_margin_usd,
            "contribution": k.contribution_usd, "margin_pct": k.margin_pct,
            "market_share": k.market_share, "avg_conversion": k.avg_conversion,
            "avg_turnover_days": k.avg_turnover_days,
            "residual_risk": k.residual_risk_usd, "total_units": k.total_units,
        },
        "vehicles": vehicles, "regions": regions,
    }


def meta() -> Dict:
    base_cells, base_k = E.run(E.preset_baseline())
    p = E.Params()
    return {
        "vehicles": [{
            "vid": v.vid, "name": v.name, "segment": v.segment, "powertrain": v.powertrain,
            "msrp": v.msrp, "ev_eligible": v.ev_eligible, "capacity": E.CAPACITY[v.vid],
            "color": SEG_COLOR[v.segment],
        } for v in E.VEHICLES],
        "regions": [{"rid": r.rid, "name": r.name} for r in E.REGIONS],
        "defaults": {
            "market_rate": p.market_rate, "loyalty_base": p.loyalty_base,
            "fleet_mix": p.fleet_mix, "seg_elasticity": p.seg_elasticity,
            "beta_price": p.beta_price,
        },
        "baseline": summarize(base_cells, base_k),
        "store": scenarios_available(),
    }


def simulate(body: Dict) -> Dict:
    cells, k = E.run(_scenario_from(body))
    return summarize(cells, k)


def do_optimize(body: Dict) -> Dict:
    cfg = E.OptConfig()
    if body.get("share_floor") is not None:
        cfg.share_floor = float(body["share_floor"])
    if body.get("subsidy_budget_usd") is not None:
        cfg.subsidy_budget_usd = float(body["subsidy_budget_usd"])
    best, evals = E.optimize(cfg)
    base_cells, base_k = E.run(E.preset_baseline())
    n = len(E._OPT_VEH)
    mkt = cfg.params.market_rate
    levers = {}
    for i, vid in enumerate(E._OPT_VEH):
        apr = best.x[n + i]
        subv = apr < mkt - 1e-9
        levers[vid] = {"price_mult_delta": best.x[i],
                       "apr": (apr if subv else None), "subvented": subv}
    name_by = {v.vid: v.name for v in E.VEHICLES}
    cap_bind = [name_by[vid] for vid, u in best.units_by.items()
                if u >= cfg.capacity[vid] * 0.99]
    duc = best.contribution - base_k.contribution_usd
    return {
        "levers": levers, "uplift_usd": duc,
        "uplift_pct": (duc / base_k.contribution_usd * 100 if base_k.contribution_usd else 0.0),
        "evals": evals,
        "binding": {
            "share": best.kpis.market_share <= cfg.share_floor + 0.005,
            "share_detail": "{:.1f}% vs {:.0f}% floor".format(
                best.kpis.market_share * 100, cfg.share_floor * 100),
            "capacity": cap_bind,
            "budget": best.subsidy_usd >= cfg.subsidy_budget_usd * 0.99,
            "budget_detail": "${:,.0f}M of ${:,.0f}M".format(
                best.subsidy_usd / 1e6, cfg.subsidy_budget_usd / 1e6),
        },
        "result": summarize(best.cells, best.kpis), "share_floor": cfg.share_floor,
    }


# --------------------------------------------------------------------------- #
# Scenarios (Turso-backed, with graceful fallback)                             #
# --------------------------------------------------------------------------- #


def scenarios_available() -> bool:
    return bool(vpx_store and vpx_store.available())


def scenarios_get() -> Dict:
    if not scenarios_available():
        return {"available": False, "scenarios": []}
    try:
        return {"available": True, "scenarios": vpx_store.list_active(3)}
    except Exception:
        return {"available": False, "scenarios": []}


def scenarios_post(body: Dict) -> Dict:
    if not scenarios_available():
        return {"available": False}
    try:
        sid = vpx_store.save(body.get("id"), body.get("name", "Scenario"),
                             body.get("color", "#EB0A1E"), body.get("levers", {}),
                             body.get("params", {}), body.get("kpis", {}))
        return {"available": True, "id": sid, "scenarios": vpx_store.list_active(3)}
    except Exception:
        return {"available": False}


def scenarios_delete(sid: str) -> Dict:
    if not scenarios_available():
        return {"available": False}
    try:
        vpx_store.delete(sid)
        return {"available": True, "scenarios": vpx_store.list_active(3)}
    except Exception:
        return {"available": False}
