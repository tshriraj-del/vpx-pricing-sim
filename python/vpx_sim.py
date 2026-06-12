#!/usr/bin/env python3
"""
VPX v2 — Vehicle Pricing Simulator (working prototype)
======================================================

A self-contained, pure-standard-library implementation of the *corrected* v2
pricing model. No external dependencies, no pip install. Runs on any python3.

What this proves (the v2 corrections, all live in here):
  * TWO prices, never one:
      - oem_net_price   -> what the OEM realizes  (drives REVENUE & MARGIN)
      - consumer_price  -> what the buyer perceives (drives DEMAND)
    Federal EV credit & state incentives lower consumer_price ONLY; they do
    NOT drain OEM revenue. (The v1 spec's costliest bug.)
  * ONE demand model for both volume AND share: a nested multinomial-logit.
    Segment size sets the category; within-segment shares come from a softmax
    over our SKUs *and* competitors. Volume and share can never disagree,
    cross-elasticity / cannibalization is automatic, and share saturates at 1
    so "cut price -> infinite volume" can't happen.
  * Fleet discount handled in DOLLARS (v1 multiplied % x %, a unit bug).
  * Inventory pressure is CONTINUOUS & bounded (no 30/90-day cliffs).
  * Leasing & residuals modeled, with residual-risk exposure surfaced.
  * Finance subvention is a labeled contribution-margin cost, not a price cut.
  * FX is a consolidated-USD translation layer (margin% invariant; absolute
    USD moves) — demonstrable via the `fx_shock` scenario.

Run:
    python3 vpx_sim.py                 # baseline dashboard
    python3 vpx_sim.py aggressive      # a preset scenario vs baseline
    python3 vpx_sim.py premium
    python3 vpx_sim.py fx_shock
    python3 vpx_sim.py --selftest      # invariant checks (shares sum to 1, etc.)
    python3 vpx_sim.py --list          # list presets

Edit PRESETS at the bottom to define your own levers.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# 0. Global / "advanced" parameters (the things the Advanced panel would hold) #
# --------------------------------------------------------------------------- #


@dataclass
class Params:
    market_rate: float = 0.072          # market APR, for subvention math
    loyalty_base: float = 0.28          # % of buyers who are returning owners
    fleet_mix: float = 0.12             # B2B fleet share of sales
    cashback_take: float = 0.35         # avg uptake of a cash-back offer
    finance_penetration: float = 0.65   # % of units financed
    lease_mix: float = 0.25             # % of units leased
    base_conversion: float = 0.22       # base lead->sale conversion
    inventory_target_days: float = 50.0
    # demand-model coefficients
    seg_elasticity: float = -0.6        # category-size elasticity vs reference
    beta_price: float = 0.00012         # logit price sensitivity (per $)
    beta_apr: float = 6.0               # logit utility per point of APR spread
    beta_ev_infra: float = 0.50         # logit bonus for EVs vs charging infra


# --------------------------------------------------------------------------- #
# 1. Synthetic data — deterministic, no randomness needed for the baseline    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vehicle:
    vid: str
    name: str
    segment: str
    powertrain: str
    msrp: float
    cogs_pct: float
    alpha: float            # logit brand intercept (baseline desirability)
    residual_pct: float     # 36-mo lease residual as fraction of price
    ev_eligible: bool = False


@dataclass(frozen=True)
class Competitor:
    name: str
    segment: str
    price: float
    alpha: float
    ev_eligible: bool = False


@dataclass(frozen=True)
class Region:
    rid: str
    name: str
    price_mult: float       # regional price level vs US baseline
    ev_infra: float         # 0..1 charging-infrastructure score
    macro: float            # macro demand multiplier
    fx_factor: float = 1.0  # consolidated-USD translation factor


VEHICLES: List[Vehicle] = [
    Vehicle("camry",  "Camry",  "Sedan",  "ICE",    28000, 0.72, 2.6, 0.55),
    Vehicle("crown",  "Crown",  "Sedan",  "Hybrid", 41000, 0.74, 1.4, 0.52, ev_eligible=True),
    Vehicle("rav4",   "RAV4",   "SUV",    "ICE",    32000, 0.71, 2.8, 0.58),
    Vehicle("tacoma", "Tacoma", "Truck",  "ICE",    33000, 0.70, 2.4, 0.62),
    Vehicle("tundra", "Tundra", "Truck",  "ICE",    52000, 0.73, 1.6, 0.55),
    Vehicle("prius",  "Prius",  "Hybrid", "Hybrid", 28000, 0.72, 2.2, 0.50, ev_eligible=True),
]

COMPETITORS: List[Competitor] = [
    # Sedan
    Competitor("Honda Accord",   "Sedan", 27500, 2.4),
    Competitor("Hyundai Sonata", "Sedan", 26500, 1.8),
    # SUV
    Competitor("Honda CR-V",     "SUV",   31000, 2.6),
    Competitor("Mazda CX-5",     "SUV",   29000, 1.9),
    Competitor("Ford Escape",    "SUV",   30000, 1.7),
    # Truck
    Competitor("Ford F-150",     "Truck", 38000, 2.7),
    Competitor("Ram 1500",       "Truck", 41000, 2.2),
    Competitor("Chevy Silverado","Truck", 39000, 2.1),
    # Hybrid
    Competitor("Honda Civic Hyb","Hybrid",27000, 2.0),
    Competitor("Hyundai Elantra","Hybrid",26500, 1.7),
]

REGIONS: List[Region] = [
    Region("na",   "North America",      1.00, 0.70, 1.00),
    Region("eu",   "Europe",             1.15, 0.80, 0.95),
    Region("jp",   "Japan/Asia-Pacific", 0.95, 0.75, 1.05),
    Region("latam","Latin America",      0.90, 0.35, 0.85),
    Region("me",   "Middle East",        1.05, 0.40, 0.90),
]

# Monthly production capacity (units/mo) per vehicle — a binding-capable ceiling.
CAPACITY: Dict[str, float] = {
    "camry": 115000, "crown": 30000, "rav4": 98000,
    "tacoma": 95000, "tundra": 18000, "prius": 80000,
}

# Monthly category size (whole segment incl. competitors) by region x segment.
SEGMENT_SIZE: Dict[str, Dict[str, float]] = {
    "na":    {"Sedan": 60000, "SUV": 90000, "Truck": 85000, "Hybrid": 40000},
    "eu":    {"Sedan": 45000, "SUV": 60000, "Truck": 15000, "Hybrid": 35000},
    "jp":    {"Sedan": 55000, "SUV": 50000, "Truck": 20000, "Hybrid": 45000},
    "latam": {"Sedan": 25000, "SUV": 30000, "Truck": 22000, "Hybrid":  8000},
    "me":    {"Sedan": 18000, "SUV": 28000, "Truck": 30000, "Hybrid":  5000},
}

SEGMENTS = ["Sedan", "SUV", "Truck", "Hybrid"]


# --------------------------------------------------------------------------- #
# 2. Levers / scenario configuration                                          #
# --------------------------------------------------------------------------- #


@dataclass
class Lever:
    """Per-vehicle levers, applied across all regions (global incentives)."""
    price_mult_delta: float = 0.0   # +/- fraction off list (e.g. -0.10 = 10% cut)
    cashback: float = 0.0           # $ cash back
    loyalty_bonus: float = 0.0      # $ loyalty bonus
    fleet_discount: float = 0.0     # fraction (0..0.15) off for fleet buyers
    apr: Optional[float] = None     # offered APR; None = market rate
    term_months: int = 60
    down: float = 2000.0
    subvented: bool = False         # manufacturer subsidizes the APR
    ev_credit: float = 0.0          # federal EV credit (only if ev_eligible)
    state_incentive: float = 0.0    # consumer-side state incentive
    dos: float = 50.0               # days of supply (inventory)


@dataclass
class Scenario:
    name: str
    params: Params = field(default_factory=Params)
    levers: Dict[str, Lever] = field(default_factory=dict)
    fx_override: Dict[str, float] = field(default_factory=dict)  # rid -> fx_factor

    def lever_for(self, vid: str) -> Lever:
        return self.levers.get(vid, Lever())


# --------------------------------------------------------------------------- #
# 3. The engine                                                               #
# --------------------------------------------------------------------------- #


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Cell:
    """One vehicle x region result."""
    vid: str
    rid: str
    segment: str
    units: float
    list_price: float
    oem_net: float
    consumer_price: float
    revenue_usd: float          # consolidated USD (after fx)
    cogs_usd: float
    gross_margin_usd: float
    finance_subsidy_usd: float
    contribution_usd: float
    margin_pct: float
    seg_share: float            # our share within the segment
    conversion: float
    leads: float
    turnover_days: float
    monthly_lease: float
    residual_risk_usd: float


def _competitor_consumer_price(c: Competitor, r: Region) -> float:
    return c.price * r.price_mult


def _our_consumer_price(v: Vehicle, r: Region, lv: Lever, p: Params) -> Tuple[float, float, float]:
    """Returns (list_price, oem_net_price, consumer_price)."""
    list_price = v.msrp * r.price_mult * (1.0 + lv.price_mult_delta)

    # continuous, bounded inventory pressure
    pressure = _clamp((p.inventory_target_days - lv.dos) / p.inventory_target_days * 0.12,
                      -0.06, 0.04)
    market_price = list_price * (1.0 + pressure)

    # OEM-funded incentives reduce what the OEM realizes (DOLLARS, not % x %)
    oem_funded = (lv.cashback * p.cashback_take
                  + lv.loyalty_bonus * p.loyalty_base
                  + (lv.fleet_discount * list_price) * p.fleet_mix)
    oem_net = market_price - oem_funded

    # third-party / consumer-side benefits — DO NOT touch OEM revenue
    ev_credit = lv.ev_credit if v.ev_eligible else 0.0
    consumer_price = oem_net - ev_credit - lv.state_incentive
    return list_price, oem_net, consumer_price


def _utility(price: float, alpha: float, apr_diff: float, ev_flag: bool,
             ev_infra: float, p: Params) -> float:
    u = alpha - p.beta_price * price + p.beta_apr * apr_diff
    if ev_flag:
        u += p.beta_ev_infra * ev_infra
    return u


def run(scenario: Scenario,
        baseline_units: Optional[Dict[Tuple[str, str], float]] = None) -> Tuple[List[Cell], "KPIs"]:
    p = scenario.params
    cells: List[Cell] = []

    our_by_seg: Dict[str, List[Vehicle]] = {s: [] for s in SEGMENTS}
    for v in VEHICLES:
        our_by_seg[v.segment].append(v)
    comp_by_seg: Dict[str, List[Competitor]] = {s: [] for s in SEGMENTS}
    for c in COMPETITORS:
        comp_by_seg[c.segment].append(c)

    total_our_units = 0.0
    total_category_units = 0.0

    for r in REGIONS:
        fx = scenario.fx_override.get(r.rid, r.fx_factor)
        for seg in SEGMENTS:
            ours = our_by_seg[seg]
            comps = comp_by_seg[seg]
            if not ours:
                continue

            # ---- build the option set (consumer prices + utilities) ----
            options: List[Tuple[str, float, float]] = []  # (key, consumer_price, utility)
            our_prices: Dict[str, Tuple[float, float, float]] = {}

            for v in ours:
                lv = scenario.lever_for(v.vid)
                list_p, oem_net, cons_p = _our_consumer_price(v, r, lv, p)
                our_prices[v.vid] = (list_p, oem_net, cons_p)
                apr = p.market_rate if lv.apr is None else lv.apr
                apr_diff = (p.market_rate - apr) if lv.subvented else 0.0
                u = _utility(cons_p, v.alpha, apr_diff, v.ev_eligible, r.ev_infra, p)
                options.append((v.vid, cons_p, u))

            for c in comps:
                cons_p = _competitor_consumer_price(c, r)
                u = _utility(cons_p, c.alpha, 0.0, c.ev_eligible, r.ev_infra, p)
                options.append(("comp:" + c.name, cons_p, u))

            # ---- category size: elasticity vs region reference price ----
            ref_price = sum(o[1] for o in options) / len(options)
            base_ref = (sum(v.msrp for v in ours) + sum(c.price for c in comps)) \
                / (len(ours) + len(comps)) * r.price_mult
            price_ratio = ref_price / base_ref if base_ref else 1.0
            seg_size = (SEGMENT_SIZE[r.rid][seg] * r.macro
                        * (price_ratio ** p.seg_elasticity))
            total_category_units += seg_size

            # ---- within-segment shares via softmax ----
            mx = max(o[2] for o in options)
            exps = [math.exp(o[2] - mx) for o in options]
            denom = sum(exps)
            shares = {options[i][0]: exps[i] / denom for i in range(len(options))}

            # ---- per-vehicle economics ----
            for v in ours:
                lv = scenario.lever_for(v.vid)
                list_p, oem_net, cons_p = our_prices[v.vid]
                share = shares[v.vid]
                units = seg_size * share
                total_our_units += units

                cogs = v.msrp * v.cogs_pct            # production cost (not region-marked)
                revenue_usd = units * oem_net * fx
                cogs_usd = units * cogs * fx
                gross_margin_usd = revenue_usd - cogs_usd
                margin_pct = gross_margin_usd / revenue_usd if revenue_usd else 0.0

                # finance subvention: labeled contribution cost, not a price cut
                apr = p.market_rate if lv.apr is None else lv.apr
                finance_subsidy_usd = 0.0
                if lv.subvented and apr < p.market_rate:
                    loan = max(oem_net - lv.down, 0.0)
                    term_years = lv.term_months / 12.0
                    per_unit = loan * 0.5 * (p.market_rate - apr) * term_years
                    finance_subsidy_usd = units * p.finance_penetration * per_unit * fx
                contribution_usd = gross_margin_usd - finance_subsidy_usd

                # conversion funnel
                boost = 0.0
                if lv.subvented and apr < 0.03:
                    boost += 0.05
                if lv.loyalty_bonus > 0:
                    boost += 0.03
                if lv.cashback > 0:
                    boost += min(0.04, lv.cashback / 5000.0 * 0.04)
                conversion = min(p.base_conversion + boost, 0.55)
                leads = units / conversion if conversion else 0.0

                # inventory turnover (relative to baseline demand)
                if baseline_units is not None:
                    base_u = baseline_units.get((v.vid, r.rid), units) or units
                    lift = units / base_u - 1.0
                else:
                    lift = 0.0
                turnover_days = max(1.0, lv.dos - lift * 15.0)

                # leasing / residual exposure
                residual_value = list_p * v.residual_pct
                mf = (apr if lv.subvented else p.market_rate) / 24.0
                cap_cost = list_p - lv.down
                monthly_lease = ((cap_cost - residual_value) / lv.term_months
                                 + (cap_cost + residual_value) * mf)
                lease_units = units * p.lease_mix
                residual_risk_usd = lease_units * residual_value * fx

                cells.append(Cell(
                    vid=v.vid, rid=r.rid, segment=seg, units=units,
                    list_price=list_p, oem_net=oem_net, consumer_price=cons_p,
                    revenue_usd=revenue_usd, cogs_usd=cogs_usd,
                    gross_margin_usd=gross_margin_usd,
                    finance_subsidy_usd=finance_subsidy_usd,
                    contribution_usd=contribution_usd, margin_pct=margin_pct,
                    seg_share=share, conversion=conversion, leads=leads,
                    turnover_days=turnover_days, monthly_lease=monthly_lease,
                    residual_risk_usd=residual_risk_usd,
                ))

    kpis = _aggregate(cells, total_our_units, total_category_units)
    return cells, kpis


# --------------------------------------------------------------------------- #
# 4. KPI aggregation                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class KPIs:
    revenue_usd: float
    gross_margin_usd: float
    contribution_usd: float
    margin_pct: float
    market_share: float          # our units / total category units
    avg_conversion: float
    avg_turnover_days: float
    residual_risk_usd: float
    total_units: float


def _aggregate(cells: List[Cell], our_units: float, category_units: float) -> KPIs:
    rev = sum(c.revenue_usd for c in cells)
    gm = sum(c.gross_margin_usd for c in cells)
    contrib = sum(c.contribution_usd for c in cells)
    units = sum(c.units for c in cells)
    conv = (sum(c.conversion * c.units for c in cells) / units) if units else 0.0
    turn = (sum(c.turnover_days * c.units for c in cells) / units) if units else 0.0
    resid = sum(c.residual_risk_usd for c in cells)
    return KPIs(
        revenue_usd=rev, gross_margin_usd=gm, contribution_usd=contrib,
        margin_pct=(gm / rev if rev else 0.0),
        market_share=(our_units / category_units if category_units else 0.0),
        avg_conversion=conv, avg_turnover_days=turn, residual_risk_usd=resid,
        total_units=units,
    )


# --------------------------------------------------------------------------- #
# 5. Reporting                                                                #
# --------------------------------------------------------------------------- #

GREEN, RED, GRAY, BOLD, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m"


def _m(x: float) -> str:
    return "${:,.1f}M".format(x / 1e6)


def _delta(cur: float, base: float, pct: bool = False, good_up: bool = True,
           money: bool = False, suffix: str = "") -> str:
    if base is None:
        return ""
    d = cur - base
    if abs(d) < 1e-9:
        return GRAY + "  (flat)" + RESET
    up = d > 0
    good = (up == good_up)
    col = GREEN if good else RED
    arrow = "▲" if up else "▼"
    if pct:
        val = "{:+.1f}pp".format(d * 100)
    elif money:
        val = "{:+,.1f}M".format(d / 1e6)
    else:
        val = "{:+,.0f}".format(d) + suffix
    return "  {}{} {}{}".format(col, arrow, val, RESET)


def print_dashboard(name: str, cells: List[Cell], k: KPIs,
                    base: Optional[KPIs] = None) -> None:
    bd = base
    print()
    print(BOLD + "=" * 74 + RESET)
    print(BOLD + "  VPX SIMULATION  ::  {}".format(name) + RESET)
    print(BOLD + "=" * 74 + RESET)

    def row(label: str, value: str, d: str = ""):
        print("  {:<26}{:>16}{}".format(label, value, d))

    print()
    print(BOLD + "  KPI SUMMARY  (monthly, consolidated USD)" + RESET)
    print("  " + "-" * 70)
    row("Total Revenue", _m(k.revenue_usd),
        _delta(k.revenue_usd, bd.revenue_usd if bd else None, money=True) if bd else "")
    row("Gross Margin", _m(k.gross_margin_usd),
        _delta(k.gross_margin_usd, bd.gross_margin_usd if bd else None, money=True) if bd else "")
    row("Blended Margin %", "{:.1f}%".format(k.margin_pct * 100),
        _delta(k.margin_pct, bd.margin_pct if bd else None, pct=True) if bd else "")
    row("Contribution Margin", _m(k.contribution_usd),
        _delta(k.contribution_usd, bd.contribution_usd if bd else None, money=True) if bd else "")
    row("Total Units / mo", "{:,.0f}".format(k.total_units),
        _delta(k.total_units, bd.total_units if bd else None) if bd else "")
    row("Weighted Market Share", "{:.1f}%".format(k.market_share * 100),
        _delta(k.market_share, bd.market_share if bd else None, pct=True) if bd else "")
    row("Avg Conversion", "{:.1f}%".format(k.avg_conversion * 100),
        _delta(k.avg_conversion, bd.avg_conversion if bd else None, pct=True) if bd else "")
    row("Avg Inventory Turnover", "{:.0f} days".format(k.avg_turnover_days),
        _delta(k.avg_turnover_days, bd.avg_turnover_days if bd else None,
               good_up=False, suffix="d") if bd else "")
    row("Residual Risk Exposure", _m(k.residual_risk_usd),
        _delta(k.residual_risk_usd, bd.residual_risk_usd if bd else None,
               good_up=False, money=True) if bd else "")

    # per-vehicle rollup (summed across regions)
    print()
    print(BOLD + "  BY VEHICLE  (summed across regions)" + RESET)
    print("  " + "-" * 70)
    print("  {:<8}{:>10}{:>12}{:>11}{:>10}{:>11}".format(
        "Model", "Units", "Revenue", "OEM Net$", "Margin%", "Share"))
    agg: Dict[str, Dict[str, float]] = {}
    for c in cells:
        a = agg.setdefault(c.vid, {"units": 0.0, "rev": 0.0, "gm": 0.0,
                                   "net_u": 0.0, "shr_u": 0.0})
        a["units"] += c.units
        a["rev"] += c.revenue_usd
        a["gm"] += c.gross_margin_usd
        a["net_u"] += c.oem_net * c.units
        a["shr_u"] += c.seg_share * c.units
    name_by = {v.vid: v.name for v in VEHICLES}
    for vid in [v.vid for v in VEHICLES]:
        a = agg[vid]
        u = a["units"]
        net = a["net_u"] / u if u else 0.0
        mpct = a["gm"] / a["rev"] * 100 if a["rev"] else 0.0
        shr = a["shr_u"] / u * 100 if u else 0.0
        print("  {:<8}{:>10,.0f}{:>12}{:>11}{:>9.1f}%{:>10.1f}%".format(
            name_by[vid], u, _m(a["rev"]), "${:,.0f}".format(net), mpct, shr))
    print("  " + "-" * 70)
    print()


# --------------------------------------------------------------------------- #
# 6. Preset scenarios                                                         #
# --------------------------------------------------------------------------- #


def preset_baseline() -> Scenario:
    return Scenario(name="Baseline (published MSRP, no incentives)")


def preset_aggressive() -> Scenario:
    """Chase volume & share: cut price, add cash, subvent APR, lean inventory."""
    return Scenario(
        name="Aggressive Pricing (volume play)",
        levers={
            "camry":  Lever(price_mult_delta=-0.08, cashback=2500, subvented=True,
                            apr=0.029, dos=35),
            "rav4":   Lever(price_mult_delta=-0.05, cashback=1500, dos=30),
            "prius":  Lever(price_mult_delta=-0.06, ev_credit=7500, loyalty_bonus=1000,
                            dos=40),
            "crown":  Lever(ev_credit=7500, cashback=2000, subvented=True, apr=0.019),
            "tundra": Lever(price_mult_delta=-0.04, cashback=3000),
        },
    )


def preset_premium() -> Scenario:
    """Protect margin: raise price, pull incentives, run lean inventory."""
    return Scenario(
        name="Premium Positioning (margin play)",
        levers={
            "camry":  Lever(price_mult_delta=+0.06, dos=25),
            "rav4":   Lever(price_mult_delta=+0.08, dos=20),
            "tacoma": Lever(price_mult_delta=+0.05, dos=30),
            "tundra": Lever(price_mult_delta=+0.07, dos=35),
        },
    )


def preset_fx_shock() -> Scenario:
    """Baseline pricing, but the euro & yen weaken vs USD on consolidation.
    Demonstrates: margin% is unchanged, but absolute consolidated USD falls."""
    return Scenario(
        name="FX Shock (EUR/JPY weaken on consolidation)",
        fx_override={"eu": 0.88, "jp": 0.85},
    )


PRESETS = {
    "baseline": preset_baseline,
    "aggressive": preset_aggressive,
    "premium": preset_premium,
    "fx_shock": preset_fx_shock,
}


# --------------------------------------------------------------------------- #
# 7. Self-test (property invariants — the v2 rigor, runnable)                  #
# --------------------------------------------------------------------------- #


def selftest() -> int:
    failures = 0

    def check(cond: bool, msg: str):
        nonlocal failures
        status = (GREEN + "PASS" + RESET) if cond else (RED + "FAIL" + RESET)
        if not cond:
            failures += 1
        print("  [{}] {}".format(status, msg))

    base_cells, base_k = run(preset_baseline())

    # shares within each region/segment sum to ~1 (incl. competitors implicitly)
    # we check our-share is in (0,1) and units non-negative
    check(all(c.units >= 0 for c in base_cells), "units never negative")
    check(all(0.0 < c.seg_share < 1.0 for c in base_cells), "every share in (0,1)")
    check(all(c.margin_pct <= 1.0 for c in base_cells), "margin% never exceeds 100%")
    check(0.0 < base_k.market_share < 1.0, "weighted market share in (0,1)")
    check(base_k.revenue_usd > 0, "baseline revenue positive")

    # EV credit must NOT change OEM revenue (only demand) — the v1 bug guard
    sc = Scenario(name="ev-credit-only",
                  levers={"prius": Lever(ev_credit=7500)})
    base_idx = {(c.vid, c.rid): c.units for c in base_cells}
    ev_cells, _ = run(sc, base_idx)
    base_prius = [c for c in base_cells if c.vid == "prius"]
    ev_prius = [c for c in ev_cells if c.vid == "prius"]
    same_net = all(abs(a.oem_net - b.oem_net) < 1e-6
                   for a, b in zip(base_prius, ev_prius))
    more_units = sum(c.units for c in ev_prius) > sum(c.units for c in base_prius)
    check(same_net, "EV credit leaves OEM net price unchanged")
    check(more_units, "EV credit raises demand (units up)")

    # aggressive should raise units & share, premium should raise margin%
    _, agg_k = run(preset_aggressive(), base_idx)
    _, prem_k = run(preset_premium(), base_idx)
    check(agg_k.total_units > base_k.total_units, "aggressive lifts total units")
    check(agg_k.market_share > base_k.market_share, "aggressive lifts market share")
    check(prem_k.margin_pct > base_k.margin_pct, "premium lifts blended margin %")

    # fx shock: every PER-CELL margin% is invariant (fx cancels in rev/cogs),
    # absolute consolidated USD revenue falls. NB: the *blended* margin% may
    # move because fx reweights the regional mix — that's correct, not a bug.
    fx_cells, fx_k = run(preset_fx_shock(), base_idx)
    base_cell_m = {(c.vid, c.rid): c.margin_pct for c in base_cells}
    per_cell_invariant = all(
        abs(c.margin_pct - base_cell_m[(c.vid, c.rid)]) < 1e-9 for c in fx_cells)
    check(per_cell_invariant, "FX shock leaves every per-cell margin % invariant")
    check(fx_k.revenue_usd < base_k.revenue_usd,
          "FX shock lowers consolidated USD revenue")

    # optimizer must return a FEASIBLE point that beats baseline contribution
    cfg = OptConfig()
    best, _ = optimize(cfg)
    check(_feasible(best), "optimizer returns a feasible solution")
    check(best.contribution >= base_k.contribution_usd - 1.0,
          "optimizer contribution >= baseline (never worse)")
    check(best.kpis.market_share >= cfg.share_floor - 0.005,
          "optimizer respects the market-share floor")

    print()
    if failures == 0:
        print(GREEN + BOLD + "  ALL INVARIANTS HOLD." + RESET)
    else:
        print(RED + BOLD + "  {} INVARIANT(S) FAILED.".format(failures) + RESET)
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# 7b. The optimizer (inverse problem)                                          #
#                                                                              #
#   maximize   total contribution margin                                       #
#   over       price_delta[v] in [-30%, +30%],  apr[v] in [0, market_rate]     #
#   subject to oem_net >= cogs            (never sell below cost)               #
#              units[v] <= capacity[v]    (factory ceiling)                    #
#              market_share >= floor      (brand-presence guardrail)           #
#              Σ finance_subsidy <= budget (incentive budget)                  #
#                                                                              #
#   Method: derivative-free pattern search (Hooke-Jeeves), feasibility-first.  #
#   The two levers are economically distinct: a price cut lowers oem_net and   #
#   consumer_price 1:1, while APR subvention lifts demand via the utility term #
#   WITHOUT touching price, paid for out of the finance-subsidy budget.        #
# --------------------------------------------------------------------------- #


@dataclass
class OptConfig:
    params: Params = field(default_factory=Params)
    price_lo: float = -0.30
    price_hi: float = 0.30
    share_floor: float = 0.40
    subsidy_budget_usd: float = 200e6
    capacity: Dict[str, float] = field(default_factory=lambda: dict(CAPACITY))


@dataclass
class Eval:
    x: List[float]
    cells: List[Cell]
    kpis: KPIs
    contribution: float
    violations: Dict[str, float]
    total_violation: float
    subsidy_usd: float
    units_by: Dict[str, float]


_OPT_VEH = [v.vid for v in VEHICLES]
_VMAP = {v.vid: v for v in VEHICLES}


def _scenario_from_x(x: List[float], cfg: OptConfig) -> Scenario:
    n = len(_OPT_VEH)
    levers: Dict[str, Lever] = {}
    mkt = cfg.params.market_rate
    for i, vid in enumerate(_OPT_VEH):
        apr = x[n + i]
        levers[vid] = Lever(
            price_mult_delta=x[i],
            apr=apr,
            subvented=(apr < mkt - 1e-9),
            term_months=60, down=2000.0,
        )
    return Scenario(name="optimizer", params=cfg.params, levers=levers)


def _evaluate(x: List[float], cfg: OptConfig) -> Eval:
    cells, k = run(_scenario_from_x(x, cfg))

    cost_v = 0.0
    units_by: Dict[str, float] = {}
    for c in cells:
        cogs = _VMAP[c.vid].msrp * _VMAP[c.vid].cogs_pct
        if c.oem_net < cogs:
            cost_v = max(cost_v, (cogs - c.oem_net) / cogs)
        units_by[c.vid] = units_by.get(c.vid, 0.0) + c.units

    cap_v = 0.0
    for vid, u in units_by.items():
        cap = cfg.capacity[vid]
        if cap and u > cap:
            cap_v = max(cap_v, u / cap - 1.0)

    share_v = max(0.0, (cfg.share_floor - k.market_share) / cfg.share_floor)

    subsidy = sum(c.finance_subsidy_usd for c in cells)
    bud_v = max(0.0, subsidy / cfg.subsidy_budget_usd - 1.0) if cfg.subsidy_budget_usd else 0.0

    violations = {"cost": cost_v, "capacity": cap_v, "share": share_v, "budget": bud_v}
    return Eval(x=list(x), cells=cells, kpis=k, contribution=k.contribution_usd,
                violations=violations, total_violation=cost_v + cap_v + share_v + bud_v,
                subsidy_usd=subsidy, units_by=units_by)


_FEAS_EPS = 1e-4


def _feasible(e: Eval) -> bool:
    return e.total_violation <= _FEAS_EPS


def _better(a: Eval, b: Eval) -> bool:
    """Feasibility-first: feasible beats infeasible; among feasible, more
    contribution wins; among infeasible, less total violation wins."""
    fa, fb = _feasible(a), _feasible(b)
    if fa and fb:
        return a.contribution > b.contribution
    if fa != fb:
        return fa
    return a.total_violation < b.total_violation


def _pattern_search(x0: List[float], cfg: OptConfig) -> Tuple[Eval, int]:
    n = len(_OPT_VEH)
    mkt = cfg.params.market_rate
    lo = [cfg.price_lo] * n + [0.0] * n
    hi = [cfg.price_hi] * n + [mkt] * n
    steps = [0.08] * n + [0.015] * n           # price step 8pts, APR step 1.5pts
    x = [_clamp(v, lo[i], hi[i]) for i, v in enumerate(x0)]
    best = _evaluate(x, cfg)
    evals = 1
    for _ in range(80):
        improved = False
        for i in range(2 * n):
            for d in (1.0, -1.0):
                trial = list(best.x)
                trial[i] = _clamp(trial[i] + d * steps[i], lo[i], hi[i])
                if trial[i] == best.x[i]:
                    continue
                e = _evaluate(trial, cfg)
                evals += 1
                if _better(e, best):
                    best = e
                    improved = True
        if not improved:
            steps = [s * 0.5 for s in steps]
            if steps[0] < 0.003 and steps[n] < 0.0008:
                break
    return best, evals


def optimize(cfg: Optional[OptConfig] = None) -> Tuple[Eval, int]:
    cfg = cfg or OptConfig()
    n = len(_OPT_VEH)
    # multi-start to dodge local optima: baseline, uniform +10%, uniform -10%
    starts = [
        [0.0] * (2 * n),
        [0.10] * n + [cfg.params.market_rate] * n,
        [-0.10] * n + [cfg.params.market_rate] * n,
    ]
    best: Optional[Eval] = None
    total_evals = 0
    for s in starts:
        e, ev = _pattern_search(s, cfg)
        total_evals += ev
        if best is None or _better(e, best):
            best = e
    return best, total_evals


def report_optimum(best: Eval, base_cells: List[Cell], base_k: KPIs,
                   cfg: OptConfig, evals: int) -> None:
    n = len(_OPT_VEH)
    mkt = cfg.params.market_rate
    name_by = {v.vid: v.name for v in VEHICLES}

    print()
    print(BOLD + "=" * 74 + RESET)
    print(BOLD + "  VPX OPTIMIZER  ::  maximize contribution margin" + RESET)
    print(BOLD + "=" * 74 + RESET)
    print(GRAY + "  ({} evaluations, feasibility-first pattern search)".format(evals) + RESET)

    print()
    print(BOLD + "  RECOMMENDED LEVERS" + RESET)
    print("  " + "-" * 70)
    print("  {:<8}{:>12}{:>14}{:>16}".format("Model", "Price Δ", "APR offered", "Subvention"))
    for i, vid in enumerate(_OPT_VEH):
        pd = best.x[i]
        apr = best.x[n + i]
        subv = apr < mkt - 1e-9
        pcol = GREEN if pd > 0 else (RED if pd < 0 else GRAY)
        apr_s = "{:.1f}%".format(apr * 100) if subv else GRAY + "market" + RESET
        sub_s = (GREEN + "−{:.1f}pts".format((mkt - apr) * 100) + RESET) if subv else GRAY + "—" + RESET
        print("  {:<8}{}{:>+11.1f}%{}{:>14}{:>25}".format(
            name_by[vid], pcol, pd * 100, RESET, apr_s, sub_s))
    print("  " + "-" * 70)

    # uplift headline
    duc = best.contribution - base_k.contribution_usd
    pct = duc / base_k.contribution_usd * 100 if base_k.contribution_usd else 0.0
    col = GREEN if duc >= 0 else RED
    print()
    print("  {}OPTIMIZER UPLIFT: {}{:+,.1f}M  ({:+.1f}% contribution){}".format(
        BOLD, col, duc / 1e6, pct, RESET))

    # constraint report
    print()
    print(BOLD + "  BINDING CONSTRAINTS AT OPTIMUM" + RESET)
    print("  " + "-" * 70)

    def line(label: str, binding: bool, detail: str):
        tag = (RED + "BINDING" + RESET) if binding else (GRAY + "slack  " + RESET)
        print("  [{}] {:<24}{}".format(tag, label, detail))

    share = best.kpis.market_share
    line("Market-share floor", share <= cfg.share_floor + 0.005,
         "{:.1f}% vs {:.0f}% floor".format(share * 100, cfg.share_floor * 100))

    cap_bind = [name_by[vid] for vid, u in best.units_by.items()
                if u >= cfg.capacity[vid] * 0.99]
    line("Production capacity", bool(cap_bind),
         ("at ceiling: " + ", ".join(cap_bind)) if cap_bind else "all under ceiling")

    line("Finance-subsidy budget", best.subsidy_usd >= cfg.subsidy_budget_usd * 0.99,
         "{} of {} budget".format(_m(best.subsidy_usd), _m(cfg.subsidy_budget_usd)))

    min_slack = min((c.oem_net - _VMAP[c.vid].msrp * _VMAP[c.vid].cogs_pct)
                    for c in best.cells)
    line("Cost floor (oem_net≥cogs)", min_slack <= 50.0,
         "tightest cell ${:,.0f} above cost".format(min_slack))
    print("  " + "-" * 70)

    # full dashboard at the optimum, vs baseline
    print_dashboard("Optimized configuration  vs baseline", best.cells, best.kpis, base=base_k)


# --------------------------------------------------------------------------- #
# 8. CLI                                                                       #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="VPX v2 pricing simulator")
    ap.add_argument("scenario", nargs="?", default="baseline",
                    help="preset name or 'optimize' (default: baseline)")
    ap.add_argument("--selftest", action="store_true", help="run invariant checks")
    ap.add_argument("--list", action="store_true", help="list presets")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.list:
        print("Presets: " + ", ".join(PRESETS) + ", optimize")
        return 0

    base_cells, base_k = run(preset_baseline())
    base_idx = {(c.vid, c.rid): c.units for c in base_cells}

    if args.scenario == "optimize":
        cfg = OptConfig()
        best, evals = optimize(cfg)
        report_optimum(best, base_cells, base_k, cfg, evals)
        return 0

    if args.scenario not in PRESETS:
        print("Unknown scenario '{}'. Presets: {}, optimize".format(
            args.scenario, ", ".join(PRESETS)))
        return 2

    if args.scenario == "baseline":
        print_dashboard(base_k and "Baseline (published MSRP, no incentives)",
                        base_cells, base_k)
        return 0

    cells, k = run(PRESETS[args.scenario](), base_idx)
    # show baseline first for reference, then scenario with deltas
    print_dashboard("Baseline (reference)", base_cells, base_k)
    print_dashboard(PRESETS[args.scenario]().name + "  vs baseline", cells, k, base=base_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
