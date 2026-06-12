# Demand Modeling with Real Data — What We'd Use, and Why

> Context: the VPX prototype uses a **calibrated nested multinomial logit** fit
> on synthetic data with deliberately injected price variation. That's the right
> *skeleton*, and the calibration harness (`vpx_calibrate.py`) proves the
> estimation machinery is correct. This note explains what the **production**
> demand model would be once real market data is available — and, more
> importantly, *why*.

---

## Short version

You don't throw the logit away — you **upgrade it to a random-coefficients
(BLP-style) discrete-choice model, estimated with instruments for price.** The
logit is the correct skeleton; real data exposes two flaws, and the "right
model" is the one that fixes both.

The reference point is **Berry–Levinsohn–Pakes (1995), "Automobile Prices in
Market Equilibrium"** — the canonical differentiated-products demand model, and
its flagship application is the US auto market. Cars are the textbook case:
many differentiated products, list prices plus incentives, rich substitution.

---

## Why a different model — the two flaws real data exposes

### 1. Price endogeneity — this is the whole ballgame

In synthetic data, price variation is *exogenous* (we drew promos at random). In
real data, **prices are set by the firm in response to demand** — discounts when
demand is soft, firm pricing when it's hot, higher residuals on models with
strong resale. So observed price and quantity are jointly determined by things
you don't observe (a hot redesign, a viral review, a regional taste shift).

Run plain MLE/OLS on that and the price coefficient is biased — usually
attenuated toward zero, sometimes even **wrong-signed** (you "discover" that
higher prices sell more, because firms price up what's already selling).

**No model class fixes this on its own.** The fix is *identification*:

- **Instrument price** with variables that move cost but not demand — exchange
  rates, input/commodity prices, plant/logistics cost shifters.
- Or **BLP / Hausman / differentiation instruments** — functions of rival
  product characteristics, or the same model's price in other regions
  (Gandhi–Houde differentiation IVs are the modern refinement).
- Estimate by **GMM**, not by maximizing a naive likelihood.

### 2. IIA — plain logit's unrealistic substitution

Multinomial logit imposes *Independence of Irrelevant Alternatives*: cut the
Tundra's price and it steals share from every other vehicle in proportion to its
share — pulling as much from a Prius as from a rival full-size truck. That's
nonsense; real substitution runs along *characteristics* (truck buyers switch to
other trucks).

**The fix is random coefficients** — let consumers differ in price sensitivity
and feature tastes, drawn from a distribution you estimate. Integrating choices
over that heterogeneity produces realistic, data-driven substitution and kills
IIA. (Hard segments, as in the prototype, are a crude poor-man's version.)

---

## So: the model

**A mixed (random-coefficients) logit, BLP-style, estimated by GMM with price
instruments.** On top of the prototype's logit it adds:

- **Random coefficients** on price and key attributes → realistic
  cross-elasticities, no IIA.
- **Instruments + GMM** → a *causal* price elasticity instead of a correlation.
- **The BLP contraction mapping** to invert observed market shares into mean
  utilities each period.

If **micro data** is available — individual transactions with some buyer
attributes (loyalty / registration / finance records) — go further: estimate
choice at the *individual* level with demographics interacted with attributes
(who actually buys what), and model the **nested decision** of vehicle → trim →
finance-vs-lease-vs-cash. Micro data is worth more than any modeling cleverness.

---

## Where ML legitimately fits

ML is not the core causal demand model, but it has real, defensible roles with
real data:

- **Forecasting the nuisance pieces** — baseline category volume, seasonality,
  macro response, competitor reactions. Gradient boosting / neural nets shine
  where you want predictive accuracy and don't need a causal coefficient.
- **Double / Debiased ML (Chernozhukov et al., 2018)** — the modern bridge. Use
  flexible ML to soak up confounders, but recover a *causal* elasticity through
  Neyman-orthogonal moments + cross-fitting (DML-IV when you have instruments).
  Structural identification **plus** ML flexibility — where a sharp team lands
  today.
- **Hierarchical Bayes** — partial pooling so a low-volume trim borrows strength
  from its siblings instead of producing a garbage standalone estimate.
- **Contextual bandits / RL** — only for the *experimentation* layer, and only
  where price/incentives can actually be randomized. Most OEMs can't at the
  franchise level, so this stays narrow.

---

## The punchline

**The model class matters less than identification.** A fancy BLP estimator with
no instrument loses to a plain logit fed by a real pricing experiment or a clean
cost-shock instrument. So if you could have *one* thing with real data, it
wouldn't be a deeper model — it'd be **exogenous price variation**: randomized
regional incentive tests, natural experiments, or a credible instrument. Get
that, and even the model already in this repo starts telling the truth. Miss it,
and no architecture saves you.

That's why the calibration harness validates the *machinery* first. The day real
history with genuine price variation arrives, the path is:

```
nested logit + IV  →  random coefficients (BLP)  →  DML for the nuisance layer
```

— in that order, gated by what the data can actually identify.

---

### References

- Berry, Levinsohn, Pakes (1995). *Automobile Prices in Market Equilibrium.*
  Econometrica.
- Berry (1994). *Estimating Discrete-Choice Models of Product Differentiation.*
- Train (2009). *Discrete Choice Methods with Simulation.*
- Gandhi, Houde (2019). *Measuring Substitution Patterns in Differentiated
  Products Industries* (differentiation IVs).
- Chernozhukov et al. (2018). *Double/Debiased Machine Learning for Treatment
  and Structural Parameters.*
