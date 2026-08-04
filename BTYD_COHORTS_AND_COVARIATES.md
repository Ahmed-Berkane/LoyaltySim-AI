# BTYD Probabilistic Models: Cohorts, Seasonality, Holidays & Macro Covariates

Research notes on how Buy-Till-You-Die (BTYD) models — the family this project's
`Notebooks/01-clv.ipynb` (BG/NBD + Gamma-Gamma via `scripts/clv_weekly.py`) belongs to —
handle acquisition cohorts, and how the academic literature (starting from
[Bruce Hardie's site](https://www.brucehardie.com/)) and Theta's commercial "CLV Ultra"
extend these models with time-invariant and time-varying covariates (seasonality,
holidays, macro shocks). Ends with a concrete, staged recipe for this repo.

> Compiled Aug 4, 2026. All claims are sourced — see [§6 References](#6-references).

---

## TL;DR

1. **Classic BG/NBD and Pareto/NBD do not model cohorts at all.** They assume one
   stationary population-level mixing distribution for everyone, and "cohort" only
   enters implicitly through each customer's `T` (age since acquisition). This
   project's `01-clv.ipynb` fits exactly this classic, cohort-blind version — which
   is the correct, defensible baseline, but it's the *starting point* in the
   literature, not the end point.
2. **Covariates were bolted onto BTYD models in stages over ~15 years**, roughly:
   time-invariant customer covariates (Fader & Hardie 2007) → time-varying covariates
   via proportional hazards (Fader & Hardie's unpublished notes, then Bachmann,
   Meierer & Näf 2021) → hierarchical Bayes that relaxes independence assumptions and
   supports MCMC-based covariates (Abe 2009) → fully joint, cross-cohort hierarchical
   models (what Theta's **CLV Ultra** appears to be, per their own blog posts — no
   paper/code is public, so this is informed inference, clearly labeled as such below).
3. This project **already has a simplified, empirical version of the "time-varying
   covariate" idea** in `scripts/clv_weekly.py` (`fit_month_seasonality` /
   `apply_month_seasonality` — a post-hoc calendar-month multiplier fit on the
   validation split). §4 shows how to make this more rigorous and how to extend the
   same idea to holidays and macro variables that already exist in the pipeline
   (`uci_context.py`) but are currently unused by the CLV model.
4. `lifetimes` (this project's current library) **has no covariate support**. Moving
   past time-invariant/time-varying covariates requires `pymc-marketing`
   (Python, Bayesian) or `CLVTools`/`BTYDplus` (R). Trade-offs in §4.5.

---

## 1 · How classic BTYD models treat "cohorts"

The two workhorse noncontractual models — **Pareto/NBD** (Schmittlein, Morrison &
Colombo 1987) and **BG/NBD** (Fader, Hardie & Lee 2005, the model this project uses via
`lifetimes.BetaGeoFitter`) — share the same skeleton:

- While "alive," a customer's transactions follow a **Poisson process** with rate `λ`.
- `λ` varies **across customers** according to a `Gamma(r, α)` mixing distribution
  (population heterogeneity), but is assumed **constant over time for a given
  customer** (this is the "stationary" assumption `MODELING.md` §1 calls out
  explicitly: *"Stationary weekly BG/NBD — no seasonality overlays"*).
- Customers "die" (permanently stop buying) at some point; Pareto/NBD models this with
  a continuous-time exponential dropout rate `μ ~ Gamma(s, β)`, BG/NBD approximates it
  with a discrete "coin flip after every transaction," `p ~ Beta(a, b)`.
- Critically, **`(r, α, a, b)` are estimated once, for the whole customer base.**
  There is no cohort index anywhere in the likelihood.

**Where a cohort effect *does* sneak in:** every customer's sufficient statistics —
frequency `x`, recency `t_x`, and age `T` (this project builds these via
`lifetimes.utils.calibration_and_holdout_data` in `clv_weekly.build_summary`) — are
computed relative to *their own* acquisition date. So a customer acquired late in the
calibration window naturally has a small `T` and gets appropriately wide/uncertain
predictions purely through the math of the Beta/Gamma mixture — **not** because the
model knows anything about "young cohorts behaving differently." If younger cohorts
really *are* systematically different (worse or better quality, different seasonality
exposure, acquired via different channels/promotions), the single-population BG/NBD
is blind to that. This is the exact gap Theta's marketing repeatedly targets (§3).

### 1.1 The two traditional ways cohorts get incorporated

| Approach | How | Trade-off |
|---|---|---|
| **Segment-then-model**: fit a separate BG/NBD per acquisition cohort | Split customers by `cohort` (e.g. acquisition quarter — this project already computes this in `scripts/customer_base_audit.py`), fit `BetaGeoFitter` independently per cohort | Simple, but each cohort model only sees its own data — young cohorts get noisy, wide-uncertainty parameter estimates ("cold start"), and cannot borrow strength from older, more data-rich cohorts |
| **Aggregate cohort-level modeling (CBCV)**: McCarthy, Fader & Hardie's *Customer-Based Corporate Valuation* framework | Model each cohort's **aggregate** retention curve (e.g. shifted-Beta-Geometric survival function) and aggregate spend, rather than individual-level RFM; cohorts are explicit units of analysis, and a cross-cohort trend term lets acquisition quality drift over time | Doesn't require individual-level data (useful when only period-level/public financials are available — the whole point of CBCV) but is a much coarser lens than customer-level BG/NBD |
| **Joint hierarchical model** (modern answer, §2.5 / §3) | Fit *all* cohorts simultaneously in one hierarchical Bayesian model, with cohort-level parameters partially pooled toward a population mean | Best of both worlds — young cohorts borrow strength, cross-cohort trends are still identifiable — but heavier to build/estimate (MCMC, no closed form) |

---

## 2 · Adding covariates to BTYD models — literature, in order of sophistication

### 2.1 Time-invariant covariates (static, customer-level)

**Source:** Fader, P. S. & Hardie, B. G. (2007), *"Incorporating Time-Invariant
Covariates into the Pareto/NBD and BG/NBD Models,"*
[Note 019, brucehardie.com](https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf).

The trick: replace the fixed scale parameters of the Gamma/Beta mixing distributions
with a log-linear function of each customer's covariate vector `x_i`, e.g.

```
α_i = α₀ · exp(−γ₁ᵀ x_i)      (transaction-rate scale)
β_i  = β₀ · exp(−γ₂ᵀ x_i)      (dropout scale, Pareto/NBD)  — or a_i, b_i for BG/NBD
```

This keeps the model closed-form (same likelihood shape, just parameterized
per-customer) and is fit by MLE alongside the shared shape parameters `r`/`a`/`b`.
This is exactly what `pymc-marketing`'s `BetaGeoModel` exposes as
`dropout_covariate_cols` / `purchase_covariate_cols`, and what R's `CLVTools`
supports out of the box for both Pareto/NBD and BG/NBD.

**Relevance to this project:** country (domestic UK vs. EU), and the synthetic CRM
fields already generated in `uci_context.py` (`tier`, `discount_sensitivity`,
`app_usage_score`, `email_opt_in`) are natural time-invariant covariate candidates —
but note the CRM fields are *synthetic* (randomly generated, not derived from real
behavior), so any lift they show in a covariate model would be an artifact, not a
real signal — flag this explicitly if pursued in a notebook.

### 2.2 Time-varying covariates — contractual/discrete settings

**Source:** Fader & Hardie, *"Incorporating Time-Varying Covariates in a Simple
Mixture Model for [Discrete] Duration-Time Data,"*
[Note 037, brucehardie.com](https://www.brucehardie.com/notes/037/time-varying_covariates_in_BG.pdf).

This note explains why you *can't* cleanly add time-varying covariates to the
Beta-Geometric (BG) survival model used for contractual churn (the shifted-BG /
sBG model from Fader & Hardie's 2007 *"How to Project Customer Retention,"*
[PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf)),
and proposes a near-identical replacement (the "G2G+covariates" model) that does
support them, built on a discretized proportional-hazards structure. Not directly
applicable to UCI (noncontractual retail), but useful background if this project
ever adds a subscription/contractual angle.

### 2.3 Time-varying covariates — noncontractual BTYD (BG/NBD, Pareto/NBD)

**Source:** Fader & Hardie, *"[Incorporating] Time-Varying Covariates in [the]
BG/NBD [Model],"* [Note 040, brucehardie.com](https://www.brucehardie.com/notes/040/time-varying_covariates_in_BGNBD.pdf).

Core idea: while a customer is "alive," inter-transaction times are exponential with
rate `λ`. Under the standard **proportional hazards** framework, a time-varying
covariate `z(t)` (shared or individual) scales that rate:

```
λ_i(t) = λ_i · exp(βᵀ z(t))
```

For the common case of **a small number of discrete "seasons"** (e.g. a "regular"
season and a "high" season represented by dummy variables), the note shows the
likelihood collapses to something you can still evaluate in closed form: instead of
raw calendar time, you replace "recency" and "age" with an **exposure-time**
transform that accumulates real time faster during high-covariate periods. This is
the closest published analogue to seasonality-as-covariate for BG/NBD specifically,
and it generalizes to "more than two seasons" (their example extends low/regular/high
season to three z-values).

Follow-on / complementary work:

- **Bachmann, Meierer & Näf (2021)**, extending Pareto/NBD with genuinely
  time-varying covariates (both aggregate — affecting all customers at once, e.g.
  seasonality/macro — and customer-specific — e.g. individual promo exposure),
  implemented as `pnbd_dyncov` in the R package **CLVTools**
  ([CRAN](https://doi.org/10.32614/cran.package.clvtools)).
- A non-Hardie extension, *"Incorporating time-dependent covariates into BG-NBD
  model for churn prediction in non-contractual settings"*
  ([SSRN](http://steppechange.com/wp-content/uploads/2017/06/SSRN-id2905307.pdf)),
  conditions the per-transaction dropout probability on covariates known up to that
  transaction (rolling avg/min/max inter-purchase gap and spend) — more flexible,
  but loses closed-form tractability and needs numerical optimization.
- **Platzer & Reutterer (2016)**, *"Ticking Away the Moments"* — adds a "purchase
  regularity" parameter (Erlang-n inter-purchase times instead of exponential); not a
  covariate per se, but a commonly-cited complementary extension available in
  `CLVTools` as the `pnbd` "regularity" option.
- A 2019 University of Iowa dissertation, *"Stochastic Process Customer Lifetime
  Value Models with Time-Varying Covariates"*
  ([PDF](https://iro.uiowa.edu/view/pdfCoverPage?download=true&filePid=13730821500002771&instCode=01IOWA_INST)),
  is a useful (if dense) survey of exactly this problem and makes an important
  identification point echoed by Theta's own writing (§3): **you generally need ≥2
  years of data to statistically separate a recurring "seasonal" effect from a
  one-off "shock"** — with less than that, the two are confounded.

### 2.4 Hierarchical Bayes: relaxing independence + adding covariates via MCMC

**Source:** Abe, M. (2009), *"'Counting Your Customers' One by One: A Hierarchical
Bayes Extension to the Pareto/NBD Model,"* Marketing Science 28(3), 541–553
([DOI](https://doi.org/10.1287/mksc.1090.0502)).

Two things at once: (1) it drops the standard BTYD assumption that a customer's
transaction rate `λ` and dropout rate `μ` are independent, instead drawing
`(log λ, log μ)` from a **bivariate log-normal**, which can be regressed on
covariates; (2) estimation is via **MCMC**, not MLE, which is what makes the
covariate regression tractable at all. Implemented in R's **BTYDplus** package as
`abe.mcmc.DrawParameters(cal.cbs, covariates = c(...))`
([docs](https://rdrr.io/cran/BTYDplus/man/abe.mcmc.DrawParameters.html)).

**Netzer, Lattin & Srinivasan (2008)**, *"A Hidden Markov Model of Customer
Relationship Dynamics"* ([PDF](https://columbia.edu/~on2110/Papers/HMM_of_Customer_Relationship_Dynamics.pdf)),
takes a structurally different approach worth knowing about: instead of a
continuously-varying hazard rate, it posits a small number of **latent relationship
states** (e.g. strong/weak/dormant), with **time-varying covariates driving the
transition probabilities between states**, estimated with hierarchical Bayes for
customer heterogeneity. This is a natural way to think about a "recession regime"
as a state that shifts *transition* probabilities rather than a continuous
multiplier on a hazard rate — conceptually elegant for macro shocks, but a heavier
model to stand up than the proportional-hazards extensions above.

### 2.5 Explicit hierarchical cohort structure

`pymc-marketing`'s hierarchical **sBG (shifted-Beta-Geometric)** implementation
([docs](https://www.pymc-marketing.io/en/stable/notebooks/clv/sbg.html)) is the
clearest modern illustration of "cohorts done properly": the *original* Fader &
Hardie sBG model required a **separate fit per cohort**; `pymc-marketing` instead
fits **all cohorts in one hierarchical Bayesian model**, with cohort-level Beta
shape parameters `(α_cohort, β_cohort)` partially pooled toward population-level
hyperpriors — and covariates can be layered on top via the same
`exp(−γᵀz)` link as §2.1. This is a contractual-retention model (not BG/NBD/Pareto-NBD),
but the *pooling* idea generalizes directly: it is the mechanism that lets a
1-month-old cohort's estimate borrow statistical strength from a 2-year-old
cohort's, instead of being estimated in isolation or blindly pooled with everyone.

---

## 3 · What Theta's "CLV Ultra" appears to do

Theta doesn't publish CLV Ultra's model or code, but their own posts (co-founder
Daniel McCarthy and team) describe the design goals and behavior in enough detail to
reconstruct the shape of it — this section is **inference from their public
marketing writing**, not a paper, and is flagged as such throughout.

- It is explicitly a **"generative"** model (in the BTYD/probabilistic tradition),
  not a discriminative ML model (random forest / gradient boosting on tabular
  features — i.e., not this project's `02-clv-enterprise.ipynb` foil). Their stated
  reasons, which double as a good critique to keep in mind for `02`:
  1. Discriminative models can only train on customers old enough to have an
     observed outcome (e.g. 1-year-ahead sales) — young cohorts, the ones you most
     need signal on, are excluded from training by construction.
  2. Longer prediction horizons throw away even more training data.
  3. Fitting separate models per horizon/target produces internally
     *inconsistent* predictions (e.g. purchase-count and probability-of-activity
     forecasts that don't agree).
  4. They tend to mis-track aggregate totals over time, with error compounding.
  
  (Source: [*"CLV Ultra: Our breakthrough new CLV model"*](https://thetaclv.com/resource/clv-ultra-breakthrough-new-clv-model/), Jan 2024.)

- **Joint estimation across all cohorts at once** (not per-cohort refits, not a
  single pooled fit either) is repeatedly called out as the key mechanism that lets
  young cohorts borrow strength from older ones while still allowing genuine
  cross-cohort differences in acquisition quality — precisely the hierarchical
  pooling idea in §2.5, generalized from a contractual retention model to a full
  transaction+spend BTYD model.

- They explicitly decompose fitted behavior into **four disentangled components**:
  "baseline goodness" (customer/cohort quality), **seasonality** (recurring,
  calendar-driven), **non-seasonal shocks** (one-off — inflation, interest-rate
  hikes, natural disasters, pandemics, new competitors), and **cross-cohort trend**.
  They note this decomposition is only statistically identifiable with enough
  history to distinguish "recurs every year" from "happened once" — the same point
  the Iowa dissertation (§2.3) makes academically.

- A **second stated pillar is customer *lifecycle* covariates**, explicitly
  distinguished from calendar seasonality:
  - **Tenure effects** — behavior that ramps up/down as a function of *time since
    acquisition* regardless of calendar date (e.g. cautious spending right after
    signup, then increasing as trust builds).
  - **Anniversary effects** — bumps tied to a customer's *own* acquisition
    anniversary (relevant mostly for subscription businesses), independent of the
    calendar season.
  
  (Source: [*"Customer Lifecycle Effects for a Better CLV Model"*](https://thetaclv.com/resource/clv-models-with-customer-lifecycle-covariates/), Mar 2023.)

- They frame this whole program as **"nothing new academically, engineered well"**:
  the building blocks are time-invariant/time-varying covariates (§2.1–2.3),
  relaxed independence + MCMC-based covariate regression (§2.4, Abe 2009 — one of
  Theta's own co-founders' academic lineage runs directly through this literature),
  and joint cross-cohort/hierarchical pooling (§2.5) — assembled into one production
  system with enough numerical-optimization work to make it fast at scale (their
  claim: hundreds of millions of customers, minutes of runtime).

---

## 4 · Concrete recipe for this project

### 4.1 What's already sitting in the pipeline, unused by the CLV model

`scripts/uci_context.py` (restored to `HEAD` earlier in this project) already computes
a rich covariate set that lands in `uci_fact_transactions.parquet` but is **not**
currently fed into `clv_weekly.py`'s BG/NBD+GG fit:

| Category | Columns already in the fact table |
|---|---|
| Calendar / seasonality | `is_public_holiday`, `is_retail_spending_day`, `day_of_week`, `is_friday`, `is_month_start`/`is_month_end`, `is_christmas_season`, `is_back_to_school`, `is_black_friday_week`, `is_cyber_monday_week`, `is_amazon_prime_day`, `days_to_christmas`, `days_to_black_friday`, `is_major_sale_period` |
| Country-specific holidays | `is_sinterklaas`, `is_three_kings_day`, `is_italian_epiphany`, `is_boxing_day`, `is_polish_childrens_day`, `is_french_winter_sale`, `is_french_summer_sale` |
| Macro (by country × month) | `cpi_index`, `inflation_mom`, `inflation_yoy`, `unemployment_rate`, `interest_rate` |
| Weather (by country × date) | `temp_c`, `rain_mm`, `snow_cm`, `precip_mm`, `weather_code`, `wind_gust_kmh`, `had_rain`, `had_snow`, `had_major_storm` |
| Synthetic CRM (time-invariant, per customer) | `tier`, `points_balance`, `email_opt_in`, `app_usage_score`, `discount_sensitivity` (⚠️ randomly generated — see caveat in §2.1) |

### 4.2 Stage 0 — already done: empirical seasonal overlay

`scripts/clv_weekly.py::fit_month_seasonality` / `apply_month_seasonality` /
`evaluate_seasonal_overlay` already implement a **simplified, empirical version** of
Note 040's idea: fit a per-calendar-month multiplier as
`Σ actual / Σ stationary-predicted` on the **validation** split only, clip to
`[0.5, 2.5]`, then apply to the stationary BG/NBD path on **test**, and compare
aggregate £/order error against the plain stationary model. This is a legitimate,
leakage-safe, easy-to-explain approximation — worth explicitly documenting in
`MODELING.md` as "Stage 0" of a seasonality roadmap, since right now it reads as a
one-off experiment rather than the first rung of a deliberate ladder.

### 4.3 Stage 1 — make seasonality/holidays part of the likelihood, not a post-hoc ratio

Rather than fitting a ratio after the fact, follow Note 040 directly: treat each
`is_*` seasonal/holiday flag as a **shared, aggregate time-varying covariate**
`z(t)` and fit `λ_i(t) = λ_i · exp(βᵀ z(t))` jointly with `(r, α, a, b)` via MLE (or
Bayesian MCMC via `pymc-marketing`). Since `lifetimes` has no covariate hook, the
practical path is:

- **Easiest:** switch the purchase-process fit from `lifetimes.BetaGeoFitter` to
  `pymc_marketing.clv.BetaGeoModel`, which supports `purchase_covariate_cols`
  reusing the same `(frequency, recency, T)` summary this project already builds —
  aggregate the `is_*` flags into per-customer-week exposure features (e.g. "share
  of calibration weeks that were Christmas-season weeks") the way Note 040's
  worked example does.
- **More faithful to Note 040, more work:** re-derive the closed-form seasonal
  likelihood directly (finite number of discrete seasons) and fit with `scipy.optimize`,
  keeping the whole pipeline in the current MLE/`lifetimes`-adjacent style rather than
  moving to a Bayesian stack.
- Either way, **keep this project's existing validation discipline**: fit any new
  covariate coefficients on calibration/validation only, gate on the unseen test
  aggregate ≤5% error exactly like today, and extend `audit_no_leakage()` to assert
  the covariate-fitting step never touches the test window.

### 4.4 Stage 2 — macro covariates as *aggregate*, not individual, time-varying effects

`cpi_index`, `inflation_yoy`, `unemployment_rate`, `interest_rate` vary by
`(country, month)`, not by individual customer — they are exactly the "aggregate
time-varying covariate" case Bachmann/Meierer/Näf's Pareto/NBD extension and Note
040's seasonal-dummy simplification both handle (same machinery as Stage 1, just a
different `z(t)`, optionally interacted with `country` for non-UK customers).

**Important caveat specific to this dataset:** UCI Online Retail II only spans
~Dec 2009 – Dec 2011 (roughly 13–25 months depending on the vintage used here). Per
§2.3/§3, reliably separating a *macro* effect from *seasonal* or *cohort-quality*
effects statistically needs more history than that, and this window barely covers
the tail of the 2008–09 recession/recovery — real macro variation is thin. Practical
recommendation: **use coarse regime dummies (e.g. "recession-adjacent quarter"
flag) instead of continuous `inflation_yoy`/`unemployment_rate` coefficients**, and
always validate the resulting test-set error against the no-macro baseline before
trusting the sign/magnitude of any fitted coefficient — treat this as a
methodology demonstration more than a claim about UK retail elasticity to inflation.

### 4.5 Stage 3 — time-invariant customer covariates

Layer in `country` (domestic vs. EU) and, with the caveat that they're synthetic,
`tier`/`discount_sensitivity` as `dropout_covariate_cols`/`purchase_covariate_cols`
on top of whatever Stage 1 model is chosen, following Note 019 / §2.1 directly —
this is the cheapest, most literature-standard extension and a natural first PR
before attempting time-varying covariates at all, if sequencing by implementation
cost rather than by conceptual generality.

### 4.6 Stage 4 (optional/ambitious) — explicit cross-cohort hierarchy

This project already computes an acquisition `cohort` (acquisition quarter) in
`scripts/customer_base_audit.py` for the Lens 3–5 descriptive views in
`Notebooks/00-customer-base-audit.ipynb`, but `01-clv.ipynb`'s BG/NBD never
conditions on it. Two ways to close that gap, cheapest first:

1. **Cohort as a covariate** — one-hot/ordinal-encode `cohort` and feed it through
   the same `purchase_covariate_cols`/`dropout_covariate_cols` mechanism as Stage 3
   (fast to try, doesn't require touching the model structure).
2. **True hierarchical pooling** — mirror `pymc-marketing`'s hierarchical sBG
   pattern (§2.5): let each cohort have its own `(r, α, a, b)` (or covariate
   coefficients) drawn from shared population hyperpriors, fit via MCMC. This is
   the closest open, buildable analogue to what Theta's cross-cohort/cold-start
   story (§3) describes, and would make a strong "Enterprise foil v2" companion to
   the existing `02-clv-enterprise.ipynb`, explicitly contrasted with both the
   plain BG/NBD (`01`) and the HistGradientBoosting foil (`02`) on the same
   45/20/35 + ≤5% gate protocol.

### 4.7 Tooling comparison

| Library | Language | Time-invariant covariates | Time-varying covariates | Hierarchical/cohort pooling | Notes |
|---|---|---|---|---|---|
| `lifetimes` (current) | Python | ❌ | ❌ | ❌ | What `clv_weekly.py` uses today; fast, simple, no covariate hooks |
| `pymc-marketing` | Python (PyMC/Bayesian) | ✅ (`*_covariate_cols`) | Partial — via custom `z(t)` features, or hierarchical sBG for contractual | ✅ (hierarchical sBG demo; BG/NBD hierarchy is buildable) | Best Python option; MCMC = slower, more setup, but keeps everything in-repo/Python |
| `CLVTools` (R) | R | ✅ | ✅ (`pnbd_dyncov`, Bachmann/Meierer/Näf) | Partial | Most complete published implementation of time-varying Pareto/NBD; would need `rpy2` bridge |
| `BTYDplus` (R) | R | ✅ (Abe HB) | ❌ (not this package) | Partial (Abe HB per-cohort) | Good for the Abe (2009) HB/MCMC extension specifically |
| Custom MLE (`scipy.optimize`) | Python | ✅ | ✅ (Note 040 closed-form case) | ❌ (needs hand-rolled hierarchy) | Most faithful to Hardie's derivations, most implementation effort |

---

## 5 · Caveats and risks

- **Identification, not just estimation.** With ~1–2 years of UCI data, don't expect
  to cleanly separate "seasonal," "macro," and "cohort quality" effects the way a
  multi-year enterprise dataset would allow (§2.3, §3). Report this limitation
  alongside any covariate results rather than treating fitted coefficients as
  ground truth.
- **`lifetimes` has no upgrade path for covariates.** Any of Stage 1–4 above
  requires either adopting `pymc-marketing` (recommended — stays in Python) or
  bridging to R (`CLVTools`/`BTYDplus`) — this is a real dependency/tooling
  decision, not a one-line change.
- **Complexity budget.** Theta's own writing stresses that covariates add "both
  mathematical and computational complexity," and that time-varying covariates in
  BTYD models were not even theoretically tractable in a general form until
  recently. For a research/portfolio project, Stage 0 (already shipped) and Stage 1
  (seasonal/holiday dummies via Note 040) are the highest value-to-effort additions;
  treat Stage 2–4 as stretch goals, gated the same ≤5% aggregate-error way
  `MODELING.md` already gates `01`/`02`.
- **Synthetic CRM fields are not real signal.** `tier`, `discount_sensitivity`,
  `app_usage_score`, `email_opt_in` are generated by `generate_synthetic_crm()` in
  `uci_context.py`, not observed behavior — useful for demonstrating covariate
  *mechanics* but any "insight" from their coefficients is definitionally circular
  and should be labeled as such in any notebook that uses them.

---

## 6 · References

**Bruce Hardie's site / core BTYD papers & notes**

- Home page — [brucehardie.com](https://www.brucehardie.com/)
- Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). *"Counting Your Customers the
  Easy Way: An Alternative to the Pareto/NBD Model."* (BG/NBD — the model this
  project uses via `lifetimes.BetaGeoFitter`.)
- Fader, P. S. & Hardie, B. G. S. (2007). *"How to Project Customer Retention."*
  Journal of Interactive Marketing, 21(1), 76–90. sBG model.
  [PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_07.pdf)
- Fader, P. S. & Hardie, B. G. S. (2007). *"Incorporating Time-Invariant Covariates
  into the Pareto/NBD and BG/NBD Models."* Note 019.
  [PDF](https://www.brucehardie.com/notes/019/time_invariant_covariates.pdf)
- Fader, P. S. & Hardie, B. G. S. *"Incorporating Time-Varying Covariates in a
  Simple Mixture Model for Discrete Duration-Time Data."* Note 037.
  [PDF](https://www.brucehardie.com/notes/037/time-varying_covariates_in_BG.pdf)
- Fader, P. S. & Hardie, B. G. S. *"[Incorporating] Time-Varying Covariates in the
  BG/NBD Model."* Note 040.
  [PDF](https://www.brucehardie.com/notes/040/time-varying_covariates_in_BGNBD.pdf)

**Hierarchical Bayes / HMM extensions**

- Abe, M. (2009). *"'Counting Your Customers' One by One: A Hierarchical Bayes
  Extension to the Pareto/NBD Model."* Marketing Science, 28(3), 541–553.
  [DOI](https://doi.org/10.1287/mksc.1090.0502) ·
  [BTYDplus implementation](https://rdrr.io/cran/BTYDplus/man/abe.mcmc.DrawParameters.html)
- Netzer, O., Lattin, J. M., & Srinivasan, V. (2008). *"A Hidden Markov Model of
  Customer Relationship Dynamics."* Marketing Science, 27(2), 185–204.
  [PDF](https://columbia.edu/~on2110/Papers/HMM_of_Customer_Relationship_Dynamics.pdf)

**Cohort-level / CBCV aggregate modeling**

- McCarthy, D., Fader, P. S., & Hardie, B. G. S. — *Customer-Based Corporate
  Valuation* methodology papers: *"Valuing Non-Contractual Firms Using Common
  Customer Metrics"* ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2923466))
  and *"Customer-Based Corporate Valuation for Publicly Traded Non-Contractual
  Firms"* ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3040422)).
- Gregory Faletto, [walkthrough of the McCarthy/Fader/Hardie retention model](https://gregoryfaletto.com/2018/11/20/the-mccarthy-fader-hardie-model-for-customer-retention/)
  (accessible derivation of the aggregate cohort survival integral).

**Time-varying covariates in BG/NBD & Pareto/NBD (post-Hardie extensions)**

- *"Incorporating time-dependent covariates into BG-NBD model for churn prediction
  in non-contractual settings."*
  [PDF](http://steppechange.com/wp-content/uploads/2017/06/SSRN-id2905307.pdf)
- Platzer, M. & Reutterer, T. (2016). *"Ticking Away the Moments: Timing
  Regularity Helps to Better Predict Customer Activity."* Marketing Science.
- Harman, D. M. (2019). *"Stochastic Process Customer Lifetime Value Models with
  Time-Varying Covariates."* University of Iowa dissertation.
  [PDF](https://iro.uiowa.edu/view/pdfCoverPage?download=true&filePid=13730821500002771&instCode=01IOWA_INST)
- Bachmann, P., Meierer, M., & Näf, J. — time-varying-covariate Pareto/NBD, as
  implemented in R's **CLVTools**.
  [CLVTools on CRAN](https://doi.org/10.32614/cran.package.clvtools) ·
  [CLVTools arXiv paper](https://doi.org/10.48550/arxiv.2602.09845)

**Software**

- [`pymc-marketing` CLV module docs](https://www.pymc-marketing.io/en/stable/notebooks/clv/sbg.html) —
  hierarchical sBG across cohorts + covariates via `dropout_covariate_cols`.
- [`pymc-labs` blog: "More Accurate CLV Forecasts Using Hierarchical Bayes"](https://www.pymc-labs.com/blog-posts/hierarchical_clv) —
  practical discussion of the "fit one model per cohort" workaround for
  seasonality vs. a joint hierarchical model.

**Theta / CLV Ultra (commercial, no public model/code — inference from their own
public writing only)**

- McCarthy, D. (2024). *"CLV Ultra: Our breakthrough new CLV model."*
  [thetaclv.com](https://thetaclv.com/resource/clv-ultra-breakthrough-new-clv-model/)
- Anderson, E. (2023). *"Customer Lifecycle Effects for a Better CLV Model
  (Series: Part 2)."* [thetaclv.com](https://thetaclv.com/resource/clv-models-with-customer-lifecycle-covariates/)
- [*"CLV Ultra: Maximize Your ROI"*](https://thetaclv.com/clv-ultra/) — product page.
- [*"Customer Lifetime Value Analytics as a Critical Strategic Asset in Private
  Equity"*](https://thetaclv.com/resource/pe-clv-ultra/) — cross-cohort / cold-start framing.

---

## Appendix: mapping onto this repo's files

| Concept above | This repo |
|---|---|
| Classic stationary BG/NBD (§1) | `scripts/clv_weekly.py::fit_purchase_model`, `Notebooks/01-clv.ipynb` |
| Acquisition cohort (currently descriptive-only) | `scripts/customer_base_audit.py` (`cohort` = acquisition quarter), `Notebooks/00-customer-base-audit.ipynb` Lenses 3–5 |
| Stage 0 empirical seasonal overlay (already shipped) | `scripts/clv_weekly.py::fit_month_seasonality`, `apply_month_seasonality`, `evaluate_seasonal_overlay` |
| Unused seasonality/holiday covariates | `is_public_holiday`, `is_black_friday_week`, `is_christmas_season`, … in `scripts/uci_context.py` output |
| Unused macro covariates | `cpi_index`, `inflation_yoy`, `unemployment_rate`, `interest_rate` in `scripts/uci_context.py::load_eu_macro` |
| Unused (synthetic) time-invariant covariates | `tier`, `discount_sensitivity`, … from `scripts/uci_context.py::generate_synthetic_crm` |
| Discriminative-ML foil Theta explicitly argues against for cold-start (§3) | `Notebooks/02-clv-enterprise.ipynb`, `scripts/clv_enterprise.py` (HistGradientBoosting) — keep the critique in mind when writing that notebook's limitations section |
| Validation discipline to preserve when adding any covariate stage | `MODELING.md` §1 (45/20/35 split, val-only scale, ≤5% test gate), `scripts/clv_weekly.py::audit_no_leakage` |
