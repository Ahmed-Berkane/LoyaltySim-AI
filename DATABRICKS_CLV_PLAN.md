# Databricks CLV Accelerator — What & Why

Reference plan distilled from the Databricks Industry Solutions accelerator and the `btyd` library it uses.

| Resource | Link |
|----------|------|
| Accelerator repo | [databricks-industry-solutions/customer-lifetime-value](https://github.com/databricks-industry-solutions/customer-lifetime-value) |
| Solution page | [Calculate Customer Lifetime Value](https://www.databricks.com/solutions/accelerators/customer-lifetime-value) |
| Blog Part 1 (retention) | [Estimating customer lifetimes](https://www.databricks.com/blog/2020/06/03/customer-lifetime-value-part-1-estimating-customer-lifetimes.html) |
| Blog Part 2 (spend) | [Estimating future spend](https://www.databricks.com/blog/2020/06/17/customer-lifetime-value-part-2-estimating-future-spend.html) |
| Library | [`btyd` on PyPI](https://pypi.org/project/btyd/) |
| Docs | [btyd User Guide](https://btyd.readthedocs.io/en/latest/User%20Guide.html) |

---

## 1. Business problem (why CLV)

They target **non-subscription / non-contractual retail**: customers buy when they want, with no contract end date. That creates two hard questions:

1. **Is this customer still “alive”?** — Will they buy again, or have they silently churned?
2. **How much will they spend if they stay?** — What is their expected future monetary value?

Simple averages fail because:

- Purchase frequency and spend are **skewed** (many low-activity buyers, a long tail of high-value ones).
- Averages describe the *population*, not *individuals*.
- Segment averages still miss when one customer’s behavior shifts.

CLV is the bridge between **marketing spend** and **customer equity**:

- Align CAC / retention investment with expected return
- Prefer high-potential customers over “brand transients”
- Personalize offers by lifetime potential, not just last purchase
- Monitor aggregate CLV as a health metric for the brand

**Framing:** CLV is not one model — it is **retention × spend**, discounted over a horizon.

---

## 2. What they are modeling

| Component | Question answered | Model |
|-----------|-------------------|--------|
| **Engagement / retention** | P(customer still active) and expected # of future purchases | **BG/NBD** (`BetaGeoFitter`) |
| **Monetary value** | Expected spend per purchase (given they buy) | **Gamma-Gamma** (`GammaGammaFitter`) |
| **CLV** | Discounted expected value over a future window (e.g. 12 months) | Combine both via `customer_lifetime_value()` |

This is the classic **Buy ’til You Die (BTYD)** family (Fader / Hardie), implemented with [`btyd`](https://pypi.org/project/btyd/).

### Inputs (RFM-style, not rich features)

Only four customer metrics from transactions:

| Metric | Meaning |
|--------|---------|
| **Frequency** | # of *repeat* purchase *dates* after first purchase |
| **Recency** | Age (days) at last purchase |
| **T (age/term)** | Days from first purchase to “today” (last date in data) |
| **Monetary value** | Average spend on *repeat* purchase dates |

Same-day purchases are collapsed to one event — standard for BTYD.

### Outputs

- `prob_alive` — P(still engaged)
- `purchases_next30days` — expected purchases in a window
- `clv` — 12-month discounted CLV (1% monthly discount ≈ ~12.7% annual)
- Deployable scoring via **MLflow** Spark UDF

---

## 3. The `btyd` library

[`btyd`](https://pypi.org/project/btyd/) is the successor to the [Lifetimes](https://github.com/CamDavidsonPilon/lifetimes) library for Buy Till You Die and CLV models in Python. Databricks’ accelerator notebook installs `btyd` (they pin `btyd==0.1a1`) and uses its fitters API.

### Core assumptions (from the package)

1. Users interact when they are active (“alive”).
2. Users may “die” (become inactive) after some period.

Example applications listed by the maintainers:

- Predicting how often a visitor returns to a website
- Understanding how frequently a patient returns to a hospital
- Predicting churn from an app using only usage history
- Predicting repeat purchases
- Predicting customer lifetime value

### Supported models (relevant to this accelerator)

| Model | Paper / role |
|-------|----------------|
| **BG/NBD** | Fader, Hardie & Lee (2005) — [Counting Your Customers the Easy Way](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf). Engagement / purchase count. |
| **Gamma-Gamma** | Fader & Hardie (2013) — [The Gamma-Gamma Model of Monetary Value](http://www.brucehardie.com/notes/025/gamma_gamma.pdf). Spend per transaction. |
| **Modified BG/NBD** | Batislam et al. (2007). Allows customers with no repeat purchases (accelerator notes this but sticks to classic BG/NBD). |

### Package notes (practical)

| Item | Detail |
|------|--------|
| **Latest on PyPI** | `0.1b3` (as of project page) |
| **Python** | Declares `>=3.8,<3.10` for current betas; accelerator used an older alpha with different pins |
| **License** | Apache 2.0 |
| **Extras** | Bayesian PyMC implementations in beta; classic Autograd fitters remain the Lifetimes-compatible path |
| **Related** | R: [BTYDplus](https://github.com/mplatzer/BTYDplus); theory: [Bruce Hardie’s site](http://brucehardie.com/) |

Install:

```bash
pip install btyd
```

---

## 4. End-to-end pipeline (what they do)

### Step 0 — Scope & data

- **Data:** [UCI Online Retail](http://archive.ics.uci.edu/ml/datasets/Online+Retail) (~1 year of UK e-commerce line items)
- **Grain:** invoice line → daily customer spend (`SalesAmount = Qty × UnitPrice`)
- **Why this data:** non-contractual, repeat purchases, enough history for calibration/holdout

### Step 1 — Clean & explore

- Drop null `CustomerID`
- Remove extreme outliers (e.g. ~£70k single-day spenders)
- Confirm distributions: frequency ~ negative binomial; spend long-tailed, not Gaussian  
  → justifies BTYD over “average revenue × average lifetime”

### Step 2 — Build RFM metrics

- Compute frequency / recency / T / monetary_value
- Show both **pandas (`btyd.utils`)** and **Spark** paths (enterprise scale)
- Split **calibration vs holdout** (last **90 days** held out)
- Filter to customers with **frequency > 0** (classic BG/NBD / Pareto-NBD style; one-timers excluded)
- Handle negatives/returns pragmatically (exclude non-positive daily totals for the demo)

### Step 3 — Train engagement model (BG/NBD)

- Fit on calibration RFM (`BetaGeoFitter`)
- Predict holdout purchase counts → RMSE + calibration plots
- Score `conditional_probability_alive` and expected purchases
- Visualize frequency×recency heatmaps (alive matrix, expected purchases)

**Why BG/NBD:** in non-contractual settings you never observe “death”; you infer it from how recency/frequency compare to population patterns.

### Step 4 — Train spend model (Gamma-Gamma)

- Check weak correlation between frequency and monetary value (model assumption)
- Fit on frequency + monetary_value (`GammaGammaFitter`)
- Validate predicted vs actual spend (histograms / RMSE)

**Why separate from BG/NBD:** retention (how often / whether they buy) is treated as independent of how much they spend when they buy.

### Step 5 — Combine into CLV

```text
CLV ≈ E[future purchases from BG/NBD] × E[spend per purchase from Gamma-Gamma]
      discounted over time (e.g. 12 months, 1% monthly)
```

Implemented as:

```python
ggm.customer_lifetime_value(
    bgf,
    frequency, recency, T, monetary_value,
    time=12,           # months
    discount_rate=0.01 # monthly
)
```

### Step 6 — Productionize

- Custom MLflow `pyfunc` wrapper
- Persist BG/NBD as artifact; wrap with Gamma-Gamma
- Expose as Spark UDF / SQL for batch scoring

---

## 5. Design rationale

```text
Transaction history
        │
        ▼
   RFM metrics
     /        \
    ▼          ▼
 BG/NBD     Gamma-Gamma
 (alive +    (spend per
  purchases)  purchase)
     \        /
      ▼      ▼
   Discounted CLV
        │
        ▼
 Marketing / retention / equity decisions
```

1. **Problem fit:** Non-contractual retail has latent churn — survival-style BTYD is purpose-built.
2. **Parsimony:** Needs only invoice ID, date, customer ID, spend — no rich feature store for the baseline.
3. **Individual-level:** Scores each customer, not a segment average.
4. **Interpretable levers:** Alive probability, expected purchases, expected spend, discounted CLV — each actionable alone.
5. **Enterprise path:** Spark for metrics at scale + MLflow for serving (the accelerator’s Databricks angle).

---

## 6. Intended business uses

| Use | How CLV helps |
|-----|----------------|
| **Acquisition** | Target lookalikes of high-CLV customers; cap CAC by expected CLV |
| **Retention** | Invest more in high-CLV / high P(alive) customers |
| **Personalization** | Match offer intensity to lifetime potential |
| **Budgeting** | Avoid over-investing in low-equity / transient buyers |
| **Monitoring** | Track aggregate CLV as customer-equity health over time |

---

## 7. Assumptions & limits

- Customers can “die” permanently (classic model has no reactivation after death)
- Purchase process while alive follows the BG/NBD generative story
- Frequency ⊥ monetary value (Gamma-Gamma assumption)
- Repeat buyers only in the main fit (`frequency > 0`); MBG/NBD exists in `btyd` for one-timers but is not the accelerator’s focus
- Horizon limited by data (~12 months CLV in the notebook)
- Demo shortcuts: returns not fully reconciled; holiday-season holdout can bias validation
- Monetary = revenue (`SalesAmount`), not margin — production CLV often wants contribution profit

---

## 8. One-line summary

They model CLV as **probabilistic future engagement (BG/NBD) × expected spend (Gamma-Gamma)**, discounted over time, because in non-subscription retail you cannot observe churn or lifetime directly — you must infer both from RFM transaction signals and then use that to allocate marketing and retention dollars.

---

## 9. Relation to this repo

This document is a **reference** to the Databricks / `btyd` approach. This project’s own CLV path is documented in [`MODELING.md`](MODELING.md): weekly noncontractual BG/NBD + Gamma-Gamma on UCI Online Retail II (`scripts/clv_weekly.py`, `Notebooks/01-clv.ipynb`), with audit-first sequencing and an aggregate acceptance gate.

`Notebooks/01-clv.ipynb` (generated by `scripts/generate_clv_notebook.py`) runs the accelerator’s **full step chain in-notebook** — data → explore → protocol → RFM → val BG/NBD → val GG (£ scale) → test BG/NBD → test GG → scaled CLV + gate → segments → scenarios → artifacts — while keeping weekly time units, the 45/20/35 split, val-only £ scale, and the ≤5% gate (not daily RFM, 90-day holdout, discounting, or MLflow).
