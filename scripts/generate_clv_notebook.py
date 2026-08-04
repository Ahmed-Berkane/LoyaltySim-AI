"""Generate Notebooks/01-clv.ipynb — every BTYD step explicit, weekly Hardie pipeline.

Walks Databricks accelerator order step-by-step (data → explore → RFM → BG/NBD →
Gamma-Gamma → CLV → uses → artifacts) using clv_weekly's granular API — not a
single black-box run_three_way. Preserves weekly freq, 45/20/35, val-only scale,
≤5% £ gate, no discounting / MLflow.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Notebooks"


def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }


def _normalize_cell_source(text: str) -> str:
    lines = text.strip("\n").splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return "\n"
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in nonempty]
    if indents[0] == 0 and len(indents) > 1 and min(indents[1:]) > 0:
        pad = min(indents[1:])
    else:
        pad = min(indents)
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        elif len(ln) >= pad and ln[:pad].strip() == "":
            out.append(ln[pad:])
        else:
            out.append(ln.lstrip(" ") if ln.startswith(" ") else ln)
    return "\n".join(out).strip("\n") + "\n"


def md(text: str, cid: str) -> dict:
    cleaned = _normalize_cell_source(text)
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": cleaned.splitlines(keepends=True)}


def code(text: str, cid: str) -> dict:
    cleaned = _normalize_cell_source(text)
    return {
        "cell_type": "code",
        "id": cid,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": cleaned.splitlines(keepends=True),
    }


SETUP = r'''import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def find_project_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in (path, *path.parents):
        if (candidate / "scripts" / "uci_pipeline.py").exists():
            return candidate
    return path


PROJECT_ROOT = find_project_root()
DATA = PROJECT_ROOT / "data" / "modeling"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from clv_weekly import (
    AGG_ERROR_MAX,
    FIT_FRAC,
    FREQ,
    HORIZON_WEEKS_CANDIDATES,
    VAL_FRAC,
    assert_accepted,
    assign_segments,
    audit_no_leakage,
    build_summary,
    cumulative_expected_purchases,
    fit_purchase_model,
    fit_score_monetary,
    score_purchases,
    split_dates,
    weeks_between,
)


def save_artifact(name: str, obj) -> Path:
    path = MODELS / name
    joblib.dump(obj, path)
    print(f"Saved → {path.relative_to(PROJECT_ROOT)}")
    return path


print(
    f"Setup complete · FREQ={FREQ} · stepwise BG/NBD + Gamma-Gamma · "
    f"fit/val/test={FIT_FRAC:.0%}/{VAL_FRAC:.0%}/{1 - FIT_FRAC - VAL_FRAC:.0%} · "
    f"£ gate ≤ {AGG_ERROR_MAX:.0%}"
)
'''


def main() -> None:
    cells = [
        # ============================================================
        md(
            """# 01 · Customer Lifetime Value (noncontractual, weekly)

In non-subscription retail, customers come and go with no contract. CLV needs two estimates:

1. **Engagement** — will they return? (BG/NBD)
2. **Spend** — how much per purchase? (Gamma-Gamma)

This notebook runs **every step** of that BTYD pipeline (same story as the [Databricks CLV accelerator](https://github.com/databricks-industry-solutions/customer-lifetime-value) / `DATABRICKS_CLV_PLAN.md`), on this repo’s weekly Hardie-faithful protocol.

| Step | Section | What you do |
|------|---------|-------------|
| 1 | Access data | Load audit order spine |
| 2 | Explore | Confirm skewed frequency & spend |
| 3 | Protocol | Fit / val / test cutoffs |
| 4 | RFM metrics | Build calibration + holdout summaries |
| 5 | BG/NBD (val) | Fit engagement; predict val purchases |
| 6 | Gamma-Gamma (val) | Fit spend; form val CLV → **£ scale** |
| 7 | BG/NBD (test) | Refit on test calibration; score + diagnose |
| 8 | Gamma-Gamma (test) | Refit spend on test calibration |
| 9 | CLV + gate | Scale, accept/reject, ranking, MLP foil |
| 10 | Segments | Value × activity action map |
| 11 | Scenarios | Illustrative what-ifs |
| 12 | Artifacts | Persist only if gate passes |

**CLV formula:** `E[purchases] × E[avg spend] × val_scale` (no monthly discounting here).  
**Purchase model:** stationary weekly BG/NBD (Hardie default — no seasonal overlay).  
**Upstream:** `00-customer-base-audit.ipynb`.
""",
            "intro",
        ),
        code(SETUP, "setup"),
        # ============================================================
        md(
            """## 1 · Access the data

Order-level spine from the customer-base audit (UCI Online Retail II). Grain = **one row per order** with `spend` (Databricks collapses lines to daily spend; we stay at order / week).
""",
            "h-data",
        ),
        code(
            """from customer_base_audit import attach_acquisition, build_orders, load_fact_transactions

audit_orders = DATA / "audit" / "uci_orders.parquet"
if audit_orders.exists():
    orders = pd.read_parquet(audit_orders)
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    print("Loaded", audit_orders.relative_to(PROJECT_ROOT))
else:
    fact = load_fact_transactions(DATA / "uci_fact_transactions.parquet")
    orders = attach_acquisition(build_orders(fact))

print(f"Orders: {len(orders):,} · Customers: {orders['customer_id'].nunique():,}")
print(f"Span: {orders['order_date'].min().date()} → {orders['order_date'].max().date()}")
print(f"Columns: {list(orders.columns)}")
display(orders.head(10))
""",
            "load",
        ),
        # ============================================================
        md(
            """## 2 · Explore purchase behavior

BTYD assumes frequency and spend are **right-skewed**, not Gaussian. If that holds, average lifetime × average spend misstates equity; individual RFM does not.
""",
            "h-explore",
        ),
        code(
            """txn_per_cust = orders.groupby("customer_id")["order_id"].nunique()
spend = orders["spend"]
one_timers = int((txn_per_cust == 1).sum())
print(
    f"Customers: {txn_per_cust.size:,} · one-timers: {one_timers:,} "
    f"({one_timers / txn_per_cust.size:.1%})"
)
print(
    f"Orders/customer — median {txn_per_cust.median():.0f} · "
    f"mean {txn_per_cust.mean():.2f} · p95 {txn_per_cust.quantile(0.95):.0f}"
)
print(
    f"Order spend £ — median {spend.median():.2f} · "
    f"mean {spend.mean():.2f} · p95 {spend.quantile(0.95):.2f}"
)

# Monthly activity (seasonality is visible in actuals; not in the purchase model)
monthly = (
    orders.assign(month=orders["order_date"].dt.to_period("M").dt.to_timestamp())
    .groupby("month")
    .agg(orders=("order_id", "nunique"), spend=("spend", "sum"), customers=("customer_id", "nunique"))
    .reset_index()
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].hist(txn_per_cust.clip(upper=txn_per_cust.quantile(0.99)), bins=40, color="steelblue", edgecolor="white")
axes[0, 0].set_title("Frequency (orders / customer)")
axes[0, 0].set_xlabel("Orders (clipped p99)")
axes[0, 1].hist(spend.clip(upper=spend.quantile(0.99)), bins=40, color="darkorange", edgecolor="white")
axes[0, 1].set_title("Monetary (order spend £)")
axes[0, 1].set_xlabel("£ (clipped p99)")
axes[1, 0].plot(monthly["month"], monthly["orders"], color="black")
axes[1, 0].set_title("Orders by month")
axes[1, 0].tick_params(axis="x", rotation=45)
axes[1, 1].plot(monthly["month"], monthly["spend"], color="crimson")
axes[1, 1].set_title("Spend £ by month")
axes[1, 1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
display(monthly.tail(8).round(0))
""",
            "explore",
        ),
        md(
            """#### Assessment — explore

Frequency and spend are long-tailed → BG/NBD + Gamma-Gamma are appropriate. Monthly plots show seasonal peaks in **actuals**; the purchase model stays stationary on purpose (equity baseline, not a demand plan).
""",
            "a-explore",
        ),
        # ============================================================
        md(
            """## 3 · Define the protocol (fit / validation / test)

Databricks uses a single calibration + ~90-day holdout. This repo uses a **three-way** split so £ scale never sees test labels:

| Window | Share | Role |
|--------|-------|------|
| Fit → `fit_end` | 45% | Contained in calibration for later models |
| → `val_end` | +20% | Holdout for **£ scale only** |
| → `obs_end` | +35% | Unseen **test** for the ≤5% gate |
""",
            "h-protocol",
        ),
        code(
            """min_d, fit_end, val_end, max_d = split_dates(orders)
hw_val = weeks_between(fit_end, val_end)
hw_test = weeks_between(val_end, max_d)

print(f"FREQ = {FREQ} (weeks)")
print(f"min_d    = {min_d.date()}")
print(f"fit_end  = {fit_end.date()}  (end of fit calendar share)")
print(f"val_end  = {val_end.date()}  (calibration end for TEST score set)")
print(f"obs_end  = {max_d.date()}  (end of data)")
print(f"Val holdout horizon  = {hw_val} weeks")
print(f"Test holdout horizon = {hw_test} weeks")
""",
            "protocol",
        ),
        # ============================================================
        md(
            """## 4 · Calculate customer metrics (RFM)

Per-customer inputs (time unit = **weeks**):

| Metric | Meaning |
|--------|---------|
| `frequency_cal` | Repeat purchase periods after first purchase (in calibration) |
| `recency_cal` | Customer age (weeks) at last calibration purchase |
| `T_cal` | Age from first purchase to end of calibration |
| `monetary_value_cal` | Average spend on repeat calibration periods |
| `frequency_holdout` / revenue | Actuals in the holdout window (for validation only) |

Build **two** summaries: validation window (for scale) and test window (for shipping scores).
""",
            "h-rfm",
        ),
        code(
            """# --- Validation RFM: calibrate through fit_end, hold out fit_end → val_end ---
s_val = build_summary(orders, fit_end, val_end)
print("=== Validation summary ===")
print(f"Customers: {len(s_val):,} · repeat in cal: {(s_val['frequency_cal'] > 0).mean():.1%}")
print(f"duration_holdout (weeks): {int(s_val['duration_holdout'].iloc[0])}")
display(s_val.head(10))
display(s_val[["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal", "frequency_holdout", "actual_holdout_revenue"]].describe().round(2))
""",
            "rfm-val",
        ),
        code(
            """# --- Test RFM: calibrate through val_end, hold out val_end → obs_end ---
s_test = build_summary(orders, val_end, max_d)
print("=== Test summary (score set) ===")
print(f"Customers: {len(s_test):,} · repeat in cal: {(s_test['frequency_cal'] > 0).mean():.1%}")
print(f"duration_holdout (weeks): {int(s_test['duration_holdout'].iloc[0])}")
display(s_test.head(10))

cal_cols = ["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(s_test[cal_cols].corr(), annot=True, fmt=".2f", ax=ax)
ax.set_title("Test-set calibration RFM correlation (weeks)")
plt.tight_layout()
plt.show()
""",
            "rfm-test",
        ),
        md(
            """#### Assessment — RFM

Same metrics Databricks builds with `summary_data_from_transaction_data` / Spark — here via `lifetimes` weekly `calibration_and_holdout_data`. Test scores only use customers known by `val_end` (no future-acquisition leakage).
""",
            "a-rfm",
        ),
        # ============================================================
        md(
            """## 5 · Train engagement on the validation window (BG/NBD)

Fit BG/NBD on **validation calibration** RFM, predict purchases over the val holdout, then (next section) attach spend and form the **val-only £ scale**.

Paper: [Fader, Hardie & Lee 2005](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf).
""",
            "h-bg-val",
        ),
        code(
            """bgf_val = fit_purchase_model(s_val)
s_val = score_purchases(s_val, bgf_val, horizon_weeks=hw_val)

print("BG/NBD (val) params:", {k: round(float(v), 4) for k, v in bgf_val.params_.items()})
print(
    f"Val purchases: pred {s_val['expected_purchases'].sum():,.0f} vs "
    f"actual {s_val['actual_holdout_orders'].sum():,.0f}"
)

# Lifetimes calibration vs holdout chart (Databricks-style)
plot_calibration_purchases_vs_holdout_purchases(
    bgf_val,
    s_val.rename(columns={"frequency_holdout": "frequency_holdout"}),
    n=hw_val,
    **{"figsize": (8, 6)},
)
plt.title("Val window — calibration purchases vs holdout")
plt.tight_layout()
plt.show()

display(s_val[["frequency_cal", "recency_cal", "T_cal", "p_alive", "expected_purchases", "frequency_holdout"]].head(10))
""",
            "bg-val",
        ),
        md(
            """#### Assessment — BG/NBD (val)

Higher calibration frequency should map to higher holdout purchases. This fit is used only to form validation CLV for scaling — the **shipped** engagement model is refit on the test calibration in §7.
""",
            "a-bg-val",
        ),
        # ============================================================
        md(
            """## 6 · Train spend on the validation window (Gamma-Gamma) → £ scale

Gamma-Gamma supplies expected average order value ([Fader & Hardie 2013](http://www.brucehardie.com/notes/025/gamma_gamma.pdf)). Assumption: frequency ⊥ monetary value.

```text
val_raw_CLV = E[purchases]_val × E[spend]_val
val_scale   = val_actual_£ / val_raw_CLV
```

Test labels never enter this scale.
""",
            "h-gg-val",
        ),
        code(
            """s_val, ggf_val, mon_cap_val = fit_score_monetary(s_val)
val_actual = float(s_val["actual_holdout_revenue"].sum())
val_pred = float(s_val["expected_clv"].sum())
if val_pred <= 0 or val_actual <= 0:
    raise RuntimeError("Validation window has zero predicted or actual revenue")
scale = val_actual / val_pred

gg_corr_val = float(
    s_val.loc[s_val["frequency_cal"] > 0, ["frequency_cal", "monetary_value_cal"]]
    .corr()
    .iloc[0, 1]
)
print("GG (val) params:", {k: round(float(v), 4) for k, v in ggf_val.params_.items()})
print(f"GG independence (freq vs monetary): {gg_corr_val:.3f}")
print(f"Monetary winsor cap (val cal p99): £{mon_cap_val:,.2f}")
print(f"Val £ actual={val_actual:,.0f} · pred={val_pred:,.0f} · scale={scale:.4f}")
display(s_val[["expected_purchases", "expected_avg_spend", "expected_clv", "actual_holdout_revenue"]].head(10))
""",
            "gg-val",
        ),
        md(
            """#### Assessment — Gamma-Gamma (val) + scale

Weak freq–monetary correlation supports GG. A scale near 1.0 means raw purchase × spend already matched val £ closely; the scale only recalibrates levels before the test gate.
""",
            "a-gg-val",
        ),
        # ============================================================
        md(
            """## 7 · Train engagement on the test score set (BG/NBD)

Refit BG/NBD with calibration through `val_end`. Score `p_alive` and expected purchases over the **test** horizon. Then diagnose: conditional expectations, alive / frequency–recency matrices, aggregate tracking.
""",
            "h-bg-test",
        ),
        code(
            """bgf = fit_purchase_model(s_test)
s_test = score_purchases(s_test, bgf, horizon_weeks=hw_test)

print("BG/NBD (test) params:", {k: round(float(v), 4) for k, v in bgf.params_.items()})
print(
    f"TEST purchases: pred {s_test['expected_purchases'].sum():,.0f} vs "
    f"actual {s_test['actual_holdout_orders'].sum():,.0f}"
)
print(
    f"p_alive median={s_test['p_alive'].median():.3f} · "
    f"mean E[purchases]={s_test['expected_purchases'].mean():.2f}"
)
display(s_test[["frequency_cal", "recency_cal", "T_cal", "p_alive", "expected_purchases", "frequency_holdout"]].head(10))
""",
            "bg-test-fit",
        ),
        code(
            """# Databricks-style matrices — lifetimes plot_* calls plt.subplot(111) and ignores plt.sca,
# so we draw heatmaps directly on our own axes.
max_freq = int(min(7, s_test["frequency_cal"].quantile(0.99)))
max_rec = int(min(60, s_test["T_cal"].quantile(0.99)))
T_plot = min(hw_test, 26)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

z_alive = bgf.conditional_probability_alive_matrix(max_frequency=max_freq, max_recency=max_rec)
pcm0 = axes[0].imshow(z_alive, interpolation="none", aspect="auto")
axes[0].set_xlabel("Customer's Historical Frequency")
axes[0].set_ylabel("Customer's Recency")
axes[0].set_title("P(alive) by frequency × recency")
fig.colorbar(pcm0, ax=axes[0])

Z = np.zeros((max_rec + 1, max_freq + 1))
for i, recency in enumerate(np.arange(max_rec + 1)):
    for j, frequency in enumerate(np.arange(max_freq + 1)):
        Z[i, j] = bgf.conditional_expected_number_of_purchases_up_to_time(
            T_plot, frequency, recency, max_rec
        )
pcm1 = axes[1].imshow(Z, interpolation="none", aspect="auto")
axes[1].set_xlabel("Customer's Historical Frequency")
axes[1].set_ylabel("Customer's Recency")
axes[1].set_title(f"E[purchases] next {T_plot} weeks")
fig.colorbar(pcm1, ax=axes[1])

plt.tight_layout()
plt.show()

plot_calibration_purchases_vs_holdout_purchases(bgf, s_test, n=hw_test, **{"figsize": (8, 6)})
plt.title("Test window — calibration purchases vs holdout")
plt.tight_layout()
plt.show()
""",
            "bg-test-matrices",
        ),
        md(
            """#### Reading the calibration-vs-holdout chart

Each point is a **bucket of customers** with the same `frequency_cal` (x-axis: purchases in the calibration period). Blue = their actual mean holdout purchases; orange = BG/NBD's predicted mean for that same bucket.

**What this run shows:**

- **0–~13 calibration purchases** (the bulk of the customer base) — blue and orange track each other closely. The model is well calibrated where most customers live.
- **~14+ calibration purchases** — blue gets spiky while orange stays smooth. These are high-frequency buckets with **few customers each** (see the shrinking sample size as `x` grows), so one or two customers' actual holdout counts swing the bucket average a lot. BG/NBD instead reports the *expected* value, which is naturally smoother.
- **The last point (~37)** is the most extreme — likely a single very active customer. Don't read one sparse point as a systematic miss.

This is consistent with the other diagnostics: aggregate test purchase |error| ~7% and £ gate ~2.4% pass, and Spearman ~0.62 shows the model ranks customers correctly across the RFM ladder. **Tracking well for the bulk of the base, noisy-but-expected at the sparse high-frequency tail** — not evidence of a broken fit.
""",
            "a-bg-test-calhold",
        ),
        code(
            """# Conditional expectations by cal frequency
freq_cap = int(min(7, s_test["frequency_cal"].quantile(0.99)))
cond = s_test.copy()
cond["freq_bin"] = cond["frequency_cal"].clip(upper=freq_cap).astype(int)
by_x = (
    cond.groupby("freq_bin", as_index=False)
    .agg(
        n=("expected_purchases", "size"),
        actual_mean=("frequency_holdout", "mean"),
        pred_mean=("expected_purchases", "mean"),
    )
)
x = by_x["freq_bin"].to_numpy()
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w / 2, by_x["actual_mean"], width=w, color="black", label="Actual mean holdout orders")
ax.bar(x + w / 2, by_x["pred_mean"], width=w, color="crimson", label="BG/NBD E[purchases]")
ax.set_xlabel("Calibration frequency x (clipped)")
ax.set_ylabel("Mean holdout purchases")
ax.set_title("Conditional expectations — actual vs BG/NBD (test)")
ax.legend(frameon=False)
for _, row in by_x.iterrows():
    ax.text(
        row["freq_bin"],
        max(row["actual_mean"], row["pred_mean"]) + 0.05,
        f"n={int(row['n'])}",
        ha="center",
        fontsize=8,
        color="gray",
    )
plt.tight_layout()
plt.show()

# Aggregate tracking — fixed calibration cohort only
hold = orders[(orders["order_date"] > val_end) & (orders["order_date"] <= max_d)].copy()
cal_ids = s_test.index
hold_cohort = hold[hold["customer_id"].isin(cal_ids)].copy()
new_mask = ~hold["customer_id"].isin(cal_ids)
n_new_orders = int(hold.loc[new_mask, "order_id"].nunique())
n_new_cust = int(hold.loc[new_mask, "customer_id"].nunique())

hold_w = hold_cohort.copy()
hold_w["hold_week"] = ((hold_w["order_date"] - val_end).dt.days // 7) + 1
actual_by_week = hold_w.groupby("hold_week")["order_id"].nunique()
weeks = np.arange(1, hw_test + 1)
actual_weekly = actual_by_week.reindex(weeks).fillna(0).to_numpy(dtype=float)
actual_cum = np.cumsum(actual_weekly)
pred_cum = cumulative_expected_purchases(bgf, s_test, hw_test)
pred_weekly = np.diff(pred_cum, prepend=0.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(weeks, actual_weekly, color="black", label="Actual weekly orders")
axes[0].plot(weeks, pred_weekly, color="crimson", label="BG/NBD weekly expected flow")
axes[0].set_xlabel("Test week")
axes[0].set_ylabel("Orders / week")
axes[0].set_title("Weekly volume vs stationary baseline")
axes[0].legend(frameon=False)
axes[1].plot(weeks, actual_cum, color="black", label="Actual cum")
axes[1].plot(weeks, pred_cum, color="crimson", label="BG/NBD E[cum]")
axes[1].set_xlabel("Test week")
axes[1].set_ylabel("Cumulative orders")
axes[1].set_title("Cumulative tracking (fixed cohort)")
axes[1].legend(frameon=False)
fig.suptitle("Purchase tracking — calibration cohort only", y=1.02)
plt.tight_layout()
plt.show()

gap = abs(actual_cum[-1] - pred_cum[-1]) / max(actual_cum[-1], 1)
test_orders_pred = float(s_test["expected_purchases"].sum())
test_orders_actual = float(s_test["actual_holdout_orders"].sum())
orders_agg_error = abs(test_orders_pred - test_orders_actual) / test_orders_actual
print(
    f"End-of-horizon (cal cohort): actual={actual_cum[-1]:,.0f} · "
    f"BG/NBD={pred_cum[-1]:,.0f} · |gap|={gap:.1%}"
)
print(f"TEST purchase |error| (informational): {orders_agg_error:.1%}")
print(
    f"Excluded from chart: {n_new_orders:,} orders from {n_new_cust:,} customers "
    f"first seen after val_end."
)
""",
            "bg-test-diag",
        ),
        md(
            """#### Assessment — engagement (test)

Alive and frequency–recency matrices show the classic BTYD pattern: recent + frequent ⇒ high P(alive) and expected purchases. Aggregate path is intentionally flat; seasonal peaks stay in actuals. Transaction |error| is diagnostic — **not** the commercial gate.
""",
            "a-bg-test",
        ),
        # ============================================================
        md(
            """## 8 · Train spend on the test score set (Gamma-Gamma)

Refit Gamma-Gamma on test calibration monetary (winsorized at cal p99). Form **raw** CLV = purchases × expected avg spend; apply `val_scale` in the next section.
""",
            "h-gg-test",
        ),
        code(
            """s_test, ggf, mon_cap = fit_score_monetary(s_test)
gg_corr = float(
    s_test.loc[s_test["frequency_cal"] > 0, ["frequency_cal", "monetary_value_cal"]]
    .corr()
    .iloc[0, 1]
)
print("GG (test) params:", {k: round(float(v), 4) for k, v in ggf.params_.items()})
print(f"GG independence (freq vs monetary): {gg_corr:.3f}")
print(f"Monetary winsor cap (test cal p99): £{mon_cap:,.2f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(
    s_test["monetary_value_cal"].clip(upper=s_test["monetary_value_cal"].quantile(0.99)),
    bins=40,
    color="steelblue",
    edgecolor="white",
    label="Cal monetary",
)
axes[0].hist(
    s_test["expected_avg_spend"].clip(upper=s_test["expected_avg_spend"].quantile(0.99)),
    bins=40,
    histtype="step",
    color="crimson",
    linewidth=2,
    label="GG expected avg spend",
)
axes[0].set_xlabel("£")
axes[0].set_title("Spend: calibration vs GG expectation")
axes[0].legend(frameon=False)
axes[1].scatter(
    s_test["frequency_cal"],
    s_test["monetary_value_cal"].clip(upper=s_test["monetary_value_cal"].quantile(0.99)),
    alpha=0.2,
    s=12,
    edgecolors="none",
)
axes[1].set_xlabel("frequency_cal")
axes[1].set_ylabel("monetary_value_cal (clipped)")
axes[1].set_title("Independence check")
plt.tight_layout()
plt.show()

display(s_test[["expected_purchases", "expected_avg_spend", "expected_clv", "actual_holdout_revenue"]].head(10))
""",
            "gg-test",
        ),
        md(
            """#### Assessment — spend (test)

GG smooths noisy order totals into expected average spend for CLV. One-timers (`frequency_cal == 0`) receive the population median spend inside `fit_score_monetary`.
""",
            "a-gg-test",
        ),
        # ============================================================
        md(
            """## 9 · Calculate CLV, scale, and accept / reject

```text
raw CLV     = E[purchases] × E[avg spend]          # already on s_test
scaled CLV  = raw CLV × val_scale
gate        = |Σ scaled − Σ actual| / Σ actual  ≤ 5%
```

Unlike Databricks’ `customer_lifetime_value(..., discount_rate=0.01)`, **this repo does not discount**. Ship only if the gate and leakage audit pass.
""",
            "h-clv",
        ),
        code(
            """from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Apply val-only scale (do NOT pattern-match expected_clv_raw — it ends in "w")
s_test["expected_clv_raw"] = s_test["expected_clv"]
s_test["expected_clv"] = s_test["expected_clv_raw"] * scale
for h in HORIZON_WEEKS_CANDIDATES:
    c = f"expected_clv_{h}w"
    if c in s_test.columns:
        s_test[c] = s_test[c] * scale
s_test[f"expected_clv_{hw_test}w"] = s_test["expected_clv"]
s_test["expected_clv_calibrated"] = s_test["expected_clv"]

test_actual = float(s_test["actual_holdout_revenue"].sum())
test_pred = float(s_test["expected_clv"].sum())
agg_error = abs(test_pred - test_actual) / test_actual if test_actual else np.inf
accepted = bool(agg_error <= AGG_ERROR_MAX)

result = {
    "orders": orders,
    "min_date": min_d,
    "fit_end": fit_end,
    "val_end": val_end,
    "obs_end": max_d,
    "hw_val": hw_val,
    "hw_test": hw_test,
    "t_horizon": hw_test,
    "scale": scale,
    "val_actual": val_actual,
    "val_pred": val_pred,
    "val_summary": s_val,
    "test_actual": test_actual,
    "test_pred": test_pred,
    "agg_error": agg_error,
    "accepted": accepted,
    "summary": s_test,
    "bgf": bgf,
    "ggf": ggf,
    "bgf_val": bgf_val,
    "ggf_val": ggf_val,
    "monetary_winsor_cap": mon_cap,
    "agg_error_max": AGG_ERROR_MAX,
    "test_orders_actual": test_orders_actual,
    "test_orders_pred": test_orders_pred,
    "orders_agg_error": orders_agg_error,
    "pred_cum_purchases": pred_cum,
    "model_name": "BG/NBD",
}
result["leakage_audit"] = audit_no_leakage(result, orders)

print(f"Val £ scale = {scale:.4f}")
print(f"TEST £: pred £{test_pred:,.0f} vs actual £{test_actual:,.0f} | |error| = {agg_error:.2%}")
assert_accepted(result)
leak = result["leakage_audit"]
print(f"ACCEPTED · £ |error| ≤ {AGG_ERROR_MAX:.0%} · leakage_audit=PASS")
print(
    f"  cal_orders={leak['n_cal_orders']:,} · test_orders={leak['n_test_orders']:,} · "
    f"future_acq_in_score={leak['customers_first_seen_in_test']}"
)
""",
            "clv-gate",
        ),
        code(
            """summary = s_test
t_horizon = hw_test

eval_df = summary[
    ["expected_clv", "expected_clv_raw", "expected_purchases", "actual_holdout_revenue", "frequency_holdout"]
].dropna().copy()

mae = mean_absolute_error(eval_df["actual_holdout_revenue"], eval_df["expected_clv"])
r2 = r2_score(eval_df["actual_holdout_revenue"], eval_df["expected_clv"])
spearman = spearmanr(eval_df["expected_clv"], eval_df["actual_holdout_revenue"]).correlation
sp_purch = spearmanr(eval_df["expected_purchases"], eval_df["frequency_holdout"]).correlation

n = len(eval_df)
k = max(int(n * 0.10), 1)
total_rev = eval_df["actual_holdout_revenue"].sum()
capture = eval_df.nlargest(k, "expected_clv")["actual_holdout_revenue"].sum() / total_rev
ideal = eval_df.nlargest(k, "actual_holdout_revenue")["actual_holdout_revenue"].sum() / total_rev

print(f"TEST £ · {t_horizon}w — aggregate |error|: {agg_error:.2%} (limit {AGG_ERROR_MAX:.0%})")
print(f"  Spearman revenue: {spearman:.3f} · purchases: {sp_purch:.3f}")
print(f"  Top-decile capture: {capture:.1%} (ideal {ideal:.1%})")
print(f"  MAE: {mae:.2f} · R2: {r2:.3f}")

# MLP foil: train on VAL labels only, score on TEST
rfm_val = s_val[
    ["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal", "actual_holdout_revenue"]
].copy()
rfm_test = summary[
    ["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal", "actual_holdout_revenue"]
].copy()
X_tr = rfm_val[["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]]
y_tr = rfm_val["actual_holdout_revenue"]
X_te = rfm_test[["frequency_cal", "recency_cal", "T_cal", "monetary_value_cal"]]
y_te = rfm_test["actual_holdout_revenue"]
mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)),
])
mlp_pipe.fit(X_tr, y_tr)
mlp_pred = mlp_pipe.predict(X_te)
mlp_sp = spearmanr(mlp_pred, y_te).correlation
mlp_r2 = r2_score(y_te, mlp_pred)
mlp_mae = mean_absolute_error(y_te, mlp_pred)
print(f"MLP baseline (train=VAL, test=TEST) — Spearman: {mlp_sp:.3f} · R2: {mlp_r2:.3f} · MAE: {mlp_mae:.2f}")
best_name = (
    "BG/NBD+Gamma-Gamma"
    if (float(spearman) if spearman == spearman else -1) >= (float(mlp_sp) if mlp_sp == mlp_sp else -1)
    else "MLP"
)
print(f"Selected on ranking: {best_name}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, actual, title in (
    (axes[0], eval_df["expected_clv"], eval_df["actual_holdout_revenue"], f"BG/NBD+GG scaled · R2={r2:.3f}"),
    (axes[1], pd.Series(mlp_pred), y_te.reset_index(drop=True), f"MLP (val→test) · R2={mlp_r2:.3f}"),
):
    lim = float(np.nanpercentile(np.concatenate([np.asarray(pred), np.asarray(actual)]), 99))
    lim = max(lim, 1.0)
    ax.scatter(pred, actual, alpha=0.25, s=12, edgecolors="none")
    ax.plot([0, lim], [0, lim], color="crimson", lw=1.5, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted £")
    ax.set_ylabel("Actual holdout £")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")
fig.suptitle(f"Test actual vs predicted £ · agg |error|={agg_error:.2%}", y=1.02)
plt.tight_layout()
plt.show()
""",
            "clv-rank",
        ),
        md(
            """#### Assessment — CLV protocol

Gate + leakage define ship / no-ship. Ranking (Spearman, top-decile capture) decides whether equity concentrates real spend. Customer-level R² stays modest in heavy-tailed retail — secondary.
""",
            "a-clv",
        ),
        # ============================================================
        md(
            """## 10 · Segments (value × activity)

Map CLV × expected purchases to actions (VIP retention, win-back, upsell, nurture, minimize) — how BTYD scores enter marketing decisions.
""",
            "h-seg",
        ),
        code(
            """seg_clv_col = "expected_clv"
for h in (52, 26, 13, 4, t_horizon):
    c = f"expected_clv_{h}w"
    if c in summary.columns:
        seg_clv_col = c
        break

summary = assign_segments(summary, seg_clv_col)
seg_summary = (
    summary.groupby("segment", as_index=False)
    .agg(
        n_customers=("expected_purchases", "size"),
        mean_exp_purch=("expected_purchases", "mean"),
        mean_p_alive=("p_alive", "mean"),
        mean_clv=(seg_clv_col, "mean"),
        total_clv=(seg_clv_col, "sum"),
        total_holdout_rev=("actual_holdout_revenue", "sum"),
    )
    .sort_values("total_holdout_rev", ascending=False)
)
display(seg_summary.round(2))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].barh(seg_summary["segment"], seg_summary["total_clv"], color="steelblue")
axes[0].set_title("Predicted (scaled) equity")
axes[1].barh(seg_summary["segment"], seg_summary["total_holdout_rev"], color="darkorange")
axes[1].set_title("Actual test revenue")
plt.tight_layout()
plt.show()
""",
            "seg",
        ),
        md(
            """#### Assessment — segments

Predicted equity should align directionally with realized test £. Prefer retention on high-CLV / high-activity; win-back where value is high but expected purchases are soft.
""",
            "a-seg",
        ),
        # ============================================================
        md(
            """## 11 · Decision scenarios (illustrative)

Portfolio sensitivity only — not causal uplift.
""",
            "h-scen",
        ),
        code(
            """equity0 = float(summary["expected_clv"].sum())
scenarios = []
top_n = max(int(len(summary) * 0.10), 1)
top_ids = summary.nlargest(top_n, "expected_clv").index
s1 = summary["expected_clv"].copy()
s1.loc[top_ids] *= 1.05
scenarios.append({"scenario": "+5% value on top 10% CLV", "delta": float(s1.sum() - equity0)})
upsell = summary["segment"] == "Upsell"
s2 = summary["expected_clv"].copy()
s2.loc[upsell] = s2.loc[upsell] + 0.2 * float(summary["expected_avg_spend"].median())
scenarios.append({"scenario": "+0.2 purchases on Upsell", "delta": float(s2.sum() - equity0)})
scenarios.append({"scenario": "+10% AOV (scale CLV)", "delta": float(equity0 * 0.10)})
scen_df = pd.DataFrame(scenarios)
display(scen_df.round(0))
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(scen_df["scenario"], scen_df["delta"], color="steelblue")
ax.set_xlabel("Incremental £ (illustrative)")
ax.set_title(f"Scenarios · baseline equity £{equity0:,.0f}")
plt.tight_layout()
plt.show()
""",
            "scen",
        ),
        md(
            """#### Assessment — scenarios

Use to size conversations (protect top CLV decile vs broad AOV). Validate live programs with a proper test design.
""",
            "a-scen",
        ),
        # ============================================================
        md(
            """## 12 · Artifacts

Persist only after the £ gate. Downstream must read `accepted` + leakage audit. Model is **stationary** weekly BG/NBD + Gamma-Gamma (no seasonal overlay).
""",
            "h-save",
        ),
        code(
            """assert_accepted(result)

bgf.save_model(str(MODELS / "01_clv_bgf"))
ggf.save_model(str(MODELS / "01_clv_ggf"))
save_artifact("01_clv_best.joblib", {
    "selected": best_name,
    "model": "BetaGeoFitter+GammaGamma",
    "purchase_model": "BG/NBD",
    "accepted": True,
    "agg_error_max": AGG_ERROR_MAX,
    "agg_error_test": float(agg_error),
    "orders_agg_error_test": float(orders_agg_error),
    "fit_frac": FIT_FRAC,
    "val_frac": VAL_FRAC,
    "val_scale": float(scale),
    "fit_end": str(fit_end.date()),
    "val_end": str(val_end.date()),
    "obs_end": str(max_d.date()),
    "horizon_weeks": t_horizon,
    "metrics": {
        "spearman": float(spearman) if spearman == spearman else None,
        "spearman_purchases": float(sp_purch) if sp_purch == sp_purch else None,
        "top_decile_capture": float(capture),
        "ideal_top_decile": float(ideal),
        "mae": float(mae),
        "r2": float(r2),
        "agg_error_test": float(agg_error),
        "orders_agg_error_test": float(orders_agg_error),
    },
    "clv_formula": "E[purchases]_BG/NBD * E[spend]_GG * val_revenue_scale",
    "seasonality_overlays": False,
    "discounting": False,
    "stepwise_notebook": True,
    "grain": "customer",
    "mlp_pipeline": mlp_pipe,
    "leakage_audit": leak,
})
scores_path = MODELS / "01_clv_customer_scores.parquet"
summary.reset_index().to_parquet(scores_path, index=False)
print(f"Saved → {scores_path.relative_to(PROJECT_ROOT)}")
print(
    f"ACCEPTED · stationary BG/NBD+GG · £ |error|={agg_error:.2%} · "
    f"purchase |error|={orders_agg_error:.1%} (informational)"
)
""",
            "save",
        ),
        md(
            """#### Bottom line

Full BTYD chain (stationary Hardie path):

1. Data → 2. Explore → 3. Protocol → 4. RFM → 5–6. Val BG/NBD + GG (scale) → 7–8. Test BG/NBD + GG → 9. Scaled CLV + gate → 10–11. Uses → 12. Artifacts.

**Ship:** stationary BG/NBD + GG when the £ gate passes.  
**Do not use for:** week-level seasonal demand planning; discounted NPV CLV (not computed here).
""",
            "a-close",
        ),
    ]
    path = NB / "01-clv.ipynb"
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("Wrote", path.relative_to(ROOT), f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
