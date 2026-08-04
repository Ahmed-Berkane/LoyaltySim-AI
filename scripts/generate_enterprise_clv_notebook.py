"""Generate Notebooks/02-clv-enterprise.ipynb — academic Hardie vs enterprise foil.

Keeps 00 / 01 untouched. Same calendar protocol; richer features + HistGradientBoosting
+ Empirical-Bayes country/cohort pooling. Documents what we cannot do on UCI
(clickstreams, transformers, true contribution margin, ad bidding APIs).
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
from clv_enterprise import (
    AGG_ERROR_MAX,
    FEATURE_CAT,
    FEATURE_NUM,
    assert_accepted,
    compare_to_hardie,
    run_enterprise_three_way,
)
from clv_weekly import FIT_FRAC, VAL_FRAC, run_three_way

print(
    f"Setup · enterprise HGB foil · same fit/val/test="
    f"{FIT_FRAC:.0%}/{VAL_FRAC:.0%}/{1 - FIT_FRAC - VAL_FRAC:.0%} · £ gate ≤ {AGG_ERROR_MAX:.0%}"
)
'''


def main() -> None:
    cells = [
        md(
            """# 02 · Enterprise-style CLV foil (keep Hardie `01` as baseline)

`00` (audit) and `01` (weekly BG/NBD + Gamma-Gamma) stay the **consulting / Hardie** path.

This notebook asks: *what would a DoorDash / Uber / Instacart-style **enterprise** stack look like on the same UCI spine?*

| World | Typical stack | On this project |
|-------|----------------|-----------------|
| Academic (Fader & Hardie) | RFM → BG/NBD + GG | **`01-clv.ipynb`** (shipped equity) |
| Marketplace / growth | Hundreds of features, GBT/NN, Bayesian pooling, real-time bidding | **This notebook** — feature-rich **HistGradientBoosting** + country/cohort Empirical-Bayes priors |
| Deep sequence CLV | Transformers on clickstreams / CASPR embeddings | **Not available** — UCI has no app opens, search, or cart events |

### Academic vs enterprise (what we can / cannot mirror)

| Feature | Academic papers | Real-world enterprise | Here |
|---------|-----------------|----------------------|------|
| Inputs | R, F, T (+ monetary) | Clickstreams, support, promo, sessions | **~25+** order/line features + country/cohort |
| Target | Expected purchases / gross revenue | Net contribution margin | **Gross holdout £** (no COGS/shipping on UCI) |
| Method | Beta / Gamma / Poisson | CatBoost / XGBoost / DNN / Bayesian BTYD | **sklearn HistGradientBoosting** + EB shrinkage |
| New users | Cold-start hard | Zero-party + first session | **One-timer slice** scored from first-order / country priors |
| Activation | Cohort insights | Ad bidding APIs (tROAS) | **Scores only** — no Meta/Google wiring |

**Protocol:** same 45% / 20% / 35% calendar as `01`. Train labels = **validation** holdout £ only; test never enters fit or scale.
""",
            "intro",
        ),
        code(SETUP, "setup"),
        # ---- 1 why ----
        md(
            """## 1 · Why a hybrid / enterprise foil?

Tech marketplaces see **fast silent churn** and frequent **promo resurrection**. Classic BG/NBD assumes irreversible death and a stationary purchase rate — great for equity ranking, weak for:

- Seasonal delivery spikes without a covariate layer
- Sparse new cohorts (need pooling)
- Non-purchase signals (we lack them on UCI)

Bayesian hierarchical BTYD (PyMC/Stan) and sequence transformers are the production pattern when you have scale + event logs. We approximate the *spirit* with:

1. **Rich tabular features** from orders + invoice lines  
2. **Gradient boosting** for nonlinear interactions  
3. **Empirical-Bayes priors** by country / acquisition cohort (lightweight hierarchical pooling)

Full PyMC Bayesian BG/NBD and Transformer embeddings are out of scope until event-level data exists.
""",
            "h-why",
        ),
        # ---- 2 data ----
        md(
            """## 2 · Load data (same spine as `01`)

Orders from the audit; optional line fact for product diversity / country.
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
    fact0 = load_fact_transactions(DATA / "uci_fact_transactions.parquet")
    orders = attach_acquisition(build_orders(fact0))

fact_path = DATA / "uci_fact_transactions.parquet"
fact = pd.read_parquet(fact_path) if fact_path.exists() else None
print(f"Orders: {len(orders):,} · Customers: {orders['customer_id'].nunique():,}")
print(f"Fact lines: {0 if fact is None else len(fact):,}")
print(f"Span: {orders['order_date'].min().date()} → {orders['order_date'].max().date()}")
""",
            "load",
        ),
        # ---- 3 features ----
        md(
            """## 3 · Feature philosophy (enterprise inputs, retail constraints)

Instead of only `(frequency, recency, T, monetary)`, we build calibration-safe features:

- **RFM core** — still there for comparability with Hardie  
- **Velocity** — orders/spend in last 4w and 13w before `cal_end`  
- **Basket** — avg lines, n products, n categories, top-product share  
- **Volatility** — spend std, AOV  
- **Pooling** — country / cohort shrunk AOV & order-rate priors (Bayesian-flavored)  
- **Cold-start flag** — one-timer in calibration  

All features use dates **≤ calibration end** only. Holdout £ is the label, never a feature.
""",
            "h-feat",
        ),
        code(
            """print("Numeric features:")
for c in FEATURE_NUM:
    print(" ", c)
print("Categorical:", FEATURE_CAT)
""",
            "feat-list",
        ),
        # ---- 4 train ----
        md(
            """## 4 · Train enterprise / hybrid models

Same protocol as `01`. We fit two foils:

1. **HGB-only** — rich tabular features (marketplace GBT pattern)  
2. **HGB-hybrid** — same features **plus** Hardie `expected_clv_raw` as a stacked signal (academic + ML hybrid)

Target is `log1p(holdout £)` with val-only scale. Clearing the ≤5% gate is nice-to-have; Hardie stays default if the foil misses.
""",
            "h-train",
        ),
        code(
            """# Academic baseline first (same engine as 01)
hardie = run_three_way(orders)
print(
    f"Hardie TEST £ |error|={hardie['agg_error']:.2%} · "
    f"Spearman={hardie['summary']['expected_clv'].corr(hardie['summary']['actual_holdout_revenue'], method='spearman'):.3f}"
)

# Pure tabular HGB (no Hardie feature)
ent_pure = run_enterprise_three_way(orders, fact)
print(
    f"HGB-only TEST £ |error|={ent_pure['agg_error']:.2%} · accepted={ent_pure['accepted']} · "
    f"Spearman={ent_pure['metrics']['spearman']:.3f}"
)

# Hybrid: Hardie raw CLV as a feature + rich tabular (marketplace stacking pattern)
ent = run_enterprise_three_way(
    orders,
    fact,
    hardie_summary=hardie["summary"],
    hardie_val_summary=hardie["val_summary"],
)
summary = ent["summary"]
print(f"Model: {ent['model_name']} · hybrid_hardie_feature={ent['hybrid_hardie_feature']}")
print(
    f"Fit ≤ {ent['fit_end'].date()} · Val ≤ {ent['val_end'].date()} ({ent['hw_val']}w) · "
    f"Test → {ent['obs_end'].date()} ({ent['hw_test']}w)"
)
print(f"Val £ scale = {ent['scale']:.4f}")
print(
    f"TEST £: pred £{ent['test_pred']:,.0f} vs actual £{ent['test_actual']:,.0f} "
    f"| |error| = {ent['agg_error']:.2%} · accepted={ent['accepted']}"
)
m = ent["metrics"]
print(
    f"Spearman={m['spearman']:.3f} · MAE={m['mae']:.1f} · R2={m['r2']:.3f} · "
    f"top-decile capture={m['top_decile_capture']:.1%} (ideal {m['ideal_top_decile']:.1%})"
)
cold_err = m["cold_start_£_error"]
print(
    f"Cold-start one-timers: n={m['n_cold']:,} · £ |error| on slice="
    f"{(cold_err if cold_err == cold_err else float('nan')):.1%}"
)
print("Leakage audit:", ent["leakage_audit"]["passed"])
if ent["accepted"]:
    assert_accepted(ent)
    print("Enterprise hybrid ACCEPTED under ≤5% gate")
else:
    print("Enterprise hybrid did NOT clear ≤5% — keep shipping Hardie `01` for equity totals.")
display(summary[list(ent["feature_num"])[:8] + ["expected_clv", "actual_holdout_revenue"]].head(10))
""",
            "train",
        ),
        md(
            """#### Assessment — enterprise fit

Clearing the £ gate is **optional** for this foil. If it fails (common when test is more seasonal than val), Hardie `01` remains the shippable equity model.

Still useful: Spearman / top-decile capture vs Hardie, and cold-start £ on one-timers — marketplace reasons to keep a tabular hybrid even when aggregates drift.
""",
            "a-train",
        ),
        # ---- 5 compare ----
        md(
            """## 5 · Head-to-head vs Hardie `01`

Compare Hardie, HGB-only, and HGB-hybrid on overlapping customers (aggregate £ error + ranking).
""",
            "h-cmp",
        ),
        code(
            """cmp = compare_to_hardie(ent, hardie["summary"])
cmp_pure = compare_to_hardie(ent_pure, hardie["summary"])
# keep only the non-Hardie row from the pure run
cmp_pure = cmp_pure[cmp_pure["model"] != "Hardie BG/NBD+GG"].copy()
cmp_pure["model"] = "HGB-only (no Hardie feat)"
cmp_all = pd.concat([cmp, cmp_pure], ignore_index=True)
display(cmp_all.round(4))

rng = np.random.default_rng(RANDOM_STATE)
idx = summary.index.intersection(hardie["summary"].index)
sample = idx if len(idx) <= 2500 else pd.Index(rng.choice(idx.to_numpy(), 2500, replace=False))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, title in (
    (
        axes[0],
        hardie["summary"].loc[sample, "expected_clv"],
        f"Hardie · |err|={hardie['agg_error']:.2%}",
    ),
    (
        axes[1],
        summary.loc[sample, "expected_clv"],
        f"{ent['model_name']} · |err|={ent['agg_error']:.2%}",
    ),
):
    actual = summary.loc[sample, "actual_holdout_revenue"]
    lim = float(np.nanpercentile(np.concatenate([pred.to_numpy(), actual.to_numpy()]), 99))
    lim = max(lim, 1.0)
    ax.scatter(pred, actual, alpha=0.25, s=12, edgecolors="none")
    ax.plot([0, lim], [0, lim], color="crimson", lw=1.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted £")
    ax.set_ylabel("Actual holdout £")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.show()

print("Lower aggregate £ |error|:", cmp_all.sort_values("£_|error|").iloc[0]["model"])
print("Higher Spearman:", cmp_all.sort_values("Spearman", ascending=False).iloc[0]["model"])
""",
            "cmp",
        ),
        md(
            """#### Assessment — Hardie vs enterprise

- **Prefer Hardie** for interpretability, P(alive), expected purchases, and consulting narratives — especially if it alone clears the £ gate.  
- **Prefer hybrid HGB** only if ranking/capture improves *and* aggregate £ stays honest.  
- Neither is **contribution margin** CLV; both predict gross holdout revenue.
""",
            "a-cmp",
        ),
        # ---- 6 bayesian note ----
        md(
            """## 6 · Bayesian hierarchical BTYD (what we did instead)

Marketplace teams often run **Bayesian BG/NBD** in PyMC/Stan to:

- put priors on seasonal intensity  
- pool sparse cohorts  
- keep full posterior uncertainty for bidding risk

**Here:** Empirical-Bayes country/cohort priors feed the booster as features — pooling without a full MCMC stack.  

To go further later: `pymc` Bayesian BG/NBD on weekly RFM, or Hardie time-varying covariates (note 040) with season dummies. Not required to keep `01` shippable.
""",
            "h-bayes",
        ),
        # ---- 7 transformers note ----
        md(
            """## 7 · Sequence transformers & embeddings (explicitly out of scope)

Advanced teams embed clickstreams / cart events with Transformers (or CASPR) and feed the embedding into a CLV head.

UCI Online Retail II has **invoice lines only** — no session timeline. Building a fake clickstream would invent data.  

When you have event logs: sequence model → user state embedding → CLV / propensity head, still with a leakage-safe time split.
""",
            "h-seq",
        ),
        # ---- 8 activation ----
        md(
            """## 8 · Activation sketch (not wired)

Enterprise CLV usually lands in **bidding / CRM**, not a notebook chart:

1. Daily batch (or streaming) score → `customer_id`, `clv_pred`, `p_alive` / propensity  
2. Push to ads (Google tROAS / Meta value optimization) or promo eligibility  
3. Monitor aggregate predicted vs realized contribution, not just revenue  

This repo stops at **artifacts** under `models/02_clv_*`.
""",
            "h-act",
        ),
        # ---- 9 artifacts ----
        md(
            """## 9 · Artifacts

Persist enterprise scores alongside Hardie (`01`). Default consulting ship remains **`01`** unless this foil clearly wins ranking *and* clears the £ gate.
""",
            "h-save",
        ),
        code(
            """# Persist hybrid as a foil; accepted flag follows the £ gate
joblib.dump(
    {
        "model": ent["model"],
        "model_name": ent["model_name"],
        "accepted": bool(ent["accepted"]),
        "agg_error_test": float(ent["agg_error"]),
        "val_scale": float(ent["scale"]),
        "fit_end": str(ent["fit_end"].date()),
        "val_end": str(ent["val_end"].date()),
        "obs_end": str(ent["obs_end"].date()),
        "horizon_weeks": ent["t_horizon"],
        "metrics": ent["metrics"],
        "feature_num": ent["feature_num"],
        "feature_cat": FEATURE_CAT,
        "hybrid_hardie_feature": ent["hybrid_hardie_feature"],
        "hgb_only_agg_error": float(ent_pure["agg_error"]),
        "hardie_agg_error": float(hardie["agg_error"]),
        "hardie_spearman": float(
            hardie["summary"]["expected_clv"].corr(
                hardie["summary"]["actual_holdout_revenue"], method="spearman"
            )
        ),
        "comparison": cmp_all.to_dict(orient="records"),
        "target": "gross_holdout_revenue",
        "contribution_margin": False,
        "seasonality_overlays": False,
        "bayesian_full_mcmc": False,
        "empirical_bayes_pooling": True,
        "default_ship": "01-clv Hardie" if not ent["accepted"] else "02 hybrid (gate passed)",
        "leakage_audit": ent["leakage_audit"],
    },
    MODELS / "02_clv_enterprise.joblib",
)
scores_path = MODELS / "02_clv_enterprise_scores.parquet"
summary.reset_index().to_parquet(scores_path, index=False)
print(f"Saved → models/02_clv_enterprise.joblib (accepted={ent['accepted']})")
print(f"Saved → {scores_path.relative_to(PROJECT_ROOT)}")
""",
            "save",
        ),
        md(
            """#### Bottom line

| Keep | Role |
|------|------|
| `00` | Customer-base audit |
| `01` | Hardie weekly BG/NBD + GG — **default equity** |
| `02` (this) | Enterprise foil — HGB / Hardie+HGB hybrid + EB pooling |

Marketplace lesson on UCI: rich features help ranking stories; **stationary Hardie still often wins honest £ totals** when the holdout is seasonal. Transformers / PyMC MCMC / ad APIs wait for event-level data and ops wiring.
""",
            "a-close",
        ),
    ]
    path = NB / "02-clv-enterprise.ipynb"
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("Wrote", path.relative_to(ROOT), f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
