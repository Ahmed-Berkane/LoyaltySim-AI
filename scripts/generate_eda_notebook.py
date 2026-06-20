"""Generate Notebooks/EDA.ipynb for UCI Online Retail II spine."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Notebooks" / "EDA.ipynb"


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


def md(text: str, cid: str = "") -> dict:
    return {"cell_type": "markdown", "id": cid or None, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cid: str = "") -> dict:
    return {
        "cell_type": "code", "id": cid or None, "metadata": {},
        "execution_count": None, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


SETUP = '''from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
np.random.seed(SEED)


def find_project_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in (path, *path.parents):
        if (candidate / "scripts" / "build_datasets.py").exists():
            return candidate
    return path


PROJECT_ROOT = find_project_root()
DATA = PROJECT_ROOT / "data" / "modeling"

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c0392b",
    "secondary": "#2ecc71",
    "neutral": "#7f8c8d",
    "tier": {"Bronze": "#cd7f32", "Silver": "#95a5a6", "Gold": "#f1c40f", "Platinum": "#9b59b6"},
}
TIER_ORDER = ["Bronze", "Silver", "Gold", "Platinum"]

plt.rcParams.update({
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.edgecolor": "#cccccc",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "font.family": "sans-serif",
})

def fmt_gbp(x, _pos=None):
    if abs(x) >= 1e6:
        return f"£{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"£{x/1e3:.0f}K"
    return f"£{x:,.0f}"

def gini_coefficient(values: pd.Series) -> float:
    x = np.sort(values.values.astype(float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cum = np.cumsum(x) / x.sum()
    return float(1 - 2 * np.trapezoid(cum, dx=1 / n))

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return (a.mean() - b.mean()) / pooled if pooled else np.nan

FEATURE_GROUPS = {
    "Core": ["R_score", "F_score", "M_score", "RFM_score", "avg_basket_size", "frequency_variance", "discount_dependency"],
    "Behavioral": ["engagement_score", "email_response_rate", "app_activity_velocity"],
    "Temporal": ["seasonality_index", "payday_effect", "holiday_sensitivity"],
    "Loyalty": ["tier_progression_speed", "reward_redemption_rate", "churn_inertia_score"],
}
FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]
print("Config loaded · parquet directory:", DATA)
'''

LOAD = '''def load(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run: python scripts/uci_pipeline.py")
    return pd.read_parquet(path)

fact = load("uci_fact_transactions.parquet")
fact["order_date"] = pd.to_datetime(fact["order_date"])
dim = load("uci_dim_customers.parquet")
feat = load("uci_customer_features.parquet")
uplift = load("uci_uplift_campaigns.parquet")
nba_events = load("uci_nba_offer_events.parquet")
nba_catalog = load("uci_nba_offer_catalog.parquet")

# Optional cross-universe benchmarks (still from build_datasets.py)
telco_path = DATA / "telco_customers.parquet"
telco = pd.read_parquet(telco_path) if telco_path.exists() else None

orders = (
    fact.groupby(["order_id", "customer_id", "country"], as_index=False)
    .agg(order_revenue=("line_total", "sum"), order_lines=("transaction_id", "count"), order_date=("order_date", "min"))
)

print(f"Lines: {len(fact):,} · Orders: {fact['order_id'].nunique():,} · Customers: {fact['customer_id'].nunique():,}")
print(f"Date range: {fact['order_date'].min().date()} → {fact['order_date'].max().date()}")
print(f"Countries: {fact['country'].nunique()} · Total revenue: £{fact['line_total'].sum():,.0f}")
fact.head(3)
'''

cells = [
    md("""# LoyaltySim-AI — UCI Online Retail II EDA

**Prerequisite:** `python scripts/uci_pipeline.py` (or run `Notebooks/get_UCI_data.ipynb` end-to-end)

**Spine:** UCI Online Retail II — EU countries, ~800k invoice lines, ~5.8k customers

| Parquet | Grain | Use |
|---------|-------|-----|
| `uci_fact_transactions` | Invoice line | CLV, seasonality, context |
| `uci_dim_customers` | Customer | Rollups |
| `uci_customer_features` | Customer | RFM, segments, engineered features |
| `uci_uplift_campaigns` | Customer × campaign wave | Uplift modeling |
| `uci_nba_offer_*` | Offer / event | Next-best-action |

> Synthetic CRM, uplift, and NBA are **seeded simulations** — not empirical ground truth.
""", "intro"),
    md("## Setup", "h-setup"),
    code(SETUP, "setup"),
    md("## Load data", "h-load"),
    code(LOAD, "load"),
    md("## 1 · Data quality", "h1"),
    code("""null_pct = (fact.isnull().mean() * 100).sort_values(ascending=False)
context_cols = [c for c in fact.columns if c.startswith(("is_", "days_", "cpi_", "inflation_", "unemployment_", "interest_", "temp_", "rain_", "had_"))]
show_cols = [c for c in null_pct.index if c in context_cols or c in ("line_total", "unit_price", "country")]
null_df = null_pct[show_cols].to_frame("null_pct").query("null_pct > 0")

fig, ax = plt.subplots(figsize=(8, max(3, len(null_df) * 0.35)))
if null_df.empty:
    ax.text(0.5, 0.5, "No nulls in core transaction columns", ha="center", va="center")
else:
    null_df["null_pct"].plot(kind="barh", ax=ax, color=PALETTE["accent"])
    ax.set_xlabel("Null %")
    ax.set_title("Missingness — context columns")
plt.tight_layout()
plt.show()

dup_lines = fact.duplicated(subset=["transaction_id"]).sum()
print(f"Duplicate transaction_id: {dup_lines:,}")
print(f"Rows after dropna(all): {len(fact.dropna()):,} ({len(fact.dropna())/len(fact):.1%})")
""", "q1"),
    md("## 2 · Revenue engine", "h2"),
    code("""monthly = (
    fact.assign(month=fact["order_date"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(revenue=("line_total", "sum"), orders=("order_id", "nunique"), customers=("customer_id", "nunique"))
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(monthly["month"], monthly["revenue"], marker="o", color=PALETTE["primary"])
axes[0].set_title("Monthly revenue")
axes[0].tick_params(axis="x", rotation=45)
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(fmt_gbp))

country_rev = dim.groupby("country")["total_revenue"].sum().sort_values(ascending=False).head(10)
country_rev.iloc[::-1].plot(kind="barh", ax=axes[1], color=PALETTE["secondary"])
axes[1].set_title("Top 10 countries by customer revenue")
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(fmt_gbp))
plt.tight_layout()
plt.show()

gini = gini_coefficient(dim["total_revenue"])
top10 = dim.nlargest(int(len(dim) * 0.1), "total_revenue")["total_revenue"].sum() / dim["total_revenue"].sum()
print(f"Gini (customer revenue): {gini:.3f}")
print(f"Top 10% customers → {top10:.1%} of revenue")
monthly[["month", "revenue", "orders"]].tail(6)
""", "rev"),
    md("## 3 · Customer economics", "h3"),
    code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.scatterplot(
    data=feat, x="recency_days", y="total_revenue", hue="rfm_segment",
    alpha=0.5, s=30, ax=axes[0], legend=False,
)
axes[0].set_title("Recency vs lifetime revenue (RFM colored)")
axes[0].set_xlabel("Recency (days)")
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(fmt_gbp))

seg_counts = feat["rfm_segment"].value_counts(normalize=True).mul(100).round(1)
seg_counts.plot(kind="bar", ax=axes[1], color=PALETTE["primary"])
axes[1].set_title("RFM segment distribution (%)")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

tier_aov = orders.merge(feat[["customer_id", "tier"]], on="customer_id").groupby("tier")["order_revenue"].mean()
tier_aov = tier_aov.reindex(TIER_ORDER)
print("AOV by tier:\\n", tier_aov.round(2).to_string())
display(feat["rfm_segment"].value_counts().to_frame("customers"))
""", "cust"),
    md("## 4 · Feature store", "h4"),
    code("""present = [c for c in FEATURE_COLS if c in feat.columns]
fig, ax = plt.subplots(figsize=(10, 5))
feat[present].select_dtypes(include=[np.number]).boxplot(ax=ax, rot=45)
ax.set_title("Feature store distributions")
plt.tight_layout()
plt.show()

corr = feat[present].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature correlation matrix")
plt.tight_layout()
plt.show()

churn_corr = corr["inactivity_churn_label"].drop("inactivity_churn_label").sort_values(key=abs, ascending=False)
print("Top correlations with inactivity churn label:")
display(churn_corr.head(8).to_frame("corr"))
""", "feat"),
    md("## 5 · Context & seasonality (EU)", "h5"),
    code("""order_ctx = orders.merge(
    fact.groupby("order_id", as_index=False).agg(
        is_holiday=("is_public_holiday", "max"),
        is_sale=("is_major_sale_period", "max"),
        is_xmas=("is_christmas_season", "max"),
        is_bf=("is_black_friday_week", "max"),
        temp_c=("temp_c", "mean"),
        had_rain=("had_rain", "max"),
    ),
    on="order_id",
)

lifts = []
for label, col in [
    ("Public holiday", "is_holiday"),
    ("Major sale period", "is_sale"),
    ("Christmas season", "is_xmas"),
    ("Black Friday week", "is_bf"),
]:
    a = order_ctx.loc[order_ctx[col], "order_revenue"]
    b = order_ctx.loc[~order_ctx[col], "order_revenue"]
    if len(a) > 10 and len(b) > 10:
        lifts.append({
            "event": label,
            "aov_on": a.mean(),
            "aov_off": b.mean(),
            "lift_pct": (a.mean() / b.mean() - 1) * 100,
            "p_value": stats.ttest_ind(a, b, equal_var=False).pvalue,
            "cohens_d": cohens_d(a, b),
        })

lift_df = pd.DataFrame(lifts).round(2)
display(lift_df)

wx = order_ctx.dropna(subset=["temp_c"])
rain_a = wx.loc[wx["had_rain"], "order_revenue"]
dry_a = wx.loc[~wx["had_rain"], "order_revenue"]
if len(rain_a) > 10:
    print(f"Rainy-day AOV: £{rain_a.mean():.2f} vs dry: £{dry_a.mean():.2f} (d={cohens_d(rain_a, dry_a):.2f})")
""", "ctx"),
    md("## 6 · Uplift & NBA (synthetic)", "h6"),
    code("""print(f"Uplift campaigns: {len(uplift):,} rows · treatment rate: {uplift['treatment'].mean():.1%}")
print(f"Response rate: {uplift['responded'].mean():.1%}")
print(f"NBA shown offers: {nba_events['shown'].sum():,} · conversion: {nba_events['converted'].mean():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
uplift.groupby("treatment")["responded"].mean().plot(kind="bar", ax=axes[0], color=[PALETTE["neutral"], PALETTE["accent"]])
axes[0].set_title("Response rate by treatment")
axes[0].set_xticklabels(["Control", "Treatment"], rotation=0)

nba_events["offer_type"].value_counts().head(8).plot(kind="barh", ax=axes[1], color=PALETTE["primary"])
axes[1].set_title("NBA offer type distribution")
plt.tight_layout()
plt.show()
""", "uplift"),
    md("## 7 · Telco churn benchmark (optional)", "h7"),
    code("""if telco is not None:
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4))
    churn_by_contract = telco.groupby("Contract")["churn_flag"].mean().sort_values(ascending=False)
    (churn_by_contract * 100).plot(kind="bar", ax=ax, color=PALETTE["accent"])
    ax.set_title("Telco churn rate by contract type")
    ax.set_ylabel("Churn %")
    plt.tight_layout()
    plt.show()
    print(f"Overall churn rate: {telco['churn_flag'].mean():.1%}")
else:
    print("Telco parquet not found — run build_datasets.py for cross-universe benchmark.")
""", "telco"),
    md("""## 8 · Modeling implications

| Model | Dataset | Key signal |
|-------|---------|------------|
| **01 Segmentation** | `uci_customer_features` | RFM + engagement + discount dependency |
| **02 Churn** | `telco_customers` (labeled) or UCI `inactivity_churn_label` (derived) |
| **03 CLV** | `uci_fact_transactions` | BG/NBD on `line_total`, ~5.8k customers |
| **04 Uplift** | `uci_uplift_campaigns` | Synthetic RCT on discount sensitivity |
| **05 NBA** | `uci_nba_offer_events` + features | Offer conversion simulation |

**Next:** run notebooks `01`–`05` (UCI paths) after this EDA.
""", "h8"),
]

NB.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
print(f"Wrote {NB.relative_to(ROOT)}")
