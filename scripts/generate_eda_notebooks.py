"""Generate UCI spine EDA notebooks (5 datasets).

Run: python scripts/generate_eda_notebooks.py
Outputs: Notebooks/eda/uci_*.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Notebooks" / "eda"

FEATURE_GROUPS = {
    "Core": [
        "R_score", "F_score", "M_score", "RFM_score",
        "avg_basket_size", "frequency_variance", "discount_dependency",
    ],
    "Behavioral": ["engagement_score", "email_response_rate", "app_activity_velocity"],
    "Temporal": ["seasonality_index", "payday_effect", "holiday_sensitivity"],
    "Loyalty": ["tier_progression_speed", "reward_redemption_rate", "churn_inertia_score"],
}
ENGINEERED_FEATURES = [c for cols in FEATURE_GROUPS.values() for c in cols]


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

warnings.filterwarnings("ignore")

SEED = 42
RANDOM_STATE = SEED
np.random.seed(SEED)
MIN_ROWS_3WAY = 500  # train / val / test when n >= this


def find_project_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in (path, *path.parents):
        if (candidate / "scripts" / "build_datasets.py").exists():
            return candidate
    return path


PROJECT_ROOT = find_project_root()
DATA = PROJECT_ROOT / "data" / "modeling"
SPLITS = PROJECT_ROOT / "data" / "splits"
SPLITS.mkdir(parents=True, exist_ok=True)

PALETTE = {"primary": "#1f4e79", "accent": "#c0392b", "secondary": "#2ecc71", "neutral": "#7f8c8d"}

plt.rcParams.update({
    "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.titleweight": "bold", "axes.titlesize": 13,
})


def load_parquet(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}\\nRun the prerequisite command from the notebook header.")
    return pd.read_parquet(path)


def audit_and_clean(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    id_col: str | None = None,
    required_cols: list[str] | None = None,
    label: str = "dataset",
) -> pd.DataFrame:
    out = df.copy()
    n0 = len(out)
    dup_subset = subset if subset is not None else ([id_col] if id_col else None)
    n_dup = (
        out.duplicated(subset=dup_subset, keep="first").sum()
        if dup_subset else out.duplicated(keep="first").sum()
    )
    if n_dup:
        out = out.drop_duplicates(subset=dup_subset, keep="first")
    req = [c for c in (required_cols or []) if c in out.columns]
    na_rows = out[req].isna().any(axis=1).sum() if req else 0
    na_by_col = out[req].isna().sum() if req else pd.Series(dtype=int)
    if req:
        out = out.dropna(subset=req)
    print(f"[{label}] {n0:,} -> {len(out):,} rows | dup dropped: {n_dup:,} | NA rows dropped: {na_rows:,}")
    if na_rows and (na_by_col > 0).any():
        print("  NA by column:", na_by_col[na_by_col > 0].to_dict())
    return out


def plot_nulls(df: pd.DataFrame, title: str = "Missing values (%)"):
    null_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    fig, ax = plt.subplots(figsize=(8, max(3, len(null_pct) * 0.35)))
    if null_pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=12)
        ax.set_title(title)
    else:
        null_pct.plot(kind="barh", ax=ax, color=PALETTE["accent"])
        ax.set_xlabel("Null %")
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return null_pct


def correlation_heatmap(frame: pd.DataFrame, title: str):
    if frame.shape[1] < 2:
        print("Need >= 2 numeric columns for correlation heatmap.")
        return
    corr = frame.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.45), max(6, len(corr) * 0.4)))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=len(corr) <= 12, fmt=".2f")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return corr


def prune_collinear(corr: pd.DataFrame, features: list[str], threshold: float = 0.85) -> list[str]:
    present = [c for c in features if c in corr.columns]
    if len(present) < 2:
        return present
    sub = corr.loc[present, present]
    upper = sub.where(np.triu(np.ones(sub.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if any(upper[c].abs() > threshold)}
    selected = [c for c in present if c not in drop]
    print(f"Collinearity prune (|r|>{threshold}): drop {sorted(drop) or 'none'}")
    print(f"Selected ({len(selected)}):", selected)
    return selected


def target_correlations(num_df: pd.DataFrame, target: str) -> pd.Series:
    if target not in num_df.columns:
        return pd.Series(dtype=float)
    return num_df.corr()[target].drop(target, errors="ignore").sort_values(key=abs, ascending=False)


def quick_feature_importance(X, y, task: str = "classification", top_n: int = 15):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    est = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1) if task == "classification" else RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", est)])
    pipe.fit(X, y)
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=X.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    imp.head(top_n).iloc[::-1].plot(kind="barh", ax=ax, color=PALETTE["primary"])
    ax.set_title(f"Random Forest feature importance (top {top_n})")
    plt.tight_layout()
    plt.show()
    return imp


def split_train_val_test(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    id_col: str | None = None,
    group_col: str | None = None,
    stratify_col: str | None = None,
    split_name: str = "split",
    task: str = "classification",
):
    """60/20/20 when n >= MIN_ROWS_3WAY; else 70/30 train/test."""
    from sklearn.model_selection import train_test_split

    work = df.dropna(subset=[c for c in feature_cols + [target_col] if c in df.columns]).copy()
    n = len(work)
    strat = work[stratify_col] if stratify_col and stratify_col in work.columns and task == "classification" else None

    if group_col and group_col in work.columns:
        groups = work[[group_col]].drop_duplicates()
        if strat is not None and stratify_col in work.columns:
            grp_label = work.groupby(group_col)[stratify_col].first()
            groups = groups.merge(grp_label.rename("_s"), left_on=group_col, right_index=True)
            strat_g = groups["_s"]
        else:
            strat_g = None
        if n >= MIN_ROWS_3WAY:
            g_train, g_temp = train_test_split(groups[group_col], test_size=0.40, random_state=RANDOM_STATE, stratify=strat_g)
            g_temp_df = groups[groups[group_col].isin(g_temp)]
            strat_temp = g_temp_df["_s"] if strat_g is not None and "_s" in g_temp_df.columns else None
            g_val, g_test = train_test_split(g_temp_df[group_col], test_size=0.50, random_state=RANDOM_STATE, stratify=strat_temp)
            train = work[work[group_col].isin(g_train)]
            val = work[work[group_col].isin(g_val)]
            test = work[work[group_col].isin(g_test)]
            scheme = "60/20/20 (grouped)"
        else:
            g_train, g_test = train_test_split(groups[group_col], test_size=0.30, random_state=RANDOM_STATE, stratify=strat_g)
            train = work[work[group_col].isin(g_train)]
            val = work.iloc[0:0]
            test = work[work[group_col].isin(g_test)]
            scheme = "70/30 train/test (grouped — too few rows for val)"
    else:
        X = work[feature_cols]
        y = work[target_col]
        if n >= MIN_ROWS_3WAY:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=strat)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp if strat is not None else None)
            train = work.loc[X_train.index]
            val = work.loc[X_val.index]
            test = work.loc[X_test.index]
            scheme = "60/20/20"
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=strat)
            train = work.loc[X_train.index]
            val = work.iloc[0:0]
            test = work.loc[X_test.index]
            scheme = "70/30 train/test (too few rows for val)"

    print(f"Split scheme: {scheme}")
    print(f"Train {len(train):,} · Val {len(val):,} · Test {len(test):,}")
    if task == "classification" and target_col in work.columns:
        for name, part in [("train", train), ("val", val), ("test", test)]:
            if len(part):
                print(f"  {name} {target_col} rate: {part[target_col].mean():.1%}")

    out_dir = SPLITS / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in [id_col, group_col, target_col, *feature_cols] if c and c in work.columns]
    train[cols].to_parquet(out_dir / "train.parquet", index=False)
    if len(val):
        val[cols].to_parquet(out_dir / "val.parquet", index=False)
    test[cols].to_parquet(out_dir / "test.parquet", index=False)
    print(f"Saved splits -> {out_dir.relative_to(PROJECT_ROOT)}")
    return train, val, test

print("EDA utilities loaded · data:", DATA)
'''


def make_notebook(cfg: dict) -> list:
    slug = cfg["slug"]
    title = cfg["title"]
    parquet = cfg["parquet"]
    prereq = cfg["prerequisite"]
    grain = cfg.get("grain", "")
    target = cfg.get("target")
    target_type = cfg.get("target_type", "none")
    id_col = cfg.get("id_col")
    dup_subset = cfg.get("dup_subset")
    required = cfg.get("required_cols", [])
    feature_cols_expr = cfg.get("feature_cols_expr", "FEATURE_COLS")
    extra_load = cfg.get("extra_load", "")
    extra_eda = cfg.get("extra_eda", "")
    skip_modeling = cfg.get("skip_modeling", False)
    sample_n = cfg.get("sample_n")
    group_col = cfg.get("group_col")
    stratify_col = cfg.get("stratify_col") or target
    filter_expr = cfg.get("filter_expr", "")
    task = "regression" if target_type == "regression" else "classification"

    intro = f"""# EDA — {title}

**Prerequisite:** `{prereq}`  
**Source:** `data/modeling/{parquet}`  
**Grain:** {grain}

Pre-modeling checklist: overview · duplicates & NAs · correlation · target vs features · feature importance · collinearity pruning · train/val/test split (when n ≥ {500}).
"""

    cells = [
        md(intro, "intro"),
        md("## Setup", "h-setup"),
        code(SETUP, "setup"),
        md("## 1 · Load & overview", "h1"),
    ]

    load_code = f'''PARQUET = "{parquet}"
SPLIT_NAME = "{slug}"
ID_COL = {repr(id_col)}
TARGET = {repr(target)}
TARGET_TYPE = {repr(target_type)}
GROUP_COL = {repr(group_col)}

df_raw = load_parquet(PARQUET)
{extra_load}
print(f"Shape: {{df_raw.shape[0]:,}} rows × {{df_raw.shape[1]}} cols")
print("Columns:", list(df_raw.columns))
display(df_raw.head(3))
display(df_raw.describe(include="all").T.head(20))
'''
    cells.append(code(load_code, "load"))

    dup_expr = repr(dup_subset) if dup_subset else ("None" if not id_col else f'["{id_col}"]')
    cells.extend([
        md("## 2 · Data quality (duplicates & missing)", "h2"),
        code(f'''req = {required!r}
df = audit_and_clean(
    df_raw,
    subset={dup_expr},
    id_col={repr(id_col)},
    required_cols=req,
    label="{slug}",
)
nulls = plot_nulls(df, "Missing values (%)")
dup_report = df_raw.duplicated(subset={dup_expr}).sum() if {dup_expr} else df_raw.duplicated().sum()
print(f"Duplicate rows (raw, before clean): {{dup_report:,}}")
print(f"Memory: {{df.memory_usage(deep=True).sum() / 1e6:.1f}} MB")
''', "quality"),
    ])

    if sample_n:
        cells.append(code(f'''# Sample for heavy plots / correlation on large tables
SAMPLE_N = {sample_n}
eda_df = df.sample(n=min(SAMPLE_N, len(df)), random_state=SEED) if len(df) > SAMPLE_N else df
print(f"EDA sample: {{len(eda_df):,}} / {{len(df):,}} rows")
''', "sample"))
        eda_var = "eda_df"
    else:
        cells.append(code("eda_df = df.copy()", "sample"))
        eda_var = "eda_df"

    cells.extend([
        md("## 3 · Target & univariate distributions", "h3"),
    ])

    if target and target_type != "none":
        if target_type == "binary":
            target_plot = f'''fig, ax = plt.subplots(figsize=(6, 4))
{eda_var}[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color=PALETTE["primary"])
ax.set_title(f"Target: {{TARGET}}")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
rate = {eda_var}[TARGET].mean()
ax.text(0.02, 0.95, f"positive rate: {{rate:.1%}}", transform=ax.transAxes, va="top")
plt.tight_layout()
plt.show()

num_cols = [c for c in {eda_var}.select_dtypes(include=[np.number]).columns if c != TARGET][:6]
if num_cols:
    {eda_var}[num_cols].hist(bins=30, layout=(2, 3), figsize=(11, 4))
    plt.suptitle("Sample numeric feature distributions", y=1.02)
    plt.tight_layout()
    plt.show()
'''
        elif target_type == "regression":
            target_plot = f'''fig, ax = plt.subplots(figsize=(7, 4))
{eda_var}[TARGET].hist(bins=40, ax=ax, color=PALETTE["primary"])
ax.set_title(f"Target distribution: {{TARGET}}")
ax.axvline({eda_var}[TARGET].median(), color=PALETTE["accent"], ls="--", label="median")
ax.legend()
plt.tight_layout()
plt.show()
print({eda_var}[TARGET].describe().round(2))
'''
        else:
            target_plot = f'''{eda_var}[TARGET].value_counts().head(15).plot(kind="barh", color=PALETTE["primary"])
plt.title(f"Target: {{TARGET}}")
plt.tight_layout()
plt.show()
'''
        cells.append(code(target_plot + (extra_eda or ""), "target"))
    else:
        cells.append(code(f'''# Reference / catalog table — no modeling target
{eda_var}.describe(include="all").T
{extra_eda}
''', "target"))

    if not skip_modeling and target and target_type != "none":
        cells.extend([
            md("## 4 · Correlation & target relationships", "h4"),
            code(f'''{feature_cols_expr}
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print("Modeling features:", FEATURE_COLS)

num = {eda_var}[[c for c in FEATURE_COLS + [TARGET] if c in {eda_var}.columns]].select_dtypes(include=[np.number])
corr = correlation_heatmap(num, "Feature correlation matrix")

if TARGET in num.columns:
    tc = target_correlations(num, TARGET)
    fig, ax = plt.subplots(figsize=(7, max(4, min(12, len(tc)) * 0.35)))
    tc.head(15).iloc[::-1].plot(kind="barh", ax=ax, color=PALETTE["secondary"])
    ax.set_title(f"Top correlations with {{TARGET}}")
    plt.tight_layout()
    plt.show()
    display(tc.head(10).to_frame("corr_with_target"))
''', "corr"),
            md("## 5 · Target vs feature plots", "h5"),
            code(f'''plot_cols = [c for c in FEATURE_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:6]
if not plot_cols:
    print("No numeric features to plot.")
else:
    n = len(plot_cols)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, col in enumerate(plot_cols):
        ax = axes[i]
        if TARGET_TYPE == "binary" and TARGET in df.columns:
            groups = df.groupby(TARGET)[col]
            ax.boxplot([groups.get_group(g).dropna() for g in sorted(df[TARGET].dropna().unique())], labels=sorted(df[TARGET].unique()))
            ax.set_title(f"{{col}} by {{TARGET}}")
        else:
            ax.scatter(df[col], df[TARGET], alpha=0.35, s=12, color=PALETTE["primary"])
            ax.set_xlabel(col)
            ax.set_ylabel(TARGET)
            ax.set_title(f"{{TARGET}} vs {{col}}")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()
''', "plots"),
            md("## 6 · Feature importance & selection", "h6"),
            code(f'''{filter_expr}
model_df = df{".query(\"" + filter_expr + "\")" if filter_expr else ""}.copy()
X_imp = model_df[[c for c in FEATURE_COLS if c in model_df.columns]].select_dtypes(include=[np.number])
y_imp = model_df[TARGET]
if len(X_imp.columns) and len(model_df) >= 30:
    imp = quick_feature_importance(X_imp, y_imp, task="{task}")
    display(imp.head(10).to_frame("importance"))
else:
    imp = None
    print("Skip importance — insufficient rows or features.")

if 'corr' in dir() and corr is not None:
    SELECTED = prune_collinear(corr, FEATURE_COLS)
else:
    SELECTED = FEATURE_COLS
''', "importance"),
            md("## 7 · Train / validation / test split", "h7"),
            code(f'''split_features = [c for c in (SELECTED if 'SELECTED' in dir() else FEATURE_COLS) if c in df.columns]
if len(split_features) and TARGET in df.columns and len(df) >= 50:
    train, val, test = split_train_val_test(
        df{".query(\"" + filter_expr + "\")" if filter_expr else ""},
        feature_cols=split_features,
        target_col=TARGET,
        id_col=ID_COL,
        group_col=GROUP_COL,
        stratify_col={repr(stratify_col)},
        split_name=SPLIT_NAME,
        task="{task}",
    )
else:
    print("Split skipped — need target, features, and >= 50 rows.")
''', "split"),
        ])
    elif not skip_modeling:
        cells.append(md("## 4 · Correlation (numeric)", "h4"))
        cells.append(code(f'''{feature_cols_expr}
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
num = {eda_var}[FEATURE_COLS].select_dtypes(include=[np.number]) if FEATURE_COLS else {eda_var}.select_dtypes(include=[np.number])
if num.shape[1] >= 2:
    correlation_heatmap(num, "Numeric correlation")
else:
    num.describe().T
''', "corr_only"))

    cells.append(md(f"""## Summary

| Item | Value |
|------|-------|
| **Dataset** | `{parquet}` |
| **Target** | `{target or "—"}` |
| **Split saved** | `data/splits/{slug}/` |

Proceed to the matching modeling notebook in `Notebooks/01`–`05` after reviewing signals above.
""", "summary"))

    return cells


DATASETS = [
    {
        "slug": "uci_fact_transactions",
        "title": "Fact Transactions (UCI Online Retail II)",
        "parquet": "uci_fact_transactions.parquet",
        "prerequisite": "python scripts/uci_pipeline.py",
        "grain": "One row = one invoice line",
        "target": "line_total",
        "target_type": "regression",
        "id_col": "transaction_id",
        "dup_subset": ["transaction_id"],
        "required_cols": ["transaction_id", "line_total", "quantity", "unit_price"],
        "feature_cols_expr": """FEATURE_COLS = [
    "quantity", "unit_price", "temp_c", "rain_mm", "precip_mm",
    "cpi_index", "inflation_mom", "unemployment_rate", "interest_rate",
    "app_usage_score", "discount_sensitivity",
]
""",
        "extra_load": 'df_raw["order_date"] = pd.to_datetime(df_raw["order_date"], errors="coerce")',
        "sample_n": 100_000,
    },
    {
        "slug": "uci_dim_customers",
        "title": "Dim Customers (UCI Online Retail II)",
        "parquet": "uci_dim_customers.parquet",
        "prerequisite": "python scripts/uci_pipeline.py",
        "grain": "One row = one UCI customer (aggregates)",
        "target": "total_revenue",
        "target_type": "regression",
        "id_col": "customer_id",
        "dup_subset": ["customer_id"],
        "required_cols": ["customer_id", "total_revenue"],
        "feature_cols_expr": """FEATURE_COLS = [
    "total_orders", "total_lines", "avg_unit_price", "avg_order_value",
    "recency_days", "tenure_days", "points_balance", "app_usage_score", "discount_sensitivity",
]
""",
    },
    {
        "slug": "uci_customer_features",
        "title": "Customer Features (UCI Online Retail II)",
        "parquet": "uci_customer_features.parquet",
        "prerequisite": "python scripts/uci_pipeline.py",
        "grain": "One row = one UCI customer",
        "target": "inactivity_churn_label",
        "target_type": "binary",
        "id_col": "customer_id",
        "dup_subset": ["customer_id"],
        "required_cols": ["customer_id", "inactivity_churn_label"],
        "feature_cols_expr": f"FEATURE_COLS = {ENGINEERED_FEATURES!r} + ['recency_days', 'total_orders', 'total_revenue']",
        "stratify_col": "inactivity_churn_label",
    },
    {
        "slug": "uci_uplift_campaigns",
        "title": "Uplift Campaigns (UCI, synthetic RCT)",
        "parquet": "uci_uplift_campaigns.parquet",
        "prerequisite": "python scripts/uci_pipeline.py",
        "grain": "One row = customer × campaign wave",
        "target": "responded",
        "target_type": "binary",
        "id_col": "customer_id",
        "dup_subset": ["customer_id", "campaign_id"],
        "required_cols": ["customer_id", "campaign_id", "treatment", "responded"],
        "feature_cols_expr": "FEATURE_COLS = ['discount_sensitivity', 'engagement_score', 'avg_order_value', 'treatment', 'offer_discount_pct']",
        "group_col": "customer_id",
        "stratify_col": "responded",
    },
    {
        "slug": "uci_nba_offer_events",
        "title": "NBA Offer Events (UCI, simulated)",
        "parquet": "uci_nba_offer_events.parquet",
        "prerequisite": "python scripts/uci_pipeline.py",
        "grain": "One row = one offer exposure per customer",
        "target": "converted",
        "target_type": "binary",
        "id_col": "customer_id",
        "dup_subset": ["customer_id", "offer_id"],
        "required_cols": ["customer_id", "offer_id", "converted", "shown"],
        "feature_cols_expr": """FEATURE_COLS = ['discount_pct', 'shown', 'clicked']
feat_path = DATA / "uci_customer_features.parquet"
if feat_path.exists():
    feat = pd.read_parquet(feat_path)
    eng = [c for c in feat.columns if c in {
        "R_score", "F_score", "M_score", "RFM_score", "engagement_score",
        "discount_sensitivity", "recency_days", "churn_inertia_score",
    }]
    df = df.merge(feat[["customer_id", *eng]], on="customer_id", how="left")
    FEATURE_COLS = FEATURE_COLS + eng
""",
        "filter_expr": "shown == 1",
        "stratify_col": "converted",
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {f"{cfg['slug']}.ipynb" for cfg in DATASETS}
    for stale in OUT.glob("*.ipynb"):
        if stale.name not in keep:
            stale.unlink()
            print("Removed", stale.relative_to(ROOT))
    written = []
    for cfg in DATASETS:
        path = OUT / f"{cfg['slug']}.ipynb"
        path.write_text(json.dumps(nb(make_notebook(cfg)), indent=1), encoding="utf-8")
        written.append(path.relative_to(ROOT))
        print("Wrote", path.relative_to(ROOT))
    print(f"\nGenerated {len(written)} UCI EDA notebooks in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
