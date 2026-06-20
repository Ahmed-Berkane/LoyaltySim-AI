"""Generate 01–05 modeling notebooks. Run once: python scripts/generate_modeling_notebooks.py"""
from __future__ import annotations

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


def md(text: str, cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cid: str) -> dict:
    return {
        "cell_type": "code", "id": cid, "metadata": {},
        "execution_count": None, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


SETUP = '''from __future__ import annotations

import json
import warnings
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
        if (candidate / "scripts" / "build_datasets.py").exists():
            return candidate
    return path


PROJECT_ROOT = find_project_root()
DATA = PROJECT_ROOT / "data" / "modeling"
MODELS = PROJECT_ROOT / "models"
MODELS.mkdir(exist_ok=True)


def load_parquet(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run: python scripts/uci_pipeline.py")
    return pd.read_parquet(path)


def save_artifact(name: str, obj) -> Path:
    path = MODELS / name
    joblib.dump(obj, path)
    print(f"Saved → {path.relative_to(PROJECT_ROOT)}")
    return path


def audit_and_clean(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    id_col: str | None = None,
    required_cols: list[str] | None = None,
    label: str = "dataset",
) -> pd.DataFrame:
    """Report and drop duplicate rows + rows with NA in required columns (before split)."""
    out = df.copy()
    n0 = len(out)
    dup_subset = subset if subset is not None else ([id_col] if id_col else None)
    n_dup = out.duplicated(subset=dup_subset, keep="first").sum() if dup_subset else out.duplicated(keep="first").sum()
    if n_dup:
        out = out.drop_duplicates(subset=dup_subset, keep="first")
    req = [c for c in (required_cols or []) if c in out.columns]
    na_rows = out[req].isna().any(axis=1).sum() if req else 0
    na_by_col = out[req].isna().sum()
    if req:
        out = out.dropna(subset=req)
    print(
        f"[{label}] {n0:,} rows -> {len(out):,} | "
        f"dropped {n_dup:,} duplicates, {na_rows:,} rows with NA"
    )
    if na_rows and (na_by_col > 0).any():
        print("  NA counts:", na_by_col[na_by_col > 0].to_dict())
    return out
'''


def write(name: str, cells: list) -> None:
    path = NB / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("Wrote", path.relative_to(ROOT))


# ── 01 Segmentation ──────────────────────────────────────────────────────────
write("01-segmentation.ipynb", [
    md("""# 01 · Segmentation Model

**Data:** `uci_customer_features.parquet` (UCI Online Retail II)  
**Algorithms:** KMeans, GMM, Hierarchical (Agglomerative)  
**Business labels:** high value · discount seekers · at risk · new users

Run `python scripts/uci_pipeline.py` first.
""", "intro"),
    code(SETUP, "setup"),
    code("""feat = load_parquet("uci_customer_features.parquet")

CLUSTER_FEATURES = [
    "R_score", "F_score", "M_score", "RFM_score",
    "avg_basket_size", "discount_dependency", "engagement_score",
    "churn_inertia_score", "recency_days", "total_orders",
]
feat = audit_and_clean(
    feat,
    id_col="customer_id",
    required_cols=["customer_id", *CLUSTER_FEATURES],
    label="customer_features",
)
X_raw = feat[CLUSTER_FEATURES]
print(f"Customers: {len(X_raw):,} · Features: {len(CLUSTER_FEATURES)}")
X_raw.describe().T.round(2)
""", "load"),
    code("""# Correlation & feature pruning (|r| > 0.85)
corr = X_raw.corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature correlation — segmentation inputs")
plt.tight_layout()
plt.show()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c].abs() > 0.85)]
SELECTED = [c for c in CLUSTER_FEATURES if c not in drop_cols]
print("Dropped (high collinearity):", drop_cols)
print("Selected:", SELECTED)
X = X_raw[SELECTED]
""", "corr"),
    code("""from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_hold = train_test_split(X, test_size=0.30, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_hold_s = scaler.transform(X_hold)

K = 4  # business segments
candidates = {}

km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
km_labels = km.fit_predict(X_train_s)
candidates["KMeans"] = {
    "model": km, "train_labels": km_labels,
    "silhouette": silhouette_score(X_train_s, km_labels),
    "davies_bouldin": davies_bouldin_score(X_train_s, km_labels),
    "calinski": calinski_harabasz_score(X_train_s, km_labels),
}

gmm = GaussianMixture(n_components=K, random_state=RANDOM_STATE, n_init=5)
gmm_labels = gmm.fit_predict(X_train_s)
candidates["GMM"] = {
    "model": gmm, "train_labels": gmm_labels,
    "silhouette": silhouette_score(X_train_s, gmm_labels),
    "davies_bouldin": davies_bouldin_score(X_train_s, gmm_labels),
    "calinski": calinski_harabasz_score(X_train_s, gmm_labels),
}

agg = AgglomerativeClustering(n_clusters=K)
agg_labels = agg.fit_predict(X_train_s)
candidates["Hierarchical"] = {
    "model": agg, "train_labels": agg_labels,
    "silhouette": silhouette_score(X_train_s, agg_labels),
    "davies_bouldin": davies_bouldin_score(X_train_s, agg_labels),
    "calinski": calinski_harabasz_score(X_train_s, agg_labels),
}

metrics = pd.DataFrame({
    name: {k: v for k, v in d.items() if k != "model" and k != "train_labels"}
    for name, d in candidates.items()
}).T.round(4)
display(metrics)
best_name = metrics["silhouette"].idxmax()
print(f"Best by silhouette: {best_name}")
""", "train"),
    code("""# Map clusters → business segments (rule-based on cluster centroids)
best = candidates[best_name]
centroids = pd.DataFrame(scaler.inverse_transform(
    getattr(best["model"], "cluster_centers_", None)
    if best_name == "KMeans"
    else pd.DataFrame(X_train_s).groupby(best["train_labels"]).mean().values
), columns=SELECTED)
centroids["cluster"] = range(len(centroids))

def label_cluster(row) -> str:
    if row["R_score"] >= 4 and row["M_score"] >= 4:
        return "high value"
    if row["discount_dependency"] >= centroids["discount_dependency"].median():
        return "discount seekers"
    if row["recency_days"] if "recency_days" in row.index else row.get("churn_inertia_score", 0) >= centroids.get("churn_inertia_score", pd.Series([1])).median():
        return "at risk"
    if row["total_orders"] <= centroids["total_orders"].quantile(0.25):
        return "new users"
    return "core"

# Apply best model to full dataset
X_full_s = scaler.transform(X)
if best_name == "KMeans":
    labels = best["model"].predict(X_full_s)
elif best_name == "GMM":
    labels = best["model"].predict(X_full_s)
else:
    labels = best["model"].fit_predict(X_full_s)

feat_out = feat.copy()
feat_out["cluster_id"] = labels
centroid_df = feat_out.groupby("cluster_id")[SELECTED].mean()
business_map = {}
for cid, row in centroid_df.iterrows():
    if row["M_score"] >= 4 and row["F_score"] >= 3:
        business_map[cid] = "high value"
    elif row["discount_dependency"] >= feat_out["discount_dependency"].median():
        business_map[cid] = "discount seekers"
    elif row["recency_days"] >= 120 or row["churn_inertia_score"] >= 1.2:
        business_map[cid] = "at risk"
    elif row["total_orders"] <= feat_out["total_orders"].quantile(0.30):
        business_map[cid] = "new users"
    else:
        business_map[cid] = "core"
feat_out["segment_label"] = feat_out["cluster_id"].map(business_map)
display(feat_out["segment_label"].value_counts())

artifact = {
    "model_name": best_name,
    "model": best["model"],
    "scaler": scaler,
    "features": SELECTED,
    "business_map": business_map,
    "metrics": metrics.loc[best_name].to_dict(),
}
save_artifact("01_segmentation_best.joblib", artifact)
""", "save"),
])

# ── 02 Churn ─────────────────────────────────────────────────────────────────
write("02-churn-prediction.ipynb", [
    md("""# 02 · Churn Prediction Model

**Data:** `telco_customers.parquet` (labeled `churn_flag`)  
**Models:** XGBoost, LightGBM, MLP (neural baseline)  
**Output:** P(churn) — proxy for 30-day churn risk

Split: 60% train · 20% validation · 20% test (stratified).
""", "intro"),
    code(SETUP, "setup"),
    code("""from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    RocCurveDisplay, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

df = load_parquet("telco_customers.parquet")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

TARGET = "churn_flag"
DROP = ["telco_customer_id", "Churn", TARGET]
feature_cols = [c for c in df.columns if c not in DROP]
df = audit_and_clean(
    df,
    id_col="telco_customer_id",
    required_cols=["telco_customer_id", TARGET, *feature_cols],
    label="telco_customers",
)
print(f"Rows: {len(df):,} · Churn rate: {df['churn_flag'].mean():.1%}")

X = df.drop(columns=[c for c in DROP if c in df.columns])
y = df[TARGET]
""", "load"),
    code("""# Correlation (numeric only)
num = X.select_dtypes(include=[np.number])
num_corr = num.assign(churn_flag=y).corr()["churn_flag"].drop("churn_flag").sort_values(key=abs, ascending=False)
fig, ax = plt.subplots(figsize=(7, 5))
num_corr.head(15).plot(kind="barh", ax=ax, color="#1f4e79")
ax.set_title("Top numeric correlations with churn")
plt.tight_layout()
plt.show()
display(num_corr.head(10).to_frame("corr_with_churn"))
""", "corr"),
    code("""cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols),
])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)
print(f"Train {len(X_train):,} · Val {len(X_val):,} · Test {len(X_test):,}")
""", "split"),
    code("""def eval_classifier(name, model, X_tr, y_tr, X_v, y_v, X_te, y_te):
    model.fit(X_tr, y_tr)
    prob_v = model.predict_proba(X_v)[:, 1]
    prob_t = model.predict_proba(X_te)[:, 1]
    pred_t = (prob_t >= 0.5).astype(int)
    return {
        "model": name,
        "val_roc_auc": roc_auc_score(y_v, prob_v),
        "test_roc_auc": roc_auc_score(y_te, prob_t),
        "test_f1": f1_score(y_te, pred_t),
        "test_precision": precision_score(y_te, pred_t),
        "test_recall": recall_score(y_te, pred_t),
        "estimator": model,
    }

models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=RANDOM_STATE, verbosity=0,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
    ),
    "MLP": Pipeline([
        ("prep", preprocess),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)),
    ]),
}

results = []
for name, est in models.items():
    if name == "MLP":
        results.append(eval_classifier(name, est, X_train, y_train, X_val, y_val, X_test, y_test))
    else:
        pipe = Pipeline([("prep", preprocess), ("clf", est)])
        results.append(eval_classifier(name, pipe, X_train, y_train, X_val, y_val, X_test, y_test))

leaderboard = pd.DataFrame([{k: v for k, v in r.items() if k != "estimator"} for r in results]).set_index("model")
display(leaderboard.round(4))
best = max(results, key=lambda r: r["val_roc_auc"])
print(f"Best model: {best['model']} (val ROC-AUC={best['val_roc_auc']:.4f})")
""", "train"),
    code("""# Feature importance (tree model)
best_pipe = best["estimator"]
if best["model"] in ("XGBoost", "LightGBM"):
    prep = best_pipe.named_steps["prep"]
    clf = best_pipe.named_steps["clf"]
    X_train_t = prep.fit_transform(X_train, y_train)
    feat_names = prep.get_feature_names_out()
    imp = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.iloc[::-1].plot(kind="barh", ax=ax, color="#c0392b")
    ax.set_title(f"Top 20 feature importances — {best['model']}")
    plt.tight_layout()
    plt.show()

RocCurveDisplay.from_predictions(y_test, best_pipe.predict_proba(X_test)[:, 1], name=best["model"])
plt.title("ROC — hold-out test set")
plt.show()

save_artifact("02_churn_best.joblib", {
    "model_name": best["model"],
    "pipeline": best_pipe,
    "metrics": {k: best[k] for k in leaderboard.columns},
    "target": "churn_flag",
})
""", "save"),
])

# ── 03 CLV ───────────────────────────────────────────────────────────────────
write("03-clv.ipynb", [
    md("""# 03 · CLV Model

**Data:** `uci_fact_transactions.parquet` (UCI Online Retail II — ~5.8k customers, ~800k lines)  
**Classical:** BG/NBD + Gamma-Gamma (`lifetimes`)  
**Neural baseline:** MLP regressor on RFM features predicting holdout 90-day revenue

Temporal split: train on orders before cutoff · test on post-cutoff revenue.
""", "intro"),
    code(SETUP, "setup"),
    code("""from datetime import timedelta

txn = load_parquet("uci_fact_transactions.parquet")
txn["order_date"] = pd.to_datetime(txn["order_date"])
txn = audit_and_clean(
    txn,
    subset=["customer_id", "order_id", "product_id", "order_date", "quantity", "unit_price"],
    required_cols=["customer_id", "order_id", "order_date", "line_total"],
    label="uci_fact_transactions",
)
print(f"Transactions: {len(txn):,} · Customers: {txn['customer_id'].nunique():,}")
print(f"Span: {txn['order_date'].min().date()} → {txn['order_date'].max().date()}")

# lifetimes format: customer_id, frequency, recency, T, monetary_value
cutoff = txn["order_date"].quantile(0.75)
print(f"Calibration cutoff: {cutoff.date()}")

cal = txn[txn["order_date"] < cutoff].copy()
hold = txn[txn["order_date"] >= cutoff].copy()
hold_end = txn["order_date"].max()

from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

summary = summary_data_from_transaction_data(
    txn, "customer_id", "order_date", monetary_value_col="line_total",
    observation_period_end=hold_end,
)
summary.head()
""", "load"),
    code("""# Correlation on RFM summary
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(summary[["frequency", "recency", "T", "monetary_value"]].corr(), annot=True, fmt=".2f", ax=ax)
ax.set_title("CLV input correlation (lifetimes summary)")
plt.tight_layout()
plt.show()
""", "corr"),
    code("""from lifetimes import BetaGeoFitter, GammaGammaFitter

bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(summary["frequency"], summary["recency"], summary["T"])

# Gamma-Gamma needs repeat buyers
gg_data = summary[summary["frequency"] > 0].copy()
ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(gg_data["frequency"], gg_data["monetary_value"])

t_horizon = 90  # days
summary["p_alive"] = bgf.conditional_probability_alive(
    summary["frequency"], summary["recency"], summary["T"]
)
summary["expected_clv_90d"] = ggf.customer_lifetime_value(
    bgf, summary["frequency"], summary["recency"], summary["T"],
    summary["monetary_value"], time=t_horizon / 30.0,  # lifetimes uses months
)

print("BG/NBD + Gamma-Gamma fitted")
summary[["frequency", "monetary_value", "p_alive", "expected_clv_90d"]].describe().round(2)
""", "bgf"),
    code("""# Holdout validation: actual 90d revenue vs predicted
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

actual = hold.groupby("customer_id")["line_total"].sum().rename("actual_90d")
pred = summary["expected_clv_90d"]
eval_df = pd.concat([pred, actual], axis=1).dropna()
eval_df = eval_df[eval_df.index.isin(actual.index)]

if len(eval_df) > 50:
    mae = mean_absolute_error(eval_df["actual_90d"], eval_df["expected_clv_90d"])
    r2 = r2_score(eval_df["actual_90d"], eval_df["expected_clv_90d"])
    print(f"Probabilistic CLV holdout — MAE: {mae:.2f} · R2: {r2:.3f}")

# Neural baseline on RFM features (drop NA rows before split)
rfm = summary[["frequency", "recency", "T", "monetary_value"]].copy()
rfm["target"] = rfm.index.map(actual).fillna(0)
rfm = audit_and_clean(
    rfm.reset_index(),
    id_col="customer_id",
    required_cols=["customer_id", "frequency", "recency", "T", "monetary_value", "target"],
    label="clv_rfm_summary",
).set_index("customer_id")
X = rfm[["frequency", "recency", "T", "monetary_value"]]
y = rfm["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)),
])
mlp_pipe.fit(X_tr, y_tr)
mlp_pred = mlp_pipe.predict(X_te)
print(f"MLP CLV — MAE: {mean_absolute_error(y_te, mlp_pred):.2f} · R2: {r2_score(y_te, mlp_pred):.3f}")

# Pick best by R2 on test
best_name = "BG/NBD+Gamma-Gamma" if len(eval_df) > 50 and r2 > r2_score(y_te, mlp_pred) else "MLP"
print(f"Selected: {best_name}")
""", "neural"),
    code("""save_artifact("03_clv_best.joblib", {
    "selected": best_name,
    "bgf": bgf,
    "ggf": ggf,
    "mlp_pipeline": mlp_pipe,
    "horizon_days": t_horizon,
    "cutoff": str(cutoff.date()),
})
save_artifact("03_clv_customer_scores.parquet", summary.reset_index(), )  # will fail - joblib only

# Save scores as parquet separately
scores_path = MODELS / "03_clv_customer_scores.parquet"
summary.reset_index().to_parquet(scores_path, index=False)
print(f"Saved → {scores_path.relative_to(PROJECT_ROOT)}")
""", "save"),
])

# Fix CLV save cell - joblib doesn't save parquet that way
cells_clv = json.loads((NB / "03-clv.ipynb").read_text())["cells"]
cells_clv[-1]["source"] = """save_artifact("03_clv_best.joblib", {
    "selected": best_name,
    "bgf": bgf,
    "ggf": ggf,
    "mlp_pipeline": mlp_pipe,
    "horizon_days": t_horizon,
    "cutoff": str(cutoff.date()),
})
scores_path = MODELS / "03_clv_customer_scores.parquet"
summary.reset_index().to_parquet(scores_path, index=False)
print(f"Saved → {scores_path.relative_to(PROJECT_ROOT)}")
""".splitlines(keepends=True)
(NB / "03-clv.ipynb").write_text(json.dumps(nb(cells_clv), indent=1), encoding="utf-8")

# Patch CLV save cell to use lifetimes native serialization (joblib can't pickle fitters on Py3.13)
cells_clv_save = json.loads((NB / "03-clv.ipynb").read_text())["cells"]
cells_clv_save[-1]["source"] = """bgf.save_model(str(MODELS / "03_clv_bgf"))
ggf.save_model(str(MODELS / "03_clv_ggf"))
save_artifact("03_clv_best.joblib", {
    "selected": best_name,
    "bgf_path": "03_clv_bgf",
    "ggf_path": "03_clv_ggf",
    "mlp_pipeline": mlp_pipe,
    "horizon_days": t_horizon,
    "cutoff": str(cutoff.date()),
})
scores_path = MODELS / "03_clv_customer_scores.parquet"
summary.reset_index().to_parquet(scores_path, index=False)
print(f"Saved → {scores_path.relative_to(PROJECT_ROOT)}")
""".splitlines(keepends=True)
(NB / "03-clv.ipynb").write_text(json.dumps(nb(cells_clv_save), indent=1), encoding="utf-8")

# ── 04 Uplift ────────────────────────────────────────────────────────────────
write("04-uplift.ipynb", [
    md("""# 04 · Uplift Modeling

**Data:** `uci_uplift_campaigns.parquet` (synthetic RCT on UCI customers)  
**Learners:** T-learner, X-learner, Uplift Random Forest (class transformation)

**Output:** incremental impact of discount offer (CATE).

Split: 60/20/20 by customer (both campaign waves in same split group).
""", "intro"),
    code(SETUP, "setup"),
    code("""from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = load_parquet("uci_uplift_campaigns.parquet")
FEATURES = ["discount_sensitivity", "engagement_score", "avg_order_value"]
df = audit_and_clean(
    df,
    subset=["customer_id", "campaign_id"],
    required_cols=["customer_id", "campaign_id", "treatment", "responded", *FEATURES],
    label="uplift_campaigns",
)
X = df[FEATURES]
treatment = df["treatment"].astype(int)
y = df["responded"].astype(int)

print(f"Rows: {len(df):,} · Treatment rate: {treatment.mean():.1%} · Response rate: {y.mean():.1%}")
""", "load"),
    code("""corr = X.assign(treatment=treatment, responded=y).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Uplift feature correlation")
plt.tight_layout()
plt.show()
""", "corr"),
    code("""# Split by customer so both waves stay together
cust = df["customer_id"].astype(str).unique().to_numpy()
c_train, c_temp = train_test_split(cust, test_size=0.40, random_state=RANDOM_STATE)
c_val, c_test = train_test_split(c_temp, test_size=0.50, random_state=RANDOM_STATE)

def mask(ids): return df["customer_id"].isin(ids)
train, val, test = mask(c_train), mask(c_val), mask(c_test)
for name, split in [("train", train), ("val", val), ("test", test)]:
    print(f"{name}: {split.sum():,} rows")
""", "split"),
    code("""def uplift_at_k(y_true, uplift, treatment, k=0.3):
    order = np.argsort(uplift)[::-1]
    n = max(1, int(len(y_true) * k))
    top = order[:n]
    tr = treatment[top] == 1
    ct = ~tr
    if tr.sum() == 0 or ct.sum() == 0:
        return 0.0
    return y_true[top][tr].mean() - y_true[top][ct].mean()


def qini_auc_score(y_true, uplift, treatment):
    order = np.argsort(uplift)[::-1]
    y, w = y_true[order], treatment[order]
    cum_tr = np.cumsum(y * w)
    cum_ct = np.cumsum(y * (1 - w))
    cum_n_tr = np.cumsum(w)
    cum_n_ct = np.cumsum(1 - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = cum_tr - cum_n_tr * np.divide(cum_ct, cum_n_ct, out=np.zeros_like(cum_ct, dtype=float), where=cum_n_ct > 0)
    if len(qini) < 2:
        return 0.0
    return float(np.trapezoid(qini / len(qini)) - np.trapezoid(cum_n_tr / len(cum_n_tr) * (y[w == 1].mean() - y[w == 0].mean())))


class TLearner:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, w):
        X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
        self.m1_ = clone(self.estimator)
        self.m0_ = clone(self.estimator)
        self.m1_.fit(X[w == 1], y[w == 1])
        self.m0_.fit(X[w == 0], y[w == 0])
        return self

    def predict(self, X):
        X = np.asarray(X)
        p1 = self.m1_.predict_proba(X)[:, 1]
        p0 = self.m0_.predict_proba(X)[:, 1]
        return p1 - p0


class XLearner:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, w):
        X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
        t = TLearner(self.estimator)
        t.fit(X, y, w)
        mu1 = t.m1_.predict_proba(X)[:, 1]
        mu0 = t.m0_.predict_proba(X)[:, 1]
        d1 = y[w == 1] - mu0[w == 1]
        d0 = mu1[w == 0] - y[w == 0]
        from sklearn.ensemble import GradientBoostingRegressor
        self.tau1_ = GradientBoostingRegressor(random_state=RANDOM_STATE)
        self.tau0_ = GradientBoostingRegressor(random_state=RANDOM_STATE)
        self.tau1_.fit(X[w == 1], d1)
        self.tau0_.fit(X[w == 0], d0)
        self.p_ = w.mean()
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.p_ * self.tau0_.predict(X) + (1 - self.p_) * self.tau1_.predict(X)


class UpliftRandomForest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y, w):
        X, y, w = np.asarray(X), np.asarray(y), np.asarray(w)
        z = np.where(w == 1, y, 1 - y)
        self.model_ = RandomForestClassifier(**self.kwargs)
        self.model_.fit(X, z)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return 2 * self.model_.predict_proba(X)[:, 1] - 1


base_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    random_state=RANDOM_STATE, verbosity=0,
)

learners = {
    "T-learner": TLearner(base_clf),
    "X-learner": XLearner(base_clf),
    "UpliftRF": UpliftRandomForest(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1),
}
results = []

X_train = X.loc[train].values
y_train = y.loc[train].values
w_train = treatment.loc[train].values
X_test = X.loc[test].values
y_test = y.loc[test].values
w_test = treatment.loc[test].values

for name, model in learners.items():
    model.fit(X_train, y_train, w_train)
    uplift_scores = model.predict(X_test)
    qini = qini_auc_score(y_test, uplift_scores, w_test)
    uplift30 = uplift_at_k(y_test, uplift_scores, w_test, k=0.3)
    results.append({"model": name, "qini_auc": qini, "uplift_at_30pct": uplift30, "estimator": model})

leaderboard = pd.DataFrame([{k: v for k, v in r.items() if k != "estimator"} for r in results]).set_index("model")
display(leaderboard.round(4))
best = max(results, key=lambda r: r["qini_auc"])
print(f"Best: {best['model']}")
""", "train"),
    code("""save_artifact("04_uplift_best.joblib", {
    "model_name": best["model"],
    "model": best["estimator"],
    "features": FEATURES,
    "metrics": {k: best[k] for k in ("qini_auc", "uplift_at_30pct")},
})
""", "save"),
])

# ── 05 NBA ───────────────────────────────────────────────────────────────────
write("05-next-best-action.ipynb", [
    md("""# 05 · Next Best Action

**Data:** `uci_nba_offer_events.parquet` + `uci_customer_features.parquet`  
**(A) Supervised:** multi-class classifier → best `offer_id`  
**(B) RL agent:** tabular Q-learning (state = RFM segment × tier, action = offer type)

Split: 60/20/20 stratified on `converted`.
""", "intro"),
    code(SETUP, "setup"),
    code("""from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

events = load_parquet("uci_nba_offer_events.parquet")
feat = load_parquet("uci_customer_features.parquet")
catalog = load_parquet("uci_nba_offer_catalog.parquet")

df = events.merge(feat, on="customer_id", how="left", suffixes=("", "_feat"))
df = df[df["shown"] == 1].copy()  # only shown offers

FEATURES = [
    "RFM_score", "engagement_score", "discount_sensitivity",
    "recency_days", "discount_pct", "clicked",
]
cat_features = ["rfm_segment", "tier", "channel", "offer_type"]
df = audit_and_clean(
    df,
    subset=["customer_id", "offer_id", "event_date", "channel"],
    required_cols=["customer_id", "offer_id", "converted", *FEATURES, *cat_features],
    label="nba_offer_events",
)
print(f"Shown offers: {len(df):,} · Conversion rate: {df['converted'].mean():.1%}")
""", "load"),
    code("""num = df[FEATURES + ["converted"]].corr()
sns.heatmap(num, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("NBA numeric feature correlation")
plt.tight_layout()
plt.show()
""", "corr"),
    code("""# (A) Supervised multi-class: predict offer_id for converted=1 subset + weighted
le = LabelEncoder()
df["offer_id_enc"] = le.fit_transform(df["offer_id"])

X_num = df[FEATURES]
X_cat = pd.get_dummies(df[cat_features], drop_first=False)
X_all = pd.concat([X_num, X_cat], axis=1)
y_cls = df["offer_id_enc"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_cls, test_size=0.40, stratify=y_cls, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

candidates = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        objective="multi:softprob", num_class=len(le.classes_),
        random_state=RANDOM_STATE, verbosity=0,
    ),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
}

sup_results = []
for name, clf in candidates.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    sup_results.append({
        "model": name,
        "test_accuracy": accuracy_score(y_test, pred),
        "test_f1_macro": f1_score(y_test, pred, average="macro"),
        "estimator": clf,
    })

sup_lb = pd.DataFrame([{k: v for k, v in r.items() if k != "estimator"} for r in sup_results]).set_index("model")
display(sup_lb.round(4))
best_sup = max(sup_results, key=lambda r: r["test_f1_macro"])
print(f"Best supervised: {best_sup['model']}")
""", "supervised"),
    code("""# (B) Tabular Q-learning for offer_type selection
ACTIONS = sorted(df["offer_type"].unique())
STATES = sorted(df["rfm_segment"].astype(str) + "|" + df["tier"].astype(str).unique()) if False else None

state_keys = (df["rfm_segment"].astype(str) + "|" + df["tier"].astype(str)).unique()
actions = sorted(df["offer_type"].unique())
Q = {s: {a: 0.0 for a in actions} for s in state_keys}

alpha, gamma, epsilon = 0.1, 0.9, 0.15
rng = np.random.default_rng(RANDOM_STATE)

def reward(row):
    profit = 1.0 if row["converted"] else 0.0
    cost = row["discount_pct"] * 0.5 + (0.1 if row["channel"] == "email" else 0.05)
    retention = 0.3 if row["rfm_segment"] in ("At Risk", "Hibernating") and row["converted"] else 0.0
    return profit + retention - cost

train_df = df.sample(frac=0.7, random_state=RANDOM_STATE)
for _, row in train_df.iterrows():
    s = f"{row['rfm_segment']}|{row['tier']}"
    a = row["offer_type"]
    r = reward(row)
    if rng.random() < epsilon:
        a_explore = rng.choice(actions)
        Q[s][a_explore] = Q[s][a_explore] + alpha * (r - Q[s][a_explore])
    else:
        Q[s][a] = Q[s][a] + alpha * (r - Q[s][a])

def rl_recommend(state):
    return max(Q.get(state, {a: 0 for a in actions}), key=Q.get(state, {}).keys())

# Evaluate RL on holdout
test_df = df.drop(train_df.index)
hits = sum(rl_recommend(f"{r['rfm_segment']}|{r['tier']}") == r["offer_type"] for _, r in test_df.iterrows())
print(f"RL policy match rate on holdout: {hits/len(test_df):.1%}")
print(f"Sample Q-table size: {len(Q)} states × {len(actions)} actions")
""", "rl"),
    code("""save_artifact("05_nba_supervised_best.joblib", {
    "model_name": best_sup["model"],
    "model": best_sup["estimator"],
    "label_encoder": le,
    "feature_columns": list(X_all.columns),
    "metrics": {k: best_sup[k] for k in ("test_accuracy", "test_f1_macro")},
})
save_artifact("05_nba_rl_qtable.joblib", {
    "Q": Q,
    "actions": actions,
    "alpha": alpha,
    "gamma": gamma,
})
""", "save"),
])

# Fix NBA RL states bug and supervised xgb num_class
cells_nba = json.loads((NB / "05-next-best-action.ipynb").read_text())["cells"]
cells_nba[-2]["source"] = '''# (B) Tabular Q-learning for offer_type selection
actions = sorted(df["offer_type"].unique())
state_keys = (df["rfm_segment"].astype(str) + "|" + df["tier"].astype(str)).unique()
Q = {s: {a: 0.0 for a in actions} for s in state_keys}

alpha, gamma, epsilon = 0.1, 0.9, 0.15
rng = np.random.default_rng(RANDOM_STATE)

def reward(row):
    profit = 1.0 if row["converted"] else 0.0
    cost = row["discount_pct"] * 0.5 + (0.1 if row["channel"] == "email" else 0.05)
    retention = 0.3 if row["rfm_segment"] in ("At Risk", "Hibernating") and row["converted"] else 0.0
    return profit + retention - cost

train_df = df.sample(frac=0.7, random_state=RANDOM_STATE)
for _, row in train_df.iterrows():
    s = f"{row['rfm_segment']}|{row['tier']}"
    a = row["offer_type"]
    r = reward(row)
    if rng.random() < epsilon:
        a = rng.choice(actions)
    Q[s][a] = Q[s][a] + alpha * (r - Q[s][a])

def rl_recommend(state):
    if state not in Q:
        return actions[0]
    return max(Q[state], key=Q[state].get)

test_df = df.drop(train_df.index)
hits = sum(rl_recommend(f"{r['rfm_segment']}|{r['tier']}") == r["offer_type"] for _, r in test_df.iterrows())
print(f"RL policy match rate on holdout: {hits/len(test_df):.1%}")
'''.splitlines(keepends=True)
(NB / "05-next-best-action.ipynb").write_text(json.dumps(nb(cells_nba), indent=1), encoding="utf-8")

print("Done.")
