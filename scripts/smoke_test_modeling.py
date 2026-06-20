"""Smoke-test all 01-05 modeling notebooks end-to-end (no plots)."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "modeling"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def ok(name: str) -> None:
    print(f"  OK  {name}")


def run_01() -> None:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feat = pd.read_parquet(DATA / "uci_customer_features.parquet")
    cols = ["R_score", "F_score", "M_score", "RFM_score", "avg_basket_size",
            "discount_dependency", "engagement_score", "churn_inertia_score",
            "recency_days", "total_orders"]
    X = feat[cols].fillna(0)
    corr = X.corr()
    drop = [c for c in corr.columns if any(corr[c].abs() > 0.85) and c != corr.columns[0]]
    selected = [c for c in cols if c not in drop]
    X = X[selected]
    X_train, _ = train_test_split(X, test_size=0.30, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    km = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20).fit(Xs)
    assert silhouette_score(Xs, km.labels_) > -1
    joblib.dump({"model": km, "scaler": scaler, "features": selected}, MODELS / "01_segmentation_best.joblib")
    ok("01-segmentation")


def run_02() -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    import xgboost as xgb

    df = pd.read_parquet(DATA / "telco_customers.parquet").dropna(subset=["TotalCharges"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    y = df["churn_flag"]
    X = df.drop(columns=[c for c in ["telco_customer_id", "Churn", "churn_flag"] if c in df.columns])
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    pipe = Pipeline([("prep", prep), ("clf", xgb.XGBClassifier(
        n_estimators=50, max_depth=4, random_state=RANDOM_STATE, verbosity=0))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    assert auc > 0.5
    joblib.dump({"pipeline": pipe}, MODELS / "02_churn_best.joblib")
    ok(f"02-churn (AUC={auc:.3f})")


def run_03() -> None:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    from sklearn.metrics import mean_absolute_error

    txn = pd.read_parquet(DATA / "uci_fact_transactions.parquet")
    txn["order_date"] = pd.to_datetime(txn["order_date"])
    hold_end = txn["order_date"].max()
    summary = summary_data_from_transaction_data(
        txn, "customer_id", "order_date", monetary_value_col="line_total",
        observation_period_end=hold_end,
    )
    bgf = BetaGeoFitter(penalizer_coef=0.1).fit(summary["frequency"], summary["recency"], summary["T"])
    gg = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.01).fit(gg["frequency"], gg["monetary_value"])
    clv = ggf.customer_lifetime_value(
        bgf, summary["frequency"], summary["recency"], summary["T"],
        summary["monetary_value"], time=3.0,
    )
    assert clv.notna().sum() > 0
    bgf.save_model(str(MODELS / "03_clv_bgf"))
    ggf.save_model(str(MODELS / "03_clv_ggf"))
    joblib.dump({"bgf_path": "03_clv_bgf", "ggf_path": "03_clv_ggf"}, MODELS / "03_clv_best.joblib")
    ok("03-clv")


def run_04() -> None:
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    df = pd.read_parquet(DATA / "uci_uplift_campaigns.parquet")
    feats = ["discount_sensitivity", "engagement_score", "avg_order_value"]
    X = df[feats].fillna(0).values
    y = df["responded"].astype(int).values
    w = df["treatment"].astype(int).values
    cust = df["customer_id"].astype(str).unique().to_numpy()
    c_tr, c_te = train_test_split(cust, test_size=0.3, random_state=RANDOM_STATE)
    tr = df["customer_id"].astype(str).isin(c_tr).values
    m1 = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE, verbosity=0)
    m0 = clone(m1)
    m1.fit(X[tr & (w == 1)], y[tr & (w == 1)])
    m0.fit(X[tr & (w == 0)], y[tr & (w == 0)])
    uplift = m1.predict_proba(X[~tr])[:, 1] - m0.predict_proba(X[~tr])[:, 1]
    assert len(uplift) > 0
    joblib.dump({"m1": m1, "m0": m0, "features": feats}, MODELS / "04_uplift_best.joblib")
    ok("04-uplift")


def run_05() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    events = pd.read_parquet(DATA / "uci_nba_offer_events.parquet")
    feat = pd.read_parquet(DATA / "uci_customer_features.parquet")
    df = events.merge(feat, on="customer_id", how="left", suffixes=("", "_feat"))
    df = df[df["shown"] == 1]
    feats = ["RFM_score", "engagement_score", "discount_sensitivity", "recency_days", "discount_pct", "clicked"]
    le = LabelEncoder()
    y = le.fit_transform(df["offer_id"])
    X = df[feats].fillna(0)
    X = pd.concat([X, pd.get_dummies(df[["rfm_segment", "tier", "channel", "offer_type"]].fillna("unknown"))], axis=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    clf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    f1 = f1_score(y_te, clf.predict(X_te), average="macro")
    joblib.dump({"model": clf, "label_encoder": le}, MODELS / "05_nba_supervised_best.joblib")
    ok(f"05-nba (F1={f1:.3f})")


def main() -> int:
    tests = [run_01, run_02, run_03, run_04, run_05]
    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed.append(fn.__name__)
    if failed:
        print(f"\nFailed: {failed}")
        return 1
    print(f"\nAll {len(tests)} modeling smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
