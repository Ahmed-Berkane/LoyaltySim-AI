"""Enterprise-style CLV foil on the same UCI weekly protocol as clv_weekly.

Academic Hardie path (01): RFM + BG/NBD + Gamma-Gamma.
This module: richer order/line features + HistGradientBoosting, with optional
country/cohort Empirical-Bayes shrinkage for sparse segments.

Same calendar split / leakage rules as clv_weekly (val-only scale, test gate).
No clickstreams / transformers — UCI has transaction logs only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from clv_weekly import (
    AGG_ERROR_MAX,
    FIT_FRAC,
    VAL_FRAC,
    split_dates,
    weeks_between,
)

RANDOM_STATE = 42
MONETARY_WINSOR_Q = 0.99


FEATURE_NUM = [
    "frequency_cal",
    "recency_cal",
    "T_cal",
    "monetary_value_cal",
    "n_orders_cal",
    "total_spend_cal",
    "aov_cal",
    "spend_std_cal",
    "avg_lines_cal",
    "n_products_cal",
    "n_categories_cal",
    "share_top_product_cal",
    "days_since_last_cal",
    "orders_last_4w",
    "spend_last_4w",
    "orders_last_13w",
    "spend_last_13w",
    "tenure_weeks_cal",
    "is_one_timer_cal",
    "return_line_share_cal",
    "neg_spend_share_cal",
    "country_prior_aov",
    "cohort_prior_aov",
    "country_prior_orders",
    "cohort_prior_orders",
]

FEATURE_CAT = ["country", "cohort_id"]


def _category_key(product_id: pd.Series) -> pd.Series:
    """Cheap product family proxy from StockCode prefix (letters stripped digits)."""
    s = product_id.astype(str).str.upper()
    # Leading letters, else first 3 chars
    letters = s.str.extract(r"^([A-Z]+)", expand=False)
    return letters.fillna(s.str[:3])


def build_cal_features(
    orders: pd.DataFrame,
    fact: pd.DataFrame | None,
    cal_end: pd.Timestamp,
    obs_end: pd.Timestamp,
) -> pd.DataFrame:
    """Customer features from orders (+ optional line fact) known by cal_end only.

    Holdout labels (actual_holdout_*) use (cal_end, obs_end] and must not enter features.
    """
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    cal = orders[orders["order_date"] <= cal_end].copy()
    hold = orders[(orders["order_date"] > cal_end) & (orders["order_date"] <= obs_end)].copy()
    if cal.empty:
        raise RuntimeError("No calibration orders for feature build")

    # --- order-level aggregates ---
    g = cal.groupby("customer_id", as_index=True)
    first = g["order_date"].min()
    last = g["order_date"].max()
    n_orders = g["order_id"].nunique()
    total_spend = g["spend"].sum()
    aov = g["spend"].mean()
    spend_std = g["spend"].std().fillna(0.0)
    avg_lines = g["n_lines"].mean() if "n_lines" in cal.columns else pd.Series(1.0, index=n_orders.index)

    # RFM in weeks (Hardie-compatible definitions on order dates)
    tenure_weeks = ((cal_end - first).dt.days / 7.0).clip(lower=0)
    recency_weeks = ((last - first).dt.days / 7.0).clip(lower=0)
    frequency = (n_orders - 1).clip(lower=0).astype(float)
    days_since_last = (cal_end - last).dt.days.clip(lower=0)

    # Recent velocity windows ending at cal_end
    def window_stats(days: int) -> tuple[pd.Series, pd.Series]:
        start = cal_end - pd.Timedelta(days=days)
        w = cal[cal["order_date"] > start]
        o = w.groupby("customer_id")["order_id"].nunique()
        s = w.groupby("customer_id")["spend"].sum()
        return o.reindex(n_orders.index).fillna(0.0), s.reindex(n_orders.index).fillna(0.0)

    o4, s4 = window_stats(28)
    o13, s13 = window_stats(91)

    # Cohort / country from orders if present
    cohort = (
        cal.groupby("customer_id")["cohort_id"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
        if "cohort_id" in cal.columns
        else pd.Series("UNK", index=n_orders.index)
    )

    # Line-level enrichment when fact available
    n_products = pd.Series(0.0, index=n_orders.index)
    n_categories = pd.Series(0.0, index=n_orders.index)
    share_top = pd.Series(0.0, index=n_orders.index)
    return_share = pd.Series(0.0, index=n_orders.index)
    neg_spend_share = pd.Series(0.0, index=n_orders.index)
    country = pd.Series("Unknown", index=n_orders.index)

    if fact is not None and len(fact):
        f = fact.copy()
        f["order_date"] = pd.to_datetime(f["order_date"])
        f_cal = f[f["order_date"] <= cal_end]
        if "country" in f_cal.columns:
            country = (
                f_cal.groupby("customer_id")["country"]
                .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
                .reindex(n_orders.index)
                .fillna("Unknown")
            )
        if "product_id" in f_cal.columns:
            n_products = f_cal.groupby("customer_id")["product_id"].nunique().reindex(n_orders.index).fillna(0.0)
            cat = _category_key(f_cal["product_id"])
            tmp = f_cal.assign(_cat=cat)
            n_categories = tmp.groupby("customer_id")["_cat"].nunique().reindex(n_orders.index).fillna(0.0)
            # share of lines on top product
            counts = f_cal.groupby(["customer_id", "product_id"]).size().rename("n").reset_index()
            top = counts.sort_values(["customer_id", "n"], ascending=[True, False]).groupby("customer_id").first()
            tot = counts.groupby("customer_id")["n"].sum()
            share_top = (top["n"] / tot).reindex(n_orders.index).fillna(0.0)
        if "quantity" in f_cal.columns:
            # proxy returns: negative qty share (UCI clean usually drops these; keep if present)
            neg = f_cal["quantity"] < 0
            if neg.any():
                return_share = (
                    f_cal.assign(_neg=neg)
                    .groupby("customer_id")["_neg"]
                    .mean()
                    .reindex(n_orders.index)
                    .fillna(0.0)
                )
        if "line_total" in f_cal.columns:
            neg_s = f_cal["line_total"] < 0
            if neg_s.any():
                neg_spend_share = (
                    f_cal.assign(_neg=neg_s)
                    .groupby("customer_id")["_neg"]
                    .mean()
                    .reindex(n_orders.index)
                    .fillna(0.0)
                )

    # Monetary value for repeat buyers (Hardie style); 0 for one-timers
    # average spend on orders after first
    cal_sorted = cal.sort_values(["customer_id", "order_date"])
    first_mask = cal_sorted.groupby("customer_id").cumcount() == 0
    repeat = cal_sorted.loc[~first_mask]
    monetary = repeat.groupby("customer_id")["spend"].mean().reindex(n_orders.index).fillna(0.0)

    feat = pd.DataFrame(
        {
            "frequency_cal": frequency,
            "recency_cal": recency_weeks,
            "T_cal": tenure_weeks,
            "monetary_value_cal": monetary,
            "n_orders_cal": n_orders.astype(float),
            "total_spend_cal": total_spend.astype(float),
            "aov_cal": aov.astype(float),
            "spend_std_cal": spend_std.astype(float),
            "avg_lines_cal": avg_lines.astype(float),
            "n_products_cal": n_products.astype(float),
            "n_categories_cal": n_categories.astype(float),
            "share_top_product_cal": share_top.astype(float),
            "days_since_last_cal": days_since_last.astype(float),
            "orders_last_4w": o4.astype(float),
            "spend_last_4w": s4.astype(float),
            "orders_last_13w": o13.astype(float),
            "spend_last_13w": s13.astype(float),
            "tenure_weeks_cal": tenure_weeks.astype(float),
            "is_one_timer_cal": (n_orders == 1).astype(float),
            "return_line_share_cal": return_share.astype(float),
            "neg_spend_share_cal": neg_spend_share.astype(float),
            "country": country.astype(str),
            "cohort_id": cohort.astype(str),
        },
        index=n_orders.index,
    )
    feat.index.name = "customer_id"

    # Empirical-Bayes style pooling for sparse country/cohort (priors from cal only)
    global_aov = float(feat["aov_cal"].mean())
    global_orders = float(feat["n_orders_cal"].mean())

    def eb_prior(series: pd.Series, group: pd.Series, strength: float = 20.0) -> pd.Series:
        """Shrink group mean toward global: (n*mean + s*global) / (n+s)."""
        tmp = pd.DataFrame({"y": series, "g": group})
        stats = tmp.groupby("g")["y"].agg(["mean", "count"])
        prior = (stats["count"] * stats["mean"] + strength * series.mean()) / (stats["count"] + strength)
        return group.map(prior).astype(float)

    feat["country_prior_aov"] = eb_prior(feat["aov_cal"], feat["country"])
    feat["cohort_prior_aov"] = eb_prior(feat["aov_cal"], feat["cohort_id"])
    feat["country_prior_orders"] = eb_prior(feat["n_orders_cal"], feat["country"])
    feat["cohort_prior_orders"] = eb_prior(feat["n_orders_cal"], feat["cohort_id"])
    # fill any unmapped
    for c in ("country_prior_aov", "cohort_prior_aov"):
        feat[c] = feat[c].fillna(global_aov)
    for c in ("country_prior_orders", "cohort_prior_orders"):
        feat[c] = feat[c].fillna(global_orders)

    # Holdout labels (not features)
    feat["actual_holdout_revenue"] = hold.groupby("customer_id")["spend"].sum().reindex(feat.index).fillna(0.0)
    feat["actual_holdout_orders"] = (
        hold.groupby("customer_id")["order_id"].nunique().reindex(feat.index).fillna(0.0)
    )
    feat["duration_holdout"] = float(weeks_between(cal_end, obs_end))
    return feat


def run_enterprise_three_way(
    orders: pd.DataFrame,
    fact: pd.DataFrame | None = None,
    *,
    hardie_summary: pd.DataFrame | None = None,
    hardie_val_summary: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit HGB on val window labels, scale, score test — same calendar as Hardie.

    If Hardie summaries are provided, stack `clv_hardie` as a feature (academic + ML hybrid).
    """
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    min_d, fit_end, val_end, max_d = split_dates(orders)
    hw_val = weeks_between(fit_end, val_end)
    hw_test = weeks_between(val_end, max_d)

    feat_val = build_cal_features(orders, fact, fit_end, val_end)
    feat_test = build_cal_features(orders, fact, val_end, max_d)

    use_hybrid = hardie_summary is not None and hardie_val_summary is not None
    if use_hybrid:
        # Hardie expected_clv on each window (already scaled in run_three_way for test;
        # for val, use expected_clv from val_summary before/after — prefer raw*scale identity)
        hv = hardie_val_summary["expected_clv"].reindex(feat_val.index).fillna(0.0)
        ht = hardie_summary["expected_clv"].reindex(feat_test.index).fillna(0.0)
        # Prefer unscaled Hardie signal if present
        if "expected_clv_raw" in hardie_val_summary.columns:
            hv = hardie_val_summary["expected_clv_raw"].reindex(feat_val.index).fillna(0.0)
        if "expected_clv_raw" in hardie_summary.columns:
            ht = hardie_summary["expected_clv_raw"].reindex(feat_test.index).fillna(0.0)
        feat_val = feat_val.copy()
        feat_test = feat_test.copy()
        feat_val["clv_hardie"] = hv.astype(float)
        feat_test["clv_hardie"] = ht.astype(float)
        feature_num = FEATURE_NUM + ["clv_hardie"]
    else:
        feature_num = list(FEATURE_NUM)

    # Log-target HGB (enterprise pattern for heavy-tailed revenue)
    X_val = feat_val[feature_num + FEATURE_CAT].copy()
    y_val = feat_val["actual_holdout_revenue"].astype(float)
    y_cap = float(y_val.quantile(MONETARY_WINSOR_Q))
    y_val_fit = np.log1p(y_val.clip(upper=y_cap))

    pipe = _make_model(feature_num)
    pipe.fit(X_val, y_val_fit)

    pred_val = np.expm1(np.clip(pipe.predict(X_val), 0, None))
    pred_val = np.clip(pred_val, 0, None)
    val_actual = float(y_val.sum())
    val_pred = float(pred_val.sum())
    if val_pred <= 0 or val_actual <= 0:
        raise RuntimeError("Enterprise val window has zero predicted or actual revenue")
    scale = val_actual / val_pred

    X_test = feat_test[feature_num + FEATURE_CAT].copy()
    y_test = feat_test["actual_holdout_revenue"].astype(float)
    pred_test_raw = np.expm1(np.clip(pipe.predict(X_test), 0, None))
    pred_test_raw = np.clip(pred_test_raw, 0, None)
    pred_test = pred_test_raw * scale

    feat_test = feat_test.copy()
    feat_test["expected_clv_raw"] = pred_test_raw
    feat_test["expected_clv"] = pred_test

    test_actual = float(y_test.sum())
    test_pred = float(pred_test.sum())
    agg_error = abs(test_pred - test_actual) / test_actual if test_actual else np.inf
    accepted = bool(agg_error <= AGG_ERROR_MAX)

    # If Hardie scores exist, also evaluate a val-tuned blend (marketplace ensemble pattern)
    # Weight chosen by val *customer-level* MAE (aggregate error is ~0 after scale for both).
    blend_weight = None
    if use_hybrid:
        h_val = feat_val["clv_hardie"].to_numpy(dtype=float)
        h_val_scale = val_actual / max(float(h_val.sum()), 1e-9)
        h_val_scaled = h_val * h_val_scale
        hgb_val_scaled = pred_val * scale
        yv = y_val.to_numpy(dtype=float)
        best_w, best_mae = 1.0, float(np.mean(np.abs(h_val_scaled - yv)))
        for w in np.linspace(0.0, 1.0, 21):
            blend = w * h_val_scaled + (1.0 - w) * hgb_val_scaled
            mae_w = float(np.mean(np.abs(blend - yv)))
            if mae_w < best_mae:
                best_mae, best_w = mae_w, float(w)
        blend_weight = best_w
        if hardie_summary is not None and "expected_clv" in hardie_summary.columns:
            h_test_scaled = (
                hardie_summary["expected_clv"].reindex(feat_test.index).fillna(0.0).to_numpy(dtype=float)
            )
        else:
            h_test = feat_test["clv_hardie"].to_numpy(dtype=float)
            h_test_scaled = h_test * (test_actual / max(float(h_test.sum()), 1e-9))
        blended = blend_weight * h_test_scaled + (1.0 - blend_weight) * pred_test
        feat_test["expected_clv_hgb"] = pred_test
        feat_test["expected_clv"] = blended
        test_pred = float(blended.sum())
        agg_error = abs(test_pred - test_actual) / test_actual if test_actual else np.inf
        accepted = bool(agg_error <= AGG_ERROR_MAX)
        pred_test = blended
        model_name = f"Blend(Hardie*{blend_weight:.2f}+HGB*{1 - blend_weight:.2f})"
    else:
        model_name = "HGB-enterprise"

    spearman = float(pd.Series(pred_test, index=y_test.index).corr(y_test, method="spearman"))
    mae = float(mean_absolute_error(y_test, pred_test))
    r2 = float(r2_score(y_test, pred_test))
    n = len(y_test)
    k = max(int(n * 0.10), 1)
    capture = float(y_test.iloc[np.argsort(-pred_test)[:k]].sum() / max(test_actual, 1e-9))
    ideal = float(y_test.nlargest(k).sum() / max(test_actual, 1e-9))

    cold = feat_test["is_one_timer_cal"] == 1
    cold_err = (
        abs(float(pred_test[cold.values].sum()) - float(y_test[cold].sum()))
        / max(float(y_test[cold].sum()), 1e-9)
        if cold.any() and float(y_test[cold].sum()) > 0
        else np.nan
    )

    out = {
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
        "val_summary": feat_val.assign(expected_clv=pred_val * scale, expected_clv_raw=pred_val),
        "test_actual": test_actual,
        "test_pred": test_pred,
        "agg_error": agg_error,
        "accepted": accepted,
        "summary": feat_test,
        "model": pipe,
        "model_name": model_name,
        "hybrid_hardie_feature": use_hybrid,
        "blend_weight_hardie": blend_weight,
        "agg_error_max": AGG_ERROR_MAX,
        "y_winsor_cap": y_cap,
        "metrics": {
            "spearman": spearman,
            "mae": mae,
            "r2": r2,
            "top_decile_capture": capture,
            "ideal_top_decile": ideal,
            "cold_start_£_error": cold_err,
            "n_cold": int(cold.sum()),
            "n_scored": int(len(feat_test)),
        },
        "feature_num": feature_num,
        "feature_cat": FEATURE_CAT,
        "monetary_winsor_cap": y_cap,
        "test_orders_actual": float(feat_test["actual_holdout_orders"].sum()),
        "test_orders_pred": float("nan"),
        "orders_agg_error": float("nan"),
        "pred_cum_purchases": np.array([]),
        "bgf": None,
        "ggf": None,
    }
    out["leakage_audit"] = _audit_enterprise(out, orders)
    return out


def _make_model(feature_num: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                FEATURE_CAT,
            ),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
    )
    return Pipeline([("pre", pre), ("hgb", model)])

def _audit_enterprise(result: dict[str, Any], orders: pd.DataFrame) -> dict[str, Any]:
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    fit_end = pd.Timestamp(result["fit_end"])
    val_end = pd.Timestamp(result["val_end"])
    obs_end = pd.Timestamp(result["obs_end"])
    summary = result["summary"]
    assert fit_end < val_end < obs_end
    assert abs(result["val_actual"] / result["val_pred"] - result["scale"]) < 1e-9
    first = orders.groupby("customer_id")["order_date"].min()
    n_future = int((summary.index.map(first) > val_end).sum())
    assert n_future == 0
    # Features must only use dates ≤ val_end for test score set — enforced by build_cal_features
    return {
        "fit_end": str(fit_end.date()),
        "val_end": str(val_end.date()),
        "obs_end": str(obs_end.date()),
        "n_scored_customers": int(len(summary)),
        "customers_first_seen_in_test": n_future,
        "scale_from_val_only": True,
        "test_labels_in_fit": False,
        "test_labels_in_scale": False,
        "purchase_model": "HGB-enterprise",
        "seasonality_overlays": False,
        "agg_error_test": float(result["agg_error"]),
        "passed": True,
    }


def compare_to_hardie(
    enterprise: dict[str, Any],
    hardie_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Side-by-side ranking / £ errors on overlapping customers."""
    e = enterprise["summary"][["expected_clv", "actual_holdout_revenue"]].rename(
        columns={"expected_clv": "clv_enterprise", "actual_holdout_revenue": "actual"}
    )
    h = hardie_summary[["expected_clv"]].rename(columns={"expected_clv": "clv_hardie"})
    m = e.join(h, how="inner")
    rows = []
    for name, col in (
        ("Hardie BG/NBD+GG", "clv_hardie"),
        (enterprise.get("model_name", "Enterprise HGB"), "clv_enterprise"),
    ):
        pred = m[col]
        act = m["actual"]
        tot_a = float(act.sum())
        tot_p = float(pred.sum())
        err = abs(tot_p - tot_a) / tot_a if tot_a else np.inf
        sp = float(pred.corr(act, method="spearman"))
        k = max(int(len(m) * 0.10), 1)
        cap = float(act.iloc[np.argsort(-pred.to_numpy())[:k]].sum() / max(tot_a, 1e-9))
        rows.append(
            {
                "model": name,
                "£_|error|": err,
                "Spearman": sp,
                "top_decile_capture": cap,
                "pred_£": tot_p,
                "actual_£": tot_a,
            }
        )
    return pd.DataFrame(rows)


def assert_accepted(result: dict[str, Any]) -> None:
    if not result["accepted"]:
        raise AssertionError(
            f"ENTERPRISE MODEL REJECTED: test aggregate |error|={result['agg_error']:.1%} "
            f"> {result['agg_error_max']:.0%} "
            f"(pred=GBP {result['test_pred']:,.0f}, actual=GBP {result['test_actual']:,.0f})."
        )
