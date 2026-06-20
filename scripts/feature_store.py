"""Customer-level feature store for LoyaltySim Superstore spine."""

from __future__ import annotations

import numpy as np
import pandas as pd

TIER_ORDINAL = {"Bronze": 1, "Silver": 2, "Gold": 3, "Platinum": 4}
INACTIVITY_CHURN_DAYS = 180

FEATURE_GROUPS = {
    "Core": [
        "R_score", "F_score", "M_score", "RFM_score",
        "avg_basket_size", "frequency_variance", "discount_dependency",
    ],
    "Behavioral": ["engagement_score", "email_response_rate", "app_activity_velocity"],
    "Temporal": ["seasonality_index", "payday_effect", "holiday_sensitivity"],
    "Loyalty": ["tier_progression_speed", "reward_redemption_rate", "churn_inertia_score"],
}


def _quintile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    ranked = series.rank(method="first", ascending=not higher_is_better)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)


def assign_rfm_segment(r: float, f: float, m: float) -> str:
    """Standard 5-bucket RFM segment labels."""
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "Promising"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r <= 2 and f <= 2:
        return "Hibernating"
    return "Needs Attention"


def build_customer_features(fact: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """Engineer modeling features + RFM segment + inactivity churn label."""
    as_of = fact["order_date"].max()

    order_meta = (
        fact.groupby("order_id", as_index=False)
        .agg(
            customer_id=("customer_id", "first"),
            order_date=("order_date", "first"),
            order_sales=("sales", "sum"),
            line_count=("transaction_id", "count"),
            is_holiday=("is_us_federal_holiday", "max"),
            is_month_start=("is_month_start", "max"),
            is_month_end=("is_month_end", "max"),
        )
        .assign(
            is_payday_window=lambda d: d["is_month_start"] | d["is_month_end"],
            quarter=lambda d: d["order_date"].dt.quarter,
        )
    )
    order_meta = order_meta.sort_values(["customer_id", "order_date"])
    order_meta["gap_days"] = order_meta.groupby("customer_id")["order_date"].diff().dt.days

    core = order_meta.groupby("customer_id").agg(
        avg_basket_size=("line_count", "mean"),
        avg_basket_value=("order_sales", "mean"),
        frequency_variance=("gap_days", "std"),
        avg_interorder_days=("gap_days", "mean"),
    )
    disc = fact.groupby("customer_id").agg(
        pct_discounted_lines=("discount", lambda s: (s > 0).mean()),
        avg_line_discount=("discount", "mean"),
    )
    disc["discount_dependency"] = 0.6 * disc["avg_line_discount"] + 0.4 * disc["pct_discounted_lines"]

    feat = dim[
        [
            "customer_id", "customer_name", "segment", "tier", "home_region",
            "total_orders", "total_sales", "total_profit", "avg_order_value",
            "first_order_date", "last_order_date", "avg_discount",
            "points_balance", "email_opt_in", "app_usage_score", "discount_sensitivity",
        ]
    ].copy()
    feat["recency_days"] = (as_of - pd.to_datetime(feat["last_order_date"])).dt.days
    feat["tenure_days"] = (
        pd.to_datetime(feat["last_order_date"]) - pd.to_datetime(feat["first_order_date"])
    ).dt.days

    feat = feat.merge(core, on="customer_id").merge(
        disc[["discount_dependency", "pct_discounted_lines"]], on="customer_id"
    )

    feat["R_score"] = _quintile_score(feat["recency_days"], higher_is_better=False)
    feat["F_score"] = _quintile_score(feat["total_orders"], higher_is_better=True)
    feat["M_score"] = _quintile_score(feat["total_sales"], higher_is_better=True)
    feat["RFM_score"] = feat["R_score"] + feat["F_score"] + feat["M_score"]
    feat["rfm_segment"] = [
        assign_rfm_segment(r, f, m)
        for r, f, m in zip(feat["R_score"], feat["F_score"], feat["M_score"])
    ]

    freq_norm = feat["total_orders"] / feat["total_orders"].max()
    feat["engagement_score"] = (
        0.45 * feat["app_usage_score"]
        + 0.25 * feat["email_opt_in"].astype(float)
        + 0.30 * freq_norm
    )
    feat["email_response_rate"] = feat["email_opt_in"].astype(float) * (
        0.35 + 0.65 * feat["app_usage_score"]
    )
    feat["app_activity_velocity"] = feat["app_usage_score"] / np.maximum(
        feat["tenure_days"] / 365, 0.25
    )

    temp = order_meta.groupby("customer_id").agg(
        q4_order_share=("quarter", lambda s: (s == 4).mean()),
        holiday_order_share=("is_holiday", "mean"),
        payday_window_share=("is_payday_window", "mean"),
    )
    global_holiday = order_meta["is_holiday"].mean()
    temp["seasonality_index"] = temp["q4_order_share"]
    temp["payday_effect"] = temp["payday_window_share"]
    temp["holiday_sensitivity"] = temp["holiday_order_share"] / max(global_holiday, 1e-6)
    feat = feat.merge(temp, on="customer_id")

    feat["tier_ordinal"] = feat["tier"].map(TIER_ORDINAL)
    feat["tier_progression_speed"] = feat["tier_ordinal"] / np.log1p(feat["tenure_days"] / 30)
    points_earned_proxy = feat["total_sales"] * 2
    feat["reward_redemption_rate"] = (
        1 - feat["points_balance"] / np.maximum(points_earned_proxy, 1)
    ).clip(0, 1)
    feat["churn_inertia_score"] = feat["recency_days"] / np.maximum(
        feat["avg_interorder_days"].fillna(feat["recency_days"]), 1
    )
    feat["inactivity_churn_label"] = (feat["recency_days"] >= INACTIVITY_CHURN_DAYS).astype(int)
    feat["feature_as_of_date"] = as_of
    return feat


def build_uci_customer_features(fact: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """Engineer modeling features for UCI Online Retail II spine."""
    as_of = fact["order_date"].max()
    holiday_col = "is_public_holiday" if "is_public_holiday" in fact.columns else "is_us_federal_holiday"

    order_meta = (
        fact.groupby("order_id", as_index=False)
        .agg(
            customer_id=("customer_id", "first"),
            order_date=("order_date", "first"),
            order_sales=("line_total", "sum"),
            line_count=("order_id", "count"),
            is_holiday=(holiday_col, "max"),
            is_month_start=("is_month_start", "max"),
            is_month_end=("is_month_end", "max"),
        )
        .assign(
            is_payday_window=lambda d: d["is_month_start"] | d["is_month_end"],
            quarter=lambda d: d["order_date"].dt.quarter,
        )
    )
    order_meta = order_meta.sort_values(["customer_id", "order_date"])
    order_meta["gap_days"] = order_meta.groupby("customer_id")["order_date"].diff().dt.days

    core = order_meta.groupby("customer_id").agg(
        avg_basket_size=("line_count", "mean"),
        avg_basket_value=("order_sales", "mean"),
        frequency_variance=("gap_days", "std"),
        avg_interorder_days=("gap_days", "mean"),
    )
    disc = fact.groupby("customer_id").agg(
        avg_unit_price=("unit_price", "mean"),
        price_std=("unit_price", "std"),
    ).reset_index()
    disc = disc.merge(dim[["customer_id", "discount_sensitivity"]], on="customer_id")
    disc["discount_dependency"] = disc["discount_sensitivity"]

    feat = dim[
        [
            "customer_id", "country", "tier",
            "total_orders", "total_revenue", "avg_order_value",
            "first_order_date", "last_order_date",
            "points_balance", "email_opt_in", "app_usage_score", "discount_sensitivity",
        ]
    ].copy()
    feat["recency_days"] = (as_of - pd.to_datetime(feat["last_order_date"])).dt.days
    feat["tenure_days"] = (
        pd.to_datetime(feat["last_order_date"]) - pd.to_datetime(feat["first_order_date"])
    ).dt.days

    feat = feat.merge(core, on="customer_id").merge(
        disc[["customer_id", "discount_dependency"]], on="customer_id"
    )

    feat["R_score"] = _quintile_score(feat["recency_days"], higher_is_better=False)
    feat["F_score"] = _quintile_score(feat["total_orders"], higher_is_better=True)
    feat["M_score"] = _quintile_score(feat["total_revenue"], higher_is_better=True)
    feat["RFM_score"] = feat["R_score"] + feat["F_score"] + feat["M_score"]
    feat["rfm_segment"] = [
        assign_rfm_segment(r, f, m)
        for r, f, m in zip(feat["R_score"], feat["F_score"], feat["M_score"])
    ]

    freq_norm = feat["total_orders"] / feat["total_orders"].max()
    feat["engagement_score"] = (
        0.45 * feat["app_usage_score"]
        + 0.25 * feat["email_opt_in"].astype(float)
        + 0.30 * freq_norm
    )
    feat["email_response_rate"] = feat["email_opt_in"].astype(float) * (
        0.35 + 0.65 * feat["app_usage_score"]
    )
    feat["app_activity_velocity"] = feat["app_usage_score"] / np.maximum(
        feat["tenure_days"] / 365, 0.25
    )

    temp = order_meta.groupby("customer_id").agg(
        q4_order_share=("quarter", lambda s: (s == 4).mean()),
        holiday_order_share=("is_holiday", "mean"),
        payday_window_share=("is_payday_window", "mean"),
    )
    global_holiday = order_meta["is_holiday"].mean()
    temp["seasonality_index"] = temp["q4_order_share"]
    temp["payday_effect"] = temp["payday_window_share"]
    temp["holiday_sensitivity"] = temp["holiday_order_share"] / max(global_holiday, 1e-6)
    feat = feat.merge(temp, on="customer_id")

    feat["tier_ordinal"] = feat["tier"].map(TIER_ORDINAL)
    feat["tier_progression_speed"] = feat["tier_ordinal"] / np.log1p(feat["tenure_days"] / 30)
    points_earned_proxy = feat["total_revenue"] * 2
    feat["reward_redemption_rate"] = (
        1 - feat["points_balance"] / np.maximum(points_earned_proxy, 1)
    ).clip(0, 1)
    feat["churn_inertia_score"] = feat["recency_days"] / np.maximum(
        feat["avg_interorder_days"].fillna(feat["recency_days"]), 1
    )
    feat["inactivity_churn_label"] = (feat["recency_days"] >= INACTIVITY_CHURN_DAYS).astype(int)
    feat["feature_as_of_date"] = as_of
    return feat
