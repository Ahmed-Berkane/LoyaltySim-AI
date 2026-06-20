"""Compute EDA summary stats for notebook finding comments."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "modeling"

fact = pd.read_parquet(DATA / "fact_transactions.parquet")
fact["order_date"] = pd.to_datetime(fact["order_date"])
dim = pd.read_parquet(DATA / "dim_customers.parquet")
feat = pd.read_parquet(DATA / "customer_features.parquet")
telco = pd.read_parquet(DATA / "telco_customers.parquet")

stats_out = {}

# Load inventory
stats_out["n_lines"] = len(fact)
stats_out["n_orders"] = fact["order_id"].nunique()
stats_out["n_customers"] = fact["customer_id"].nunique()
stats_out["date_min"] = str(fact["order_date"].min().date())
stats_out["date_max"] = str(fact["order_date"].max().date())
stats_out["total_sales"] = float(fact["sales"].sum())
stats_out["total_profit"] = float(fact["profit"].sum())
stats_out["margin_pct"] = stats_out["total_profit"] / stats_out["total_sales"] * 100

# Missingness
null_cols = fact.columns[fact.isnull().any()].tolist()
stats_out["null_cols"] = null_cols
stats_out["null_pct"] = {c: float(fact[c].isnull().mean() * 100) for c in null_cols}

# Monthly revenue trend
monthly = fact.assign(month_dt=fact["order_date"].dt.to_period("M")).groupby("month_dt")["sales"].sum()
stats_out["monthly_first"] = float(monthly.iloc[0])
stats_out["monthly_last"] = float(monthly.iloc[-1])
stats_out["monthly_pct_change"] = (stats_out["monthly_last"] / stats_out["monthly_first"] - 1) * 100
stats_out["best_month"] = str(monthly.idxmax())
stats_out["worst_month"] = str(monthly.idxmin())

# Lorenz / Pareto
cust = dim.sort_values("total_sales", ascending=True)
cum_share = cust["total_sales"].cumsum() / cust["total_sales"].sum()
stats_out["gini"] = float(1 - 2 * np.trapezoid(cum_share.values, dx=1 / len(cum_share)))
stats_out["top10_share"] = float(cust.nlargest(int(len(cust) * 0.1), "total_sales")["total_sales"].sum() / cust["total_sales"].sum() * 100)
stats_out["top20_share"] = float(cust.nlargest(int(len(cust) * 0.2), "total_sales")["total_sales"].sum() / cust["total_sales"].sum() * 100)

# Discount vs margin
fact_disc = fact.assign(discount_bucket=pd.cut(fact["discount"], bins=[-0.01, 0, 0.1, 0.2, 0.4, 1.0]))
disc_margin = fact_disc.groupby("discount_bucket", observed=True).agg(
    sales=("sales", "sum"), profit=("profit", "sum"), margin=("profit", lambda x: x.sum() / fact_disc.loc[x.index, "sales"].sum())
)
stats_out["zero_disc_margin"] = float(
    fact_disc.loc[fact_disc["discount"] == 0, "profit"].sum()
    / fact_disc.loc[fact_disc["discount"] == 0, "sales"].sum() * 100
)
stats_out["high_disc_margin"] = float(
    fact_disc.loc[fact_disc["discount"] >= 0.2, "profit"].sum()
    / fact_disc.loc[fact_disc["discount"] >= 0.2, "sales"].sum() * 100
)
stats_out["pct_lines_discounted"] = float((fact["discount"] > 0).mean() * 100)

# RFM segments
seg_counts = feat["rfm_segment"].value_counts(normalize=True).mul(100).round(1).to_dict()
stats_out["rfm_segments"] = seg_counts
stats_out["at_risk_pct"] = float((feat["rfm_segment"] == "At Risk").mean() * 100)
stats_out["champions_pct"] = float((feat["rfm_segment"] == "Champions").mean() * 100)

# Tier AOV
orders = fact.groupby(["order_id", "customer_id"], as_index=False).agg(order_sales=("sales", "sum"))
orders = orders.merge(feat[["customer_id", "tier"]], on="customer_id")
tier_aov = orders.groupby("tier")["order_sales"].mean().round(2).to_dict()
stats_out["tier_aov"] = tier_aov

# Feature correlations with churn proxy
churn_corr = feat.select_dtypes(include=[np.number]).corr()["inactivity_churn_label"].drop("inactivity_churn_label").sort_values(key=abs, ascending=False)
stats_out["top_churn_corr"] = churn_corr.head(5).round(3).to_dict()

# Calendar lift
order_ctx = fact.groupby("order_id", as_index=False).agg(
    order_sales=("sales", "sum"),
    is_holiday=("is_us_federal_holiday", "max"),
    is_retail_day=("is_retail_spending_day", "max"),
)
for label in ["is_holiday", "is_retail_day"]:
    a = order_ctx.loc[order_ctx[label], "order_sales"]
    b = order_ctx.loc[~order_ctx[label], "order_sales"]
    stats_out[f"{label}_aov"] = float(a.mean())
    stats_out[f"{label}_lift_pct"] = float((a.mean() / b.mean() - 1) * 100)
    stats_out[f"{label}_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)

# Weather
wx = fact.groupby("order_id", as_index=False).agg(
    order_sales=("sales", "sum"), temp_c=("temp_c", "mean"), had_rain=("had_rain", "max")
).dropna(subset=["temp_c"])
rain_a = wx.loc[wx["had_rain"], "order_sales"]
rain_b = wx.loc[~wx["had_rain"], "order_sales"]
stats_out["rain_aov"] = float(rain_a.mean())
stats_out["dry_aov"] = float(rain_b.mean())
stats_out["rain_p"] = float(stats.ttest_ind(rain_a, rain_b, equal_var=False).pvalue)

# Category margin
cat = fact.groupby("category", as_index=False).agg(sales=("sales", "sum"), profit=("profit", "sum"))
cat["margin_pct"] = cat["profit"] / cat["sales"] * 100
stats_out["best_category"] = cat.loc[cat["margin_pct"].idxmax(), "category"]
stats_out["best_cat_margin"] = float(cat["margin_pct"].max())
stats_out["worst_category"] = cat.loc[cat["margin_pct"].idxmin(), "category"]
stats_out["worst_cat_margin"] = float(cat["margin_pct"].min())

# Telco
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
stats_out["telco_churn_rate"] = float(telco["churn_flag"].mean() * 100)
month_churn = telco.groupby("Contract")["churn_flag"].mean().sort_values(ascending=False)
stats_out["contract_churn"] = (month_churn * 100).round(1).to_dict()
stats_out["month_to_month_churn"] = float(month_churn.get("Month-to-month", 0) * 100)
stats_out["two_year_churn"] = float(month_churn.get("Two year", 0) * 100)
churned_tenure = telco.loc[telco["churn_flag"] == 1, "tenure"].median()
retained_tenure = telco.loc[telco["churn_flag"] == 0, "tenure"].median()
stats_out["churned_median_tenure"] = float(churned_tenure)
stats_out["retained_median_tenure"] = float(retained_tenure)

# Instacart
inst_path = DATA / "instacart_users.parquet"
if inst_path.exists():
    ic = pd.read_parquet(inst_path)
    stats_out["instacart_users"] = len(ic)
    stats_out["instacart_median_orders"] = float(ic["total_orders"].median())
    stats_out["instacart_reorder_rate"] = float(ic["reorder_rate"].median() * 100)

# Payday/holiday order share
order_meta = fact.groupby("order_id", as_index=False).agg(
    order_sales=("sales", "sum"),
    is_holiday=("is_us_federal_holiday", "max"),
    is_month_start=("is_month_start", "max"),
    is_month_end=("is_month_end", "max"),
).assign(is_payday_window=lambda d: d["is_month_start"] | d["is_month_end"])
stats_out["payday_order_pct"] = float(order_meta["is_payday_window"].mean() * 100)
stats_out["holiday_order_pct"] = float(order_meta["is_holiday"].mean() * 100)

# Engagement by tier
eng_by_tier = feat.groupby("tier")["engagement_score"].mean().round(3).to_dict()
stats_out["eng_by_tier"] = eng_by_tier

print(json.dumps(stats_out, indent=2, default=str))
