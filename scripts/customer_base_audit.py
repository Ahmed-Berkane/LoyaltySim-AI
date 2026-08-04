"""Customer-base audit spine (Fader / Hardie / Ross) for UCI Online Retail II.

Builds the order-level customer×time view used by the Five Lenses audit and by
noncontractual CLV (BG/NBD + Gamma-Gamma). Revenue-only — no COGS/margin.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AUDIT_DIRNAME = "audit"


def default_audit_dir(project_root: Path | str) -> Path:
    return Path(project_root) / "data" / "modeling" / AUDIT_DIRNAME


def load_fact_transactions(path: Path | str) -> pd.DataFrame:
    """Load line-grain fact and coerce order_date."""
    df = pd.read_parquet(path)
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    required = {"customer_id", "order_id", "order_date", "line_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fact missing columns: {sorted(missing)}")
    return df


def build_orders(fact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate invoice lines → one row per order (customer × order_id)."""
    g = (
        fact.groupby(["customer_id", "order_id"], as_index=False)
        .agg(
            order_date=("order_date", "min"),
            spend=("line_total", "sum"),
            n_lines=("line_total", "size"),
        )
    )
    g["order_date"] = pd.to_datetime(g["order_date"])
    g = g.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    return g


def attach_acquisition(orders: pd.DataFrame) -> pd.DataFrame:
    """Add first-order date and quarterly acquisition cohort.

    Idempotent: safe to call when ``acquisition_date`` / ``cohort_id`` already exist.
    """
    out = orders.copy()
    if "acquisition_date" not in out.columns:
        out["acquisition_date"] = out.groupby("customer_id")["order_date"].transform("min")
    else:
        out["acquisition_date"] = pd.to_datetime(out["acquisition_date"])
    out["cohort_id"] = out["acquisition_date"].dt.to_period("Q").astype(str)
    return out


def customer_period_panel(
    orders: pd.DataFrame,
    *,
    freq: str = "Q",
) -> pd.DataFrame:
    """Customer × period rollups: n_orders, spend, active flag.

    ``freq`` is a pandas period alias (``Q``, ``Y``, ``M``, …).
    """
    o = orders.copy()
    o["period"] = o["order_date"].dt.to_period(freq).astype(str)
    panel = (
        o.groupby(["customer_id", "period"], as_index=False)
        .agg(n_orders=("order_id", "nunique"), spend=("spend", "sum"))
    )
    panel["active"] = 1
    if "acquisition_date" in o.columns:
        acq = o.groupby("customer_id")["acquisition_date"].first()
        panel = panel.merge(acq.rename("acquisition_date"), on="customer_id", how="left")
    if "cohort_id" in o.columns:
        ch = o.groupby("customer_id")["cohort_id"].first()
        panel = panel.merge(ch, on="customer_id", how="left")
    return panel.sort_values(["period", "customer_id"]).reset_index(drop=True)


def customer_lifetime_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Per-customer lifetime stats from the order spine."""
    o = attach_acquisition(orders) if "acquisition_date" not in orders.columns else orders
    g = o.groupby("customer_id", as_index=False).agg(
        acquisition_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        n_orders=("order_id", "nunique"),
        total_spend=("spend", "sum"),
    )
    g["aov"] = g["total_spend"] / g["n_orders"].clip(lower=1)
    g["is_one_time_buyer"] = (g["n_orders"] == 1).astype(int)
    g["cohort_id"] = g["acquisition_date"].dt.to_period("Q").astype(str)
    return g


def period_active_customers(
    orders: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Customers with ≥1 order in [period_start, period_end], with period spend stats."""
    mask = (orders["order_date"] >= period_start) & (orders["order_date"] <= period_end)
    sub = orders.loc[mask]
    if sub.empty:
        return pd.DataFrame(
            columns=["customer_id", "n_orders", "spend", "aov", "is_one_order"]
        )
    g = sub.groupby("customer_id", as_index=False).agg(
        n_orders=("order_id", "nunique"),
        spend=("spend", "sum"),
    )
    g["aov"] = g["spend"] / g["n_orders"].clip(lower=1)
    g["is_one_order"] = (g["n_orders"] == 1).astype(int)
    return g


def whale_curve(spend: pd.Series) -> pd.DataFrame:
    """Lorenz / whale curve from customer spend (descending).

    Returns cumulative % customers and cumulative % revenue, plus
    ``pct_customers_for_half_revenue`` (share of customers needed for 50% of £).
    """
    s = spend.fillna(0).astype(float).sort_values(ascending=False).reset_index(drop=True)
    total = s.sum()
    n = len(s)
    if n == 0 or total <= 0:
        return pd.DataFrame(
            {
                "rank": [],
                "pct_customers": [],
                "pct_revenue": [],
                "spend": [],
            }
        )
    cum = s.cumsum()
    out = pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "pct_customers": (np.arange(1, n + 1) / n) * 100.0,
            "pct_revenue": (cum / total) * 100.0,
            "spend": s.values,
        }
    )
    half_idx = int(np.searchsorted(out["pct_revenue"].values, 50.0, side="left"))
    half_idx = min(half_idx, n - 1)
    out.attrs["pct_customers_for_half_revenue"] = float(out.loc[half_idx, "pct_customers"])
    out.attrs["total_spend"] = float(total)
    out.attrs["n_customers"] = int(n)
    return out


def multiplicative_decomposition(active: pd.DataFrame) -> dict[str, float]:
    """Lens-1 style: total spend = #buyers × AOF × AOV."""
    n_buyers = float(len(active))
    if n_buyers == 0:
        return {"n_buyers": 0.0, "aof": 0.0, "aov": 0.0, "total_spend": 0.0}
    total_orders = float(active["n_orders"].sum())
    total_spend = float(active["spend"].sum())
    aof = total_orders / n_buyers
    aov = total_spend / total_orders if total_orders else 0.0
    return {
        "n_buyers": n_buyers,
        "aof": aof,
        "aov": aov,
        "total_spend": total_spend,
        "pct_one_order": float(active["is_one_order"].mean() * 100.0)
        if "is_one_order" in active.columns
        else float((active["n_orders"] == 1).mean() * 100.0),
    }


def period_vs_period(
    orders: pd.DataFrame,
    period_a: tuple[pd.Timestamp, pd.Timestamp],
    period_b: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, object]:
    """Lens 2: retained / lapsed / newly active (in B not A) / reactivated vs acquired.

    ``newly_active_in_b`` = active in B only (includes true new + reactivations from
    before A). Callers with acquisition dates can further split.
    """
    a = set(period_active_customers(orders, *period_a)["customer_id"])
    b = set(period_active_customers(orders, *period_b)["customer_id"])
    retained = a & b
    lapsed = a - b
    only_b = b - a
    return {
        "n_active_a": len(a),
        "n_active_b": len(b),
        "n_retained": len(retained),
        "n_lapsed": len(lapsed),
        "n_only_b": len(only_b),
        "retained_ids": retained,
        "lapsed_ids": lapsed,
        "only_b_ids": only_b,
    }


def same_customer_up_down(
    orders: pd.DataFrame,
    period_a: tuple[pd.Timestamp, pd.Timestamp],
    period_b: tuple[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    """For customers active in both periods: ↑/↓ on n_orders and AOV."""
    ga = period_active_customers(orders, *period_a).set_index("customer_id")
    gb = period_active_customers(orders, *period_b).set_index("customer_id")
    both = ga.index.intersection(gb.index)
    if len(both) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "n_orders_a": ga.loc[both, "n_orders"],
            "n_orders_b": gb.loc[both, "n_orders"],
            "aov_a": ga.loc[both, "aov"],
            "aov_b": gb.loc[both, "aov"],
            "spend_a": ga.loc[both, "spend"],
            "spend_b": gb.loc[both, "spend"],
        }
    )
    out["orders_up"] = out["n_orders_b"] >= out["n_orders_a"]
    out["aov_up"] = out["aov_b"] >= out["aov_a"]
    out["spend_up"] = out["spend_b"] >= out["spend_a"]
    return out.reset_index()


def cohort_members(orders: pd.DataFrame, cohort_id: str) -> pd.Index:
    o = attach_acquisition(orders) if "cohort_id" not in orders.columns else orders
    return pd.Index(o.loc[o["cohort_id"] == cohort_id, "customer_id"].unique())


def cohort_evolution(
    orders: pd.DataFrame,
    cohort_id: str,
    *,
    freq: str = "Q",
) -> pd.DataFrame:
    """Lens 3/4: % active, AOF, AOV by period for one acquisition cohort."""
    o = attach_acquisition(orders) if "cohort_id" not in orders.columns else orders.copy()
    members = set(cohort_members(o, cohort_id))
    if not members:
        return pd.DataFrame(
            columns=["period", "cohort_size", "n_active", "pct_active", "aof", "aov", "spend"]
        )
    cohort_size = len(members)
    sub = o[o["customer_id"].isin(members)].copy()
    sub["period"] = sub["order_date"].dt.to_period(freq).astype(str)
    rows = []
    for period, g in sub.groupby("period"):
        active = g.groupby("customer_id").agg(
            n_orders=("order_id", "nunique"),
            spend=("spend", "sum"),
        )
        n_active = len(active)
        total_orders = float(active["n_orders"].sum())
        total_spend = float(active["spend"].sum())
        rows.append(
            {
                "period": period,
                "cohort_id": cohort_id,
                "cohort_size": cohort_size,
                "n_active": n_active,
                "pct_active": 100.0 * n_active / cohort_size,
                "aof": total_orders / n_active if n_active else 0.0,
                "aov": total_spend / total_orders if total_orders else 0.0,
                "spend": total_spend,
            }
        )
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def interpurchase_gaps(orders: pd.DataFrame, customer_ids: Iterable | None = None) -> pd.Series:
    """Days between consecutive orders (excludes first order per customer)."""
    o = orders.sort_values(["customer_id", "order_date"])
    if customer_ids is not None:
        o = o[o["customer_id"].isin(customer_ids)]
    gaps = o.groupby("customer_id")["order_date"].diff().dt.days
    return gaps.dropna()


def ever_repeat_rate(orders: pd.DataFrame, cohort_id: str) -> float:
    """Share of cohort with ≥2 lifetime orders (to end of data)."""
    o = attach_acquisition(orders) if "cohort_id" not in orders.columns else orders
    members = cohort_members(o, cohort_id)
    if len(members) == 0:
        return 0.0
    n_orders = o[o["customer_id"].isin(members)].groupby("customer_id")["order_id"].nunique()
    return float((n_orders >= 2).mean())


def vtd_deciles(orders: pd.DataFrame, cohort_id: str) -> pd.DataFrame:
    """Value-to-date deciles for a cohort (cumulative spend through obs end)."""
    o = attach_acquisition(orders) if "cohort_id" not in orders.columns else orders
    members = cohort_members(o, cohort_id)
    spend = (
        o[o["customer_id"].isin(members)]
        .groupby("customer_id")
        .agg(vtd=("spend", "sum"), n_orders=("order_id", "nunique"))
    )
    if spend.empty:
        return spend
    # qcut can fail on ties; rank then bin
    spend = spend.sort_values("vtd", ascending=False)
    try:
        spend["decile"] = pd.qcut(spend["vtd"].rank(method="first"), 10, labels=False) + 1
        # 1 = highest VTD
        spend["decile"] = 11 - spend["decile"]
    except ValueError:
        spend["decile"] = 1
    total = spend["vtd"].sum()
    summary = (
        spend.groupby("decile", as_index=False)
        .agg(
            n_customers=("vtd", "size"),
            pct_cohort=("vtd", "size"),
            total_vtd=("vtd", "sum"),
            avg_vtd=("vtd", "mean"),
            avg_orders=("n_orders", "mean"),
        )
    )
    summary["pct_cohort"] = 100.0 * summary["n_customers"] / len(spend)
    summary["pct_vtd"] = 100.0 * summary["total_vtd"] / total if total else 0.0
    return summary.sort_values("decile").reset_index(drop=True)


def stacked_active_by_cohort(
    orders: pd.DataFrame,
    *,
    freq: str = "Q",
) -> pd.DataFrame:
    """Lens 5: n active customers per period × acquisition cohort."""
    o = attach_acquisition(orders) if "cohort_id" not in orders.columns else orders.copy()
    o["period"] = o["order_date"].dt.to_period(freq).astype(str)
    active = (
        o.groupby(["period", "cohort_id"])["customer_id"]
        .nunique()
        .rename("n_active")
        .reset_index()
    )
    return active.sort_values(["period", "cohort_id"]).reset_index(drop=True)


def second_purchase_cumulative(
    orders: pd.DataFrame,
    cohort_id: str,
    *,
    windows_days: tuple[int, ...] = (90, 180, 365, 730),
) -> pd.DataFrame:
    """Cumulative % of cohort making a 2nd purchase within N days of acquisition."""
    o = attach_acquisition(orders) if "acquisition_date" not in orders.columns else orders.copy()
    members = cohort_members(o, cohort_id)
    if len(members) == 0:
        return pd.DataFrame(columns=["window_days", "pct_second_purchase"])
    first = o.groupby("customer_id")["order_date"].min()
    # second purchase date
    ranked = o.sort_values(["customer_id", "order_date"])
    ranked["ord_n"] = ranked.groupby("customer_id").cumcount() + 1
    second = ranked.loc[ranked["ord_n"] == 2, ["customer_id", "order_date"]].set_index(
        "customer_id"
    )["order_date"]
    rows = []
    for w in windows_days:
        n_hit = 0
        for cid in members:
            if cid not in second.index:
                continue
            if (second[cid] - first[cid]).days <= w:
                n_hit += 1
        rows.append(
            {
                "cohort_id": cohort_id,
                "window_days": w,
                "pct_second_purchase": 100.0 * n_hit / len(members),
            }
        )
    return pd.DataFrame(rows)


def rfm_as_of(
    orders: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Classic RFM scores on orders with order_date ≤ as_of."""
    sub = orders[orders["order_date"] <= as_of]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby("customer_id").agg(
        last_order=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("spend", "sum"),
    )
    g["recency_days"] = (as_of - g["last_order"]).dt.days
    # Quintiles: R high = recent (invert), F/M high = more
    g["R_score"] = pd.qcut(g["recency_days"].rank(method="first"), n_bins, labels=False) + 1
    g["R_score"] = (n_bins + 1) - g["R_score"]
    g["F_score"] = pd.qcut(g["frequency"].rank(method="first"), n_bins, labels=False) + 1
    g["M_score"] = pd.qcut(g["monetary"].rank(method="first"), n_bins, labels=False) + 1
    g["RFM_score"] = g["R_score"] + g["F_score"] + g["M_score"]
    return g.reset_index()


def repeat_buying_rate(
    orders: pd.DataFrame,
    period_a: tuple[pd.Timestamp, pd.Timestamp],
    period_b: tuple[pd.Timestamp, pd.Timestamp],
) -> float:
    """Share of customers active in A who are also active in B."""
    a = set(period_active_customers(orders, *period_a)["customer_id"])
    if not a:
        return 0.0
    b = set(period_active_customers(orders, *period_b)["customer_id"])
    return float(len(a & b) / len(a))


def build_customer_master(
    orders: pd.DataFrame,
    *,
    obs_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """One row per customer for consulting Phase 3 (master customer table)."""
    o = attach_acquisition(orders) if "acquisition_date" not in orders.columns else orders.copy()
    end = pd.Timestamp(obs_end) if obs_end is not None else pd.Timestamp(o["order_date"].max())
    g = o.groupby("customer_id", as_index=False).agg(
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
        n_orders=("order_id", "nunique"),
        revenue=("spend", "sum"),
        cohort_id=("cohort_id", "first"),
    )
    g["tenure_days"] = (g["last_purchase"] - g["first_purchase"]).dt.days
    g["tenure_weeks"] = (g["tenure_days"] / 7.0).round(2)
    g["customer_age_days"] = (end - g["first_purchase"]).dt.days
    g["customer_age_weeks"] = (g["customer_age_days"] / 7.0).round(2)
    g["recency_days"] = (end - g["last_purchase"]).dt.days
    g["recency_weeks"] = (g["recency_days"] / 7.0).round(2)
    g["aov"] = g["revenue"] / g["n_orders"].clip(lower=1)
    ranked = o.sort_values(["customer_id", "order_date"])
    mean_gap = (
        ranked.groupby("customer_id")["order_date"]
        .diff()
        .dt.days
        .groupby(ranked["customer_id"])
        .mean()
        .rename("mean_interpurchase_days")
    )
    g = g.merge(mean_gap, on="customer_id", how="left")
    g["is_one_time_buyer"] = (g["n_orders"] == 1).astype(int)
    return g.sort_values("revenue", ascending=False).reset_index(drop=True)


def monthly_acquisition(orders: pd.DataFrame) -> pd.DataFrame:
    """New customers by acquisition month + cumulative base."""
    o = attach_acquisition(orders) if "acquisition_date" not in orders.columns else orders.copy()
    first = o.groupby("customer_id", as_index=False)["acquisition_date"].min()
    first["acq_month"] = first["acquisition_date"].dt.to_period("M").astype(str)
    m = (
        first.groupby("acq_month", as_index=False)
        .agg(n_new=("customer_id", "nunique"))
        .sort_values("acq_month")
        .reset_index(drop=True)
    )
    m["cumulative_customers"] = m["n_new"].cumsum()
    return m


def new_vs_repeat_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Monthly order counts split into first-ever vs repeat orders."""
    o = attach_acquisition(orders) if "acquisition_date" not in orders.columns else orders.copy()
    o = o.sort_values(["customer_id", "order_date", "order_id"]).copy()
    o["order_rank"] = o.groupby("customer_id").cumcount() + 1
    o["is_first_order"] = (o["order_rank"] == 1).astype(int)
    o["month"] = o["order_date"].dt.to_period("M").astype(str)
    g = (
        o.groupby("month", as_index=False)
        .agg(
            n_orders=("order_id", "nunique"),
            n_first_orders=("is_first_order", "sum"),
            spend=("spend", "sum"),
        )
        .sort_values("month")
        .reset_index(drop=True)
    )
    g["n_repeat_orders"] = g["n_orders"] - g["n_first_orders"]
    g["pct_first_orders"] = 100.0 * g["n_first_orders"] / g["n_orders"].clip(lower=1)
    return g


def frequency_distribution(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders-per-customer distribution summary rows."""
    life = customer_lifetime_summary(orders)
    return life[["customer_id", "n_orders", "total_spend", "aov", "is_one_time_buyer"]].copy()


def duration_snapshot(
    orders: pd.DataFrame,
    *,
    active_within_weeks: int = 13,
    obs_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tenure / recency / still-active flag (purchased in last K weeks)."""
    master = build_customer_master(orders, obs_end=obs_end)
    master["still_active"] = (master["recency_weeks"] <= active_within_weeks).astype(int)
    master["active_within_weeks"] = active_within_weeks
    return master


def active_survival_by_tenure_weeks(
    orders: pd.DataFrame,
    *,
    max_weeks: int = 104,
    window_weeks: int = 13,
) -> pd.DataFrame:
    """Share of acquired customers with a purchase in [w, w+window) weeks after acquisition.

    Proxy survival curve for noncontractual data (no cancel date).
    """
    o = attach_acquisition(orders) if "acquisition_date" not in orders.columns else orders.copy()
    first = o.groupby("customer_id")["acquisition_date"].min()
    cohort_size = float(len(first))
    if cohort_size == 0:
        return pd.DataFrame(columns=["week", "pct_active", "n_active"])
    merged = o.merge(first.rename("acq"), left_on="customer_id", right_index=True, how="left")
    age_days = (merged["order_date"] - merged["acq"]).dt.days
    step = max(1, window_weeks // 2)
    rows = []
    for w in range(0, max_weeks + 1, step):
        lo = w * 7
        hi = (w + window_weeks) * 7
        hit = merged.loc[(age_days >= lo) & (age_days < hi), "customer_id"].nunique()
        rows.append({"week": w, "pct_active": 100.0 * hit / cohort_size, "n_active": int(hit)})
    return pd.DataFrame(rows)


def monetary_concentration(orders: pd.DataFrame) -> dict[str, float]:
    """Top-decile / top-percentile spend shares + whale half-revenue point."""
    life = customer_lifetime_summary(orders)
    spend = life["total_spend"].sort_values(ascending=False)
    total = float(spend.sum())
    n = len(spend)
    if n == 0 or total <= 0:
        return {
            "n_customers": 0.0,
            "top1_share": 0.0,
            "top10_share": 0.0,
            "bottom90_share": 0.0,
            "pct_customers_for_half_revenue": float("nan"),
        }
    whale = whale_curve(spend)
    top1_n = max(int(n * 0.01), 1)
    top10_n = max(int(n * 0.10), 1)
    top1 = float(spend.head(top1_n).sum()) / total
    top10 = float(spend.head(top10_n).sum()) / total
    return {
        "n_customers": float(n),
        "top1_share": top1,
        "top10_share": top10,
        "bottom90_share": 1.0 - top10,
        "pct_customers_for_half_revenue": float(
            whale.attrs.get("pct_customers_for_half_revenue", float("nan"))
        ),
    }


def export_audit_summaries(
    project_root: Path | str,
    orders: pd.DataFrame | None = None,
    fact_path: Path | str | None = None,
) -> dict[str, Path]:
    """Write reusable audit parquets under ``data/modeling/audit/``."""
    root = Path(project_root)
    out_dir = default_audit_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if orders is None:
        fact = load_fact_transactions(
            fact_path or (root / "data" / "modeling" / "uci_fact_transactions.parquet")
        )
        orders = build_orders(fact)
    orders = attach_acquisition(orders)

    paths: dict[str, Path] = {}
    orders_path = out_dir / "uci_orders.parquet"
    orders.to_parquet(orders_path, index=False)
    paths["orders"] = orders_path

    lifetime = customer_lifetime_summary(orders)
    life_path = out_dir / "uci_customer_lifetime.parquet"
    lifetime.to_parquet(life_path, index=False)
    paths["lifetime"] = life_path

    master = build_customer_master(orders)
    master_path = out_dir / "uci_customer_master.parquet"
    master.to_parquet(master_path, index=False)
    paths["master"] = master_path

    acq = monthly_acquisition(orders)
    acq_path = out_dir / "uci_monthly_acquisition.parquet"
    acq.to_parquet(acq_path, index=False)
    paths["acquisition"] = acq_path

    panel_q = customer_period_panel(orders, freq="Q")
    panel_path = out_dir / "uci_customer_period_Q.parquet"
    panel_q.to_parquet(panel_path, index=False)
    paths["panel_Q"] = panel_path

    stacked = stacked_active_by_cohort(orders, freq="Q")
    stacked_path = out_dir / "uci_active_by_cohort_Q.parquet"
    stacked.to_parquet(stacked_path, index=False)
    paths["stacked_Q"] = stacked_path

    # Lens-1 style snapshot on last full calendar year in the data
    max_d = orders["order_date"].max()
    year = int(max_d.year)
    # Prefer last complete year if data ends mid-year after Jan
    if max_d.month < 12:
        year = year - 1 if year > orders["order_date"].min().year else year
    p_start = pd.Timestamp(f"{year}-01-01")
    p_end = pd.Timestamp(f"{year}-12-31")
    active = period_active_customers(orders, p_start, p_end)
    whale = whale_curve(active["spend"]) if len(active) else whale_curve(pd.Series(dtype=float))
    decomp = multiplicative_decomposition(active) if len(active) else {}
    lens1 = {
        "period_start": str(p_start.date()),
        "period_end": str(p_end.date()),
        **decomp,
        "pct_customers_for_half_revenue": whale.attrs.get("pct_customers_for_half_revenue"),
    }
    lens1_path = out_dir / "uci_lens1_summary.parquet"
    pd.DataFrame([lens1]).to_parquet(lens1_path, index=False)
    paths["lens1"] = lens1_path

    if len(whale):
        whale_path = out_dir / "uci_whale_curve.parquet"
        whale.to_parquet(whale_path, index=False)
        paths["whale"] = whale_path

    return paths


def load_clv_scores(project_root: Path | str) -> pd.DataFrame:
    """Load ``models/01_clv_customer_scores.parquet`` if present."""
    root = Path(project_root)
    for name in ("01_clv_customer_scores.parquet", "03_clv_customer_scores.parquet"):
        path = root / "models" / name
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(
        f"Missing models/01_clv_customer_scores.parquet — run Notebooks/01-clv.ipynb "
        f"(or scripts/smoke_test_modeling.py)."
    )


def join_clv_to_customers(
    customer_features: pd.DataFrame,
    clv_scores: pd.DataFrame,
    *,
    cols: tuple[str, ...] = ("p_alive", "expected_clv", "expected_purchases"),
) -> pd.DataFrame:
    """Left-join forward CLV scores onto the customer feature store."""
    score_cols = [c for c in cols if c in clv_scores.columns]
    # tolerate expected_clv_90d naming
    if "expected_clv" not in score_cols and "expected_clv_90d" in clv_scores.columns:
        score_cols = list(score_cols) + ["expected_clv_90d"]
    if "expected_purchases" not in score_cols and "expected_purchases_holdout" in clv_scores.columns:
        score_cols = list(score_cols) + ["expected_purchases_holdout"]
    keep = ["customer_id", *dict.fromkeys(score_cols)]
    keep = [c for c in keep if c in clv_scores.columns or c == "customer_id"]
    right = clv_scores.reset_index() if "customer_id" not in clv_scores.columns else clv_scores
    if "customer_id" not in right.columns and right.index.name == "customer_id":
        right = right.reset_index()
    cols_present = [c for c in keep if c in right.columns]
    return customer_features.merge(right[cols_present], on="customer_id", how="left")


_CURRENCY_SYMBOLS = {
    "usd": "$",
    "dollar": "$",
    "dollars": "$",
    "eur": "\u20ac",
    "euro": "\u20ac",
    "euros": "\u20ac",
    "gbp": "\u00a3",
    "pound": "\u00a3",
    "pounds": "\u00a3",
    "sterling": "\u00a3",
    "jpy": "\u00a5",
    "yen": "\u00a5",
    "cny": "\u00a5",
    "yuan": "\u00a5",
    "rmb": "\u00a5",
    "inr": "\u20b9",
    "rupee": "\u20b9",
    "rupees": "\u20b9",
    "krw": "\u20a9",
    "won": "\u20a9",
}


def _resolve_currency_symbol(currency: str) -> str:
    """Map a currency name/code (e.g. ``"Euro"``, ``"EUR"``) to its symbol.

    Falls through unrecognized input (e.g. an already-a-symbol string like
    ``"$"`` or ``"CHF"``) unchanged.
    """
    return _CURRENCY_SYMBOLS.get(currency.strip().lower(), currency)


def _format_spend(x: float, currency: str = "$") -> str:
    currency = _resolve_currency_symbol(currency)
    if x >= 1000:
        return f"{currency}{x / 1000:g}K"
    return f"{currency}{x:,.0f}"


def _format_number(x: float) -> str:
    return f"{x:,.0f}"


def _format_percent(x: float) -> str:
    return f"{x * 100:,.0f}%"


def hist_plot(
    data: pd.DataFrame,
    threshold: float,
    bin_width: float,
    *,
    x_col: str = "Total_spend",
    title: str = "Distribution of Customer Total Spend",
    xlabel: str = "Total Spend",
    value_kind: str = "currency",
    currency: str = "$",
    value_fmt: Callable[[float], str] | None = None,
    max_labelled_bins: int = 20,
    bin_start: float | None = None,
    negative_bin: bool = False,
    edge_margin: float = 0.02,
    figsize: tuple[float, float] = (14, 7),
    print_table: bool = False,
):
    """Plot the distribution of ``x_col`` as a %/# bar chart.

    Bins ``x_col`` into ``bin_width``-wide buckets, with a final catch-all
    bar for anything at/above ``threshold``. Left axis shows % of customers
    per bin, right axis shows the raw customer count. Mean and median are
    marked with vertical lines.

    By default the first bin starts at ``floor(x_col.min() / bin_width) *
    bin_width`` (pass ``bin_start`` to override) so there's no guaranteed-
    empty leading bin — e.g. a transaction-count column whose minimum is 1
    won't waste a bin on an impossible "0" bucket.

    ``negative_bin``, when True, groups every negative value into its own
    catch-all ``"<0"`` bar at the low end (mirroring the ``threshold+``
    catch-all at the high end) and starts the regular bins at 0 — use this
    for columns that can go negative, e.g. profit or margin. Defaults to
    False, i.e. bins start at the data minimum as usual.

    ``value_kind`` picks the default formatter/wording: ``"currency"``
    (default) renders values with the ``currency`` symbol/K-suffix (e.g.
    ``$1.2K``); ``"number"`` renders plain integers with no prefix (e.g.
    ``7``, ``10+``) — use this for count-like columns such as number of
    transactions; ``"percent"`` renders a 0\u20131-scaled ratio as a
    percentage (e.g. ``0.05`` \u2192 ``"5%"``) — use this for margin/rate
    columns, typically together with a small ``bin_width`` like ``0.05``
    for 5-point-wide bins. Ignored if ``value_fmt`` is passed explicitly.

    ``currency`` sets the symbol/prefix used when ``value_kind="currency"``
    (e.g. ``currency="Euro"`` or ``currency="\u20ac"``). Ignored otherwise.

    ``value_fmt`` controls how bin edges/mean/median are rendered as text,
    overriding ``value_kind``/``currency`` entirely — pass something like
    ``lambda x: f"{x:.1f}%"`` for custom formatting.

    If the number of regular bins is at most ``max_labelled_bins``, every
    bin gets its own tick label, named after its *lower* edge (e.g. a
    ``bin_width=1`` bin covering ``[3, 4)`` is labelled ``3``). Otherwise
    bins are grouped into ~10 labelled ticks to keep the axis readable.

    If ``print_table`` is True, also builds and displays a small summary
    table (customer count, total/mean/median/min/max, share of customers
    in the low/high catch-all bins, and the share of customers below the
    mean).
    """
    if value_kind not in ("currency", "number", "percent"):
        raise ValueError(f"value_kind must be 'currency', 'number', or 'percent', got {value_kind!r}")

    currency = _resolve_currency_symbol(currency)
    if value_fmt is None:
        if value_kind == "currency":
            value_fmt = lambda x: _format_spend(x, currency)
        elif value_kind == "percent":
            value_fmt = _format_percent
        else:
            value_fmt = _format_number
    spend = data[x_col].dropna()

    if bin_start is None:
        bin_start = 0.0 if negative_bin else np.floor(spend.min() / bin_width) * bin_width

    regular_bins = np.arange(bin_start, threshold + bin_width, bin_width)
    if negative_bin:
        bins = np.concatenate([[-np.inf], regular_bins, [np.inf]])
    else:
        bins = np.append(regular_bins, np.inf)

    counts, edges = np.histogram(spend, bins=bins)
    percent = counts / counts.sum() * 100

    mean_spend = spend.mean()
    median_spend = spend.median()

    # Plot on finite edges only: swap the leading/trailing `inf` edges for
    # `bin_start - bin_width`/`threshold + bin_width` so the catch-all bars
    # are the same width as a regular bin instead of overlapping/collapsing
    # onto the bin next to them.
    plot_edges = edges.copy()
    if negative_bin:
        plot_edges[0] = bin_start - bin_width
    plot_edges[-1] = threshold + bin_width
    widths = np.diff(plot_edges)
    centers = plot_edges[:-1] + widths / 2

    _fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(centers, percent, width=widths, alpha=0.7, edgecolor="black")

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Customer %")
    ax1.set_title(title)

    # Pin the x-axis close to the actual bar range (plus a thin margin) so
    # bars don't touch the y-axis/right edge but there's no large empty gap.
    span = plot_edges[-1] - plot_edges[0]
    pad = span * edge_margin
    ax1.set_xlim(plot_edges[0] - pad, plot_edges[-1] + pad)

    n_regular_bins = round((threshold - bin_start) / bin_width)
    regular_centers = centers[1:-1] if negative_bin else centers[:-1]
    neg_center = centers[0] if negative_bin else None
    catchall_center = centers[-1]

    if n_regular_bins <= max_labelled_bins:
        # Few enough bins to label every one of them individually, by
        # their lower edge (the value the bin actually represents).
        tick_positions = list(regular_centers)
        tick_labels = [value_fmt(x - bin_width / 2) for x in tick_positions]
    else:
        # Group bins so we end up with roughly 10 labelled ticks.
        bins_per_tick = max(1, round(n_regular_bins / 10))
        tick_step = bins_per_tick * bin_width
        tick_positions = list(np.arange(bin_start + tick_step / 2, threshold, tick_step))
        tick_labels = [
            f"{value_fmt(x - tick_step / 2)}\u2013{value_fmt(x + tick_step / 2)}"
            for x in tick_positions
        ]

    # Drop any regular ticks that sit too close to the final "threshold+" tick,
    # otherwise their labels overlap and become unreadable.
    min_gap = bin_width / 2
    keep = [i for i, x in enumerate(tick_positions) if catchall_center - x >= min_gap]
    tick_positions = [tick_positions[i] for i in keep]
    tick_labels = [tick_labels[i] for i in keep]

    if negative_bin:
        tick_positions.insert(0, neg_center)
        tick_labels.insert(0, "<0")

    tick_positions.append(catchall_center)
    tick_labels.append(f"{value_fmt(threshold)}+")

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    # Keep the mean/median markers within the plotted range (their labels
    # still show the real, unclipped value) — matters most with
    # `negative_bin`, where extreme negative outliers are all folded into
    # the single "<0" bar and could otherwise sit far outside the axis.
    def _clip(x: float) -> float:
        return min(max(x, plot_edges[0]), plot_edges[-1])

    ax1.axvline(
        _clip(mean_spend),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {value_fmt(mean_spend)}",
    )
    ax1.axvline(
        _clip(median_spend),
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Median: {value_fmt(median_spend)}",
    )
    ax1.legend(loc="upper right", frameon=True)

    ax2 = ax1.twinx()
    ax2.set_ylim(
        ax1.get_ylim()[0] / 100 * counts.sum(),
        ax1.get_ylim()[1] / 100 * counts.sum(),
    )
    ax2.set_ylabel("Customer #")

    plt.tight_layout()
    plt.show()

    if print_table:
        n_customers = len(spend)
        total_revenue = spend.sum()

        if negative_bin:
            low_bin_count = int((spend < 0).sum())
        else:
            low_bin_count = int((spend < bin_width).sum())
        low_bin_pct = low_bin_count / n_customers * 100

        high_bin_count = int((spend >= threshold).sum())
        high_bin_pct = high_bin_count / n_customers * 100

        below_mean_count = int((spend < mean_spend).sum())
        below_mean_pct = below_mean_count / n_customers * 100

        low_threshold_label = value_fmt(0) if negative_bin else value_fmt(bin_width)

        if value_kind == "currency":
            metric_labels = [
                "Customers",
                "Total revenue",
                "Median spend per customer",
                "Average spend per customer",
                "Min customer spend",
                "Max customer spend",
                f"Customers spending < {low_threshold_label}",
                f"Customers spending >= {value_fmt(threshold)}",
                "Customers spending less than the mean",
            ]
            values = [
                f"{n_customers:,}",
                value_fmt(total_revenue),
                value_fmt(median_spend),
                value_fmt(mean_spend),
                value_fmt(spend.min()),
                value_fmt(spend.max()),
                f"{low_bin_count:,} ({low_bin_pct:.1f}%)",
                f"{high_bin_count:,} ({high_bin_pct:.1f}%)",
                f"{below_mean_count:,} ({below_mean_pct:.1f}%)",
            ]
        else:
            label = xlabel.lower()
            metric_labels = [
                "Customers",
                f"Median {label} per customer",
                f"Average {label} per customer",
                f"Min {label} per customer",
                f"Max {label} per customer",
                f"Customers with {label} < {low_threshold_label}",
                f"Customers with {label} >= {value_fmt(threshold)}",
                f"Customers below average {label}",
            ]
            values = [
                f"{n_customers:,}",
                value_fmt(median_spend),
                value_fmt(mean_spend),
                value_fmt(spend.min()),
                value_fmt(spend.max()),
                f"{low_bin_count:,} ({low_bin_pct:.1f}%)",
                f"{high_bin_count:,} ({high_bin_pct:.1f}%)",
                f"{below_mean_count:,} ({below_mean_pct:.1f}%)",
            ]
            if value_kind == "number":
                # For plain counts (unlike ratios/percentages), a total
                # across customers is still a meaningful figure.
                metric_labels.insert(1, f"Total {label}")
                values.insert(1, value_fmt(total_revenue))

        summary_table = pd.DataFrame({"Metric": metric_labels, "Value": values})

        try:
            from IPython.display import display

            display(summary_table)
        except ImportError:
            print(summary_table.to_string(index=False))


def profit_decile_table(
    data: pd.DataFrame,
    *,
    profit_col: str = "Total_profit",
    spend_col: str = "Total_spend",
    trans_col: str = "Num_trans",
    by: str = "customers",
    currency: str = "$",
) -> pd.DataFrame:
    """Decile decomposition of profit (Fader/Hardie/Ross ``Customer-Base Audit``,
    Tables 3.3/3.4).

    Ranks customers by ``profit_col`` (descending) and splits them into 10
    groups, then reports, per group: share of customers, share of total
    profit, average spend/profit per customer, average order frequency
    (AOF, transactions per customer), average order value (AOV, revenue per
    transaction), and average margin (profit / revenue). All per-group
    metrics are computed from group *totals* (e.g. ``AOF = sum(trans) /
    sum(customers)``) rather than by averaging individual customer ratios,
    so that ``AOF \u00d7 AOV = average spend per customer`` and
    ``average spend per customer \u00d7 average margin = average profit per
    customer`` hold exactly (up to display rounding) — matching the book's
    decomposition identities.

    ``by`` picks how customers are grouped into deciles:
      - ``"customers"`` (Table 3.3): 10 equal-*sized* groups (~10% of
        customers each), ranked by profit — shows how profit share varies
        across equal-sized customer segments.
      - ``"profit"`` (Table 3.4): 10 equal-*profit* groups (~10% of total
        profit each) — shows how many customers (as a % of the base) it
        takes to generate each successive slice of profit, from the most to
        the least profitable.

    Returns a formatted (string-valued) ``DataFrame`` with a ``"Total"`` row,
    ready to display.
    """
    if by not in ("customers", "profit"):
        raise ValueError(f"by must be 'customers' or 'profit', got {by!r}")

    currency = _resolve_currency_symbol(currency)

    df = (
        data[[profit_col, spend_col, trans_col]]
        .dropna()
        .sort_values(profit_col, ascending=False)
        .reset_index(drop=True)
    )
    n = len(df)
    total_profit = df[profit_col].sum()

    if by == "customers":
        decile = np.arange(n) * 10 // n
    else:
        cum_frac = df[profit_col].cumsum() / total_profit
        decile = np.minimum(np.floor(cum_frac * 10).astype(int), 9)
        # Guard against a dip in cumulative profit (e.g. a run of
        # loss-making customers) pulling the decile index backwards.
        decile = np.maximum.accumulate(decile)
    df["_decile"] = np.minimum(decile, 9)

    rows = []
    for i in range(10):
        g = df.loc[df["_decile"] == i]
        n_cust = len(g)
        spend = g[spend_col].sum()
        profit = g[profit_col].sum()
        trans = g[trans_col].sum()
        rows.append(
            {
                "Decile": i + 1,
                "% Customers": n_cust / n if n else np.nan,
                "% Profit": profit / total_profit if total_profit else np.nan,
                "Avg spend per cust": spend / n_cust if n_cust else np.nan,
                "Avg profit per cust": profit / n_cust if n_cust else np.nan,
                "AOF": trans / n_cust if n_cust else np.nan,
                "AOV": spend / trans if trans else np.nan,
                "Avg Margin": profit / spend if spend else np.nan,
            }
        )

    total_spend = df[spend_col].sum()
    total_trans = df[trans_col].sum()
    rows.append(
        {
            "Decile": "Total",
            "% Customers": 1.0,
            "% Profit": 1.0,
            "Avg spend per cust": total_spend / n if n else np.nan,
            "Avg profit per cust": total_profit / n if n else np.nan,
            "AOF": total_trans / n if n else np.nan,
            "AOV": total_spend / total_trans if total_trans else np.nan,
            "Avg Margin": total_profit / total_spend if total_spend else np.nan,
        }
    )

    result = pd.DataFrame(rows)

    def _pct(x: float) -> str:
        if pd.isna(x):
            return "-"
        value = x * 100
        # Use just enough decimals so a small-but-nonzero share (e.g. the
        # top profit-decile needing well under 1% of customers) doesn't
        # get rounded away to a misleading "0%".
        decimals = 0
        while decimals < 2 and value != 0 and round(value, decimals) == 0:
            decimals += 1
        return f"{value:.{decimals}f}%"

    def _money(x: float) -> str:
        # Plain comma-grouped integer amount (no "K" suffix, no cents) —
        # unlike `_format_spend`, which is tuned for compact axis labels.
        return "-" if pd.isna(x) else f"{currency}{x:,.0f}"

    def _num(x: float) -> str:
        return "-" if pd.isna(x) else f"{x:.1f}"

    formatted = pd.DataFrame(
        {
            "Decile": result["Decile"],
            "% Customers": result["% Customers"].map(_pct),
            "% Profit": result["% Profit"].map(_pct),
            "Avg spend per cust": result["Avg spend per cust"].map(_money),
            "Avg profit per cust": result["Avg profit per cust"].map(_money),
            "AOF": result["AOF"].map(_num),
            "AOV": result["AOV"].map(_money),
            "Avg Margin": result["Avg Margin"].map(_pct),
        }
    )
    return formatted
