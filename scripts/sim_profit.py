"""Simulated transaction-line profit (COGS / payment fees / fulfillment / shipping).

UCI Online Retail II has no cost data — only revenue (``quantity × unit_price``).
This module layers a *simulated* profit on top of the existing revenue-only fact
table so exploratory notebooks can look at profit-weighted views without
inventing a single flat margin % (``profit = revenue × 30%``).

Every simulated rate is driven by a **deterministic MD5 hash** of a stable id
(``product_id`` for COGS, ``order_id``/``country`` for the per-order fees below)
— never ``np.random`` / a seed. The same product always gets the same simulated
COGS rate no matter how many times a notebook is rerun or how the data is
sliced/filtered.

Not wired into ``customer_base_audit.py``, ``clv_weekly.py`` or
``clv_enterprise.py`` — those stay revenue-only per ``MODELING.md``. Import this
module separately wherever a ``sim_profit`` view is wanted (currently
``Notebooks/Customer-base_audit-lenses.ipynb``).

Scope note: returns/cancellations are **not** modeled here. The cached
``uci_fact_transactions.parquet`` already has cancelled / negative-quantity
invoices dropped during cleaning (``modeling_extensions.py::_clean_online_retail``
filters ``quantity > 0``), so an empirical "invoices starting with C" rate isn't
derivable without re-downloading the raw UCI file.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tunable assumptions — all easy to eyeball/adjust in one place.
# ---------------------------------------------------------------------------

# COGS: simulated per product_id, uniformly spread across this range.
COGS_RATE_MIN = 0.30
COGS_RATE_MAX = 0.60

# Payment processing: card-processor-style flat + percentage, per order.
PAYMENT_FEE_PCT = 0.029
PAYMENT_FEE_FIXED = 0.30

# Fulfillment: pick/pack style base + per-item, per order. International
# orders (non-domestic) cost more to fulfill.
FULFILLMENT_BASE = 1.50
FULFILLMENT_PER_ITEM = 0.20
FULFILLMENT_INTL_MULTIPLIER = 1.5

# Shipping: base handling + per-item, tiered by geography. UCI is already
# EU-filtered (see modeling_extensions.EU_COUNTRIES), so tiers are
# domestic (UK) / near neighbors / rest of the EU set — not domestic vs.
# overseas as a non-EU-filtered retailer might see.
DOMESTIC_COUNTRY = "United Kingdom"
NEAR_COUNTRIES = frozenset({"EIRE", "Netherlands", "Belgium"})
SHIPPING_TIERS = {
    "domestic": {"base": 1.00, "per_item": 0.15},
    "near": {"base": 2.00, "per_item": 0.25},
    "far": {"base": 3.50, "per_item": 0.35},
}

SIM_COST_COLS = (
    "sim_cogs",
    "sim_payment_fee",
    "sim_fulfillment_cost",
    "sim_shipping_cost",
)


def _hash_unit(ids: pd.Series) -> pd.Series:
    """Deterministic float in ``[0, 1)`` per distinct id value.

    MD5-based (not ``hash()``/RNG), so it's stable across processes, runs and
    ``PYTHONHASHSEED`` — the same id always maps to the same float.
    """
    uniq = ids.astype(str).unique()
    lut = {
        v: (int(hashlib.md5(v.encode()).hexdigest()[:8], 16) % 10_000) / 10_000.0
        for v in uniq
    }
    return ids.astype(str).map(lut)


def _shipping_tier(country: pd.Series) -> pd.Series:
    is_domestic = country.eq(DOMESTIC_COUNTRY)
    is_near = country.isin(NEAR_COUNTRIES)
    tier = np.where(is_domestic, "domestic", np.where(is_near, "near", "far"))
    return pd.Series(tier, index=country.index)


def simulate_line_profit(fact: pd.DataFrame, *, revenue_col: str = "line_total") -> pd.DataFrame:
    """Add simulated ``sim_*`` cost + profit columns to a copy of ``fact``.

    Expects line grain with at least ``order_id``, ``product_id``,
    ``quantity`` and ``revenue_col`` (``line_total`` by default). ``country``
    is optional — if present it drives fulfillment/shipping tiers, otherwise
    every order is treated as domestic.

    Fixed per-order costs (payment's ``$0.30``, fulfillment's ``$1.50`` base,
    shipping's base handling) are computed once per ``order_id`` and then
    allocated back down to line grain (by revenue share for payment fees, by
    quantity share for fulfillment/shipping), so line-level sims always sum
    exactly to the order-level formula.

    New columns: ``sim_cogs_rate``, ``sim_cogs``, ``sim_payment_fee``,
    ``sim_fulfillment_cost``, ``sim_shipping_cost``, ``sim_total_cost``,
    ``sim_profit``.
    """
    required = {"order_id", "product_id", "quantity", revenue_col}
    missing = required - set(fact.columns)
    if missing:
        raise ValueError(f"simulate_line_profit missing columns: {sorted(missing)}")

    df = fact.copy()
    revenue = df[revenue_col].astype(float)
    quantity = df["quantity"].astype(float)

    # 1) COGS — deterministic per product_id, spread across [MIN, MAX].
    df["sim_cogs_rate"] = COGS_RATE_MIN + (COGS_RATE_MAX - COGS_RATE_MIN) * _hash_unit(
        df["product_id"]
    )
    df["sim_cogs"] = revenue * df["sim_cogs_rate"]

    # Order-level aggregates used to allocate per-order fees back to lines.
    order_revenue = df.groupby("order_id")[revenue_col].transform("sum").astype(float)
    order_qty = df.groupby("order_id")["quantity"].transform("sum").astype(float)
    revenue_share = (revenue / order_revenue.replace(0, np.nan)).fillna(0.0)
    qty_share = (quantity / order_qty.replace(0, np.nan)).fillna(0.0)

    if "country" in df.columns:
        order_country = df.groupby("order_id")["country"].transform("first")
    else:
        order_country = pd.Series(DOMESTIC_COUNTRY, index=df.index)

    # 2) Payment processing — pct + fixed per order, split by revenue share.
    order_payment_fee = order_revenue * PAYMENT_FEE_PCT + PAYMENT_FEE_FIXED
    df["sim_payment_fee"] = order_payment_fee * revenue_share

    # 3) Fulfillment — base + per-item per order, ×multiplier if non-domestic,
    #    split by quantity share.
    is_intl = order_country.ne(DOMESTIC_COUNTRY)
    order_fulfillment = (FULFILLMENT_BASE + FULFILLMENT_PER_ITEM * order_qty) * np.where(
        is_intl, FULFILLMENT_INTL_MULTIPLIER, 1.0
    )
    df["sim_fulfillment_cost"] = order_fulfillment * qty_share

    # 4) Shipping — base handling + per-item per order, tiered by geography,
    #    split by quantity share.
    tier = _shipping_tier(order_country)
    tier_base = tier.map(lambda t: SHIPPING_TIERS[t]["base"])
    tier_per_item = tier.map(lambda t: SHIPPING_TIERS[t]["per_item"])
    order_shipping = tier_base + tier_per_item * order_qty
    df["sim_shipping_cost"] = order_shipping * qty_share

    df["sim_total_cost"] = df[list(SIM_COST_COLS)].sum(axis=1)
    df["sim_profit"] = revenue - df["sim_total_cost"]
    return df


def customer_profit_summary(
    df: pd.DataFrame,
    *,
    revenue_col: str = "line_total",
    group_cols: tuple[str, ...] = ("customer_id", "year", "cohort"),
    freq: str = "Y",
) -> pd.DataFrame:
    """Aggregate simulated profit to customer × period.

    If ``group_cols`` (default ``customer_id``/``year``/``cohort``) are
    already present on ``df`` — e.g. the ``year``/``cohort`` columns the
    audit-lenses notebook derives from ``order_date`` — they're used as-is.
    Otherwise falls back to deriving a single ``period`` column from
    ``order_date`` via ``freq``.

    Runs ``simulate_line_profit`` first if ``sim_profit`` isn't already on
    ``df``.

    Returns one row per group with ``Total_spend``, ``Num_trans``,
    ``Total_profit`` and ``Profit_margin``.
    """
    if "sim_profit" not in df.columns:
        df = simulate_line_profit(df, revenue_col=revenue_col)

    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        if "order_date" not in df.columns:
            raise ValueError(
                "customer_profit_summary needs group_cols present on df, or an "
                "order_date column to derive a period from"
            )
        df = df.copy()
        df["period"] = df["order_date"].dt.to_period(freq).astype(str)
        cols = ["customer_id", "period"]
    elif "customer_id" not in cols:
        cols = ["customer_id", *cols]

    g = df.groupby(cols, as_index=False).agg(
        Total_spend=(revenue_col, "sum"),
        Num_trans=("order_id", "nunique"),
        Total_profit=("sim_profit", "sum"),
    )
    g["Profit_margin"] = np.where(
        g["Total_spend"] > 0, g["Total_profit"] / g["Total_spend"], np.nan
    )
    return g
