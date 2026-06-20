"""Extended modeling datasets: Instacart depth, CLV universe, uplift, NBA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ONLINE_RETAIL_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
)
ONLINE_RETAIL_II_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"
)

NBA_OFFERS = pd.DataFrame([
    {"offer_id": "NBA-01", "offer_type": "category_cross_sell", "channel": "email", "target_category": "Technology", "discount_pct": 0.10},
    {"offer_id": "NBA-02", "offer_type": "category_cross_sell", "channel": "app", "target_category": "Furniture", "discount_pct": 0.15},
    {"offer_id": "NBA-03", "offer_type": "category_cross_sell", "channel": "email", "target_category": "Office Supplies", "discount_pct": 0.05},
    {"offer_id": "NBA-04", "offer_type": "tier_upgrade", "channel": "app", "target_category": None, "discount_pct": 0.0},
    {"offer_id": "NBA-05", "offer_type": "points_bonus", "channel": "email", "target_category": None, "discount_pct": 0.0},
    {"offer_id": "NBA-06", "offer_type": "win_back", "channel": "email", "target_category": None, "discount_pct": 0.20},
    {"offer_id": "NBA-07", "offer_type": "free_shipping", "channel": "app", "target_category": None, "discount_pct": 0.0},
    {"offer_id": "NBA-08", "offer_type": "bundle", "channel": "email", "target_category": "Technology", "discount_pct": 0.12},
])


def build_instacart_orders(raw_dir: Path) -> pd.DataFrame | None:
    path = raw_dir / "orders.csv"
    if not path.exists():
        return None
    orders = pd.read_csv(path)
    orders = orders.sort_values(["user_id", "order_number"])
    orders["days_since_first"] = (
        orders.groupby("user_id")["days_since_prior_order"].cumsum().fillna(0)
    )
    # Instacart raw has no calendar date — reconstruct a pseudo timeline for sequencing
    orders["order_date"] = pd.Timestamp("2015-01-01") + pd.to_timedelta(
        orders["days_since_first"], unit="D"
    )
    return orders


def build_instacart_baskets(raw_dir: Path) -> pd.DataFrame | None:
    """Order-line grain with product/department — for basket & reorder models."""
    prior = raw_dir / "order_products__prior.csv"
    train = raw_dir / "order_products__train.csv"
    products_path = raw_dir / "products.csv"
    if not all(p.exists() for p in (prior, train, products_path)):
        return None

    lines = pd.concat([pd.read_csv(prior), pd.read_csv(train)], ignore_index=True)
    products = pd.read_csv(products_path)[["product_id", "product_name", "aisle_id", "department_id"]]
    if (raw_dir / "departments.csv").exists():
        depts = pd.read_csv(raw_dir / "departments.csv")
        products = products.merge(depts, on="department_id", how="left")

    baskets = lines.merge(products, on="product_id", how="left")
    return baskets


def _clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={
        "InvoiceNo": "order_id",
        "Invoice": "order_id",
        "StockCode": "product_id",
        "Description": "product_name",
        "Quantity": "quantity",
        "InvoiceDate": "order_date",
        "UnitPrice": "unit_price",
        "Price": "unit_price",
        "CustomerID": "customer_id",
        "Customer ID": "customer_id",
        "Country": "country",
    })
    out["order_date"] = pd.to_datetime(out["order_date"])
    out["customer_id"] = out["customer_id"].astype("Int64").astype(str)
    out["product_id"] = out["product_id"].astype(str)
    out = out[
        out["customer_id"].notna() & (out["quantity"] > 0) & (out["unit_price"] > 0)
    ].copy()
    out["line_total"] = out["quantity"] * out["unit_price"]
    out["order_id"] = out["order_id"].astype(str)
    return out


def load_online_retail(raw_dir: Path) -> pd.DataFrame:
    """UCI Online Retail I — ~400k UK transactions, better CLV sample size."""
    cache = raw_dir / "online_retail" / "transactions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    xlsx = raw_dir / "online_retail" / "Online Retail.xlsx"
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    if not xlsx.exists():
        import requests
        r = requests.get(ONLINE_RETAIL_URL, timeout=120)
        r.raise_for_status()
        xlsx.write_bytes(r.content)

    df = _clean_online_retail(pd.read_excel(xlsx, engine="openpyxl"))
    df.to_parquet(cache, index=False)
    return df


def load_online_retail_ii(raw_dir: Path, *, eu_only: bool = True) -> pd.DataFrame:
    """UCI Online Retail II — ~800k lines; eu_only keeps Eurostat-mapped countries (~799k)."""
    subdir = raw_dir / "online_retail_ii"
    cache = subdir / ("eu_transactions.parquet" if eu_only else "transactions.parquet")
    if cache.exists():
        return pd.read_parquet(cache)

    xlsx = subdir / "online_retail_II.xlsx"
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    if not xlsx.exists():
        import requests
        r = requests.get(ONLINE_RETAIL_II_URL, timeout=180)
        r.raise_for_status()
        xlsx.write_bytes(r.content)

    sheets = pd.ExcelFile(xlsx, engine="openpyxl").sheet_names
    frames = [pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl") for sheet in sheets]
    df = _clean_online_retail(pd.concat(frames, ignore_index=True))
    if eu_only:
        from uci_context import COUNTRY_META
        df = df[df["country"].isin(COUNTRY_META)].copy()
    df.to_parquet(cache, index=False)
    return df


def build_online_retail_customers(transactions: pd.DataFrame) -> pd.DataFrame:
    as_of = transactions["order_date"].max()
    dim = transactions.groupby("customer_id", as_index=False).agg(
        country=("country", "first"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        total_orders=("order_id", "nunique"),
        total_lines=("order_id", "count"),
        total_revenue=("line_total", "sum"),
        avg_unit_price=("unit_price", "mean"),
    )
    dim["avg_order_value"] = dim["total_revenue"] / dim["total_orders"].clip(lower=1)
    dim["recency_days"] = (as_of - dim["last_order_date"]).dt.days
    dim["tenure_days"] = (dim["last_order_date"] - dim["first_order_date"]).dt.days
    return dim


def build_uplift_campaigns(fact: pd.DataFrame, features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic RCT-style uplift dataset (seeded, reproducible).

    Each customer appears in 2 waves with randomized discount offers.
    Response probability scales with discount_sensitivity; outcome = 30d revenue lift.
    """
    rng = np.random.default_rng(seed)
    as_of = fact["order_date"].max()
    segment_col = "segment" if "segment" in features.columns else "rfm_segment"
    cust = features[
        ["customer_id", segment_col, "tier", "discount_sensitivity", "engagement_score", "avg_order_value"]
    ].copy().rename(columns={segment_col: "segment"})

    waves = []
    for wave_id in (1, 2):
        wave = cust.copy()
        wave["campaign_id"] = f"UPLIFT-W{wave_id}"
        wave["treatment"] = rng.binomial(1, 0.5, len(wave))
        wave["offer_discount_pct"] = np.where(wave["treatment"] == 1, rng.choice([0.10, 0.15, 0.20], len(wave)), 0.0)

        logit = (
            -1.2
            + 2.5 * wave["discount_sensitivity"] * wave["treatment"]
            + 1.0 * wave["engagement_score"]
            + 0.3 * wave["offer_discount_pct"] * 10
        )
        prob = 1 / (1 + np.exp(-logit))
        wave["responded"] = rng.binomial(1, prob.clip(0.05, 0.95))

        base_rev = wave["avg_order_value"].fillna(wave["avg_order_value"].median())
        lift = wave["responded"] * wave["treatment"] * base_rev * (
            0.15 + 0.35 * wave["discount_sensitivity"]
        )
        noise = rng.normal(0, base_rev * 0.05, len(wave))
        wave["revenue_30d"] = base_rev + lift + noise
        wave["incremental_revenue"] = np.where(wave["treatment"] == 1, lift, 0.0)
        wave["campaign_date"] = as_of - pd.Timedelta(days=30 * (3 - wave_id))
        waves.append(wave)

    out = pd.concat(waves, ignore_index=True)
    out["uplift_label"] = out["incremental_revenue"] > 0
    return out


def build_nba_catalog() -> pd.DataFrame:
    return NBA_OFFERS.copy()


def build_nba_events(fact: pd.DataFrame, features: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Simulated next-best-action log: offer shown -> click -> convert.

    Offer match uses top customer category affinity; conversion uses RFM + engagement.
    """
    rng = np.random.default_rng(seed)
    as_of = fact["order_date"].max()

    if "category" in fact.columns:
        affinity = (
            fact.groupby(["customer_id", "category"])["sales"]
            .sum()
            .reset_index()
            .sort_values(["customer_id", "sales"], ascending=[True, False])
            .drop_duplicates("customer_id")
            .rename(columns={"category": "top_category"})
        )
    else:
        value_col = "line_total" if "line_total" in fact.columns else "sales"
        fact_aff = fact.copy()
        fact_aff["product_family"] = (
            fact_aff["product_name"].astype(str).str.split().str[0].str.upper()
        )
        affinity = (
            fact_aff.groupby(["customer_id", "product_family"])[value_col]
            .sum()
            .reset_index()
            .sort_values(["customer_id", value_col], ascending=[True, False])
            .drop_duplicates("customer_id")
            .rename(columns={"product_family": "top_category"})
        )
    cust = features.merge(affinity, on="customer_id", how="left")

    events = []
    for _, row in cust.iterrows():
        eligible = NBA_OFFERS.copy()
        # Prefer cross-sell offers aligned to non-top categories or win-back if stale
        if row["recency_days"] >= 120:
            weights = np.where(eligible["offer_type"] == "win_back", 3.0, 1.0)
        else:
            weights = np.where(
                eligible["target_category"].eq(row.get("top_category")), 0.3, 1.0
            )
        weights = weights / weights.sum()
        offer = eligible.iloc[rng.choice(len(eligible), p=weights)]

        show_prob = 0.55 + 0.35 * row["engagement_score"]
        shown = rng.random() < show_prob
        click_prob = 0.08 + 0.25 * row["engagement_score"] if shown else 0.0
        clicked = rng.random() < click_prob
        conv_logit = (
            -2.0
            + 0.4 * row["RFM_score"]
            + 1.5 * row["discount_sensitivity"] * offer["discount_pct"] * 10
            + (0.8 if clicked else -1.5)
        )
        converted = rng.random() < (1 / (1 + np.exp(-conv_logit)))

        events.append({
            "customer_id": row["customer_id"],
            "rfm_segment": row["rfm_segment"],
            "top_category": row.get("top_category"),
            "offer_id": offer["offer_id"],
            "offer_type": offer["offer_type"],
            "channel": offer["channel"],
            "discount_pct": offer["discount_pct"],
            "shown": int(shown),
            "clicked": int(clicked and shown),
            "converted": int(converted and shown),
            "event_date": as_of,
        })

    return pd.DataFrame(events)
