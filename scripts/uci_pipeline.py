"""Build enriched UCI Online Retail II modeling datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_store import build_uci_customer_features
from modeling_extensions import (
    build_nba_catalog,
    build_nba_events,
    build_uplift_campaigns,
    load_online_retail_ii,
)
from uci_context import (
    add_context_uci,
    build_uci_dim_customers,
    generate_synthetic_crm,
    load_eu_macro,
    load_eu_weather,
    order_weather_keys,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_MODELING = PROJECT_ROOT / "data" / "modeling"
DATA_EXTERNAL = DATA_RAW / "external"
MACRO_PATH = DATA_EXTERNAL / "eu_macro_by_country.csv"
WEATHER_PATH = DATA_EXTERNAL / "eu_weather_by_country.csv"

# False = fetch missing weather from Open-Meteo (~21 API calls, ~4 min first run).
# True  = use eu_weather_by_country.csv only (fast rebuild when cache exists).
CACHE_ONLY = False


def build_uci_spine(
    *,
    cache_only_weather: bool = CACHE_ONLY,
) -> dict[str, Path]:
    """Full UCI pipeline: transactions → context → CRM → features → uplift/NBA."""
    DATA_MODELING.mkdir(parents=True, exist_ok=True)

    retail = load_online_retail_ii(DATA_RAW, eu_only=True)
    date_start = retail["order_date"].min().normalize()
    date_end = retail["order_date"].max().normalize()
    month_start = (date_start - pd.DateOffset(months=12)).strftime("%Y-%m")
    month_end = date_end.to_period("M").strftime("%Y-%m")

    countries = retail["country"].dropna().unique()
    from uci_context import country_to_geo

    geos = sorted({country_to_geo(c) for c in countries if country_to_geo(c)})

    macro = load_eu_macro(geos, month_start, month_end, cache_path=MACRO_PATH)
    order_keys = order_weather_keys(retail, str(date_start.date()), str(date_end.date()))
    weather = load_eu_weather(
        order_keys,
        WEATHER_PATH,
        cache_only=cache_only_weather,
        verbose=True,
    )

    df = add_context_uci(retail, macro, weather)
    df = df.reset_index(drop=True)
    df["transaction_id"] = df.index.map(lambda i: f"uci_{i:07d}")

    customer_ids = df["customer_id"].drop_duplicates().sort_values()
    crm = generate_synthetic_crm(customer_ids, seed=42)
    fact = df.merge(crm, on="customer_id", how="left")

    dim = build_uci_dim_customers(fact)
    features = build_uci_customer_features(fact, dim)
    uplift = build_uplift_campaigns(fact, features)
    nba_catalog = build_nba_catalog()
    nba_events = build_nba_events(fact, features)

    outputs = {
        "uci_fact_transactions": DATA_MODELING / "uci_fact_transactions.parquet",
        "uci_dim_customers": DATA_MODELING / "uci_dim_customers.parquet",
        "uci_customer_features": DATA_MODELING / "uci_customer_features.parquet",
        "uci_uplift_campaigns": DATA_MODELING / "uci_uplift_campaigns.parquet",
        "uci_nba_offer_catalog": DATA_MODELING / "uci_nba_offer_catalog.parquet",
        "uci_nba_offer_events": DATA_MODELING / "uci_nba_offer_events.parquet",
    }
    fact.to_parquet(outputs["uci_fact_transactions"], index=False)
    dim.to_parquet(outputs["uci_dim_customers"], index=False)
    features.to_parquet(outputs["uci_customer_features"], index=False)
    uplift.to_parquet(outputs["uci_uplift_campaigns"], index=False)
    nba_catalog.to_parquet(outputs["uci_nba_offer_catalog"], index=False)
    nba_events.to_parquet(outputs["uci_nba_offer_events"], index=False)
    return outputs


if __name__ == "__main__":
    for name, path in build_uci_spine().items():
        df = pd.read_parquet(path)
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}  ({len(df):,} rows)")
