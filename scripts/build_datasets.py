"""Download raw USA data and build final modeling datasets.

Orchestrates the same pipeline as Notebooks/data_prep.ipynb:
  Steps 1–5  Superstore spine (inline in this file)
  Steps 6–8  scripts/feature_store.py + scripts/modeling_extensions.py
"""

from __future__ import annotations

import io
import subprocess
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from feature_store import build_customer_features
from modeling_extensions import (
    build_instacart_baskets,
    build_instacart_orders,
    build_nba_catalog,
    build_nba_events,
    build_online_retail_customers,
    build_uplift_campaigns,
    load_online_retail_ii,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_MODELING = PROJECT_ROOT / "data" / "modeling"

US_SUPERSTORE_URL = (
    "https://raw.githubusercontent.com/ThigasSantos/BaseSuperStore/main/sales.csv"
)
US_TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
CONTEXT_MONTH_START = "2016-01"
CONTEXT_MONTH_END = "2019-12"
CONTEXT_DATE_START = "2016-01-01"
CONTEXT_DATE_END = "2019-12-31"
MACRO_FETCH_MONTH_START = "2015-01"
MACRO_FETCH_DATE_START = "2015-01-01"

US_MACRO_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=CPIAUCSL,UNRATE,FEDFUNDS"
    f"&cosd={MACRO_FETCH_DATE_START}&coed={CONTEXT_DATE_END}"
)
MACRO_COLUMNS = [
    "month",
    "cpi_index",
    "inflation_mom",
    "inflation_yoy",
    "unemployment_rate",
    "interest_rate",
]
INSTACART_DATASET = "yasserh/instacart-online-grocery-basket-analysis-dataset"
INSTACART_FILES = [
    "orders.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
]

WEATHER_CACHE_PATH = DATA_RAW / "external" / "us_weather_by_city.csv"
# Production builds read the CSV cache only — never call Open-Meteo (too slow / rate-limited).
# Populate the cache once via Notebooks/data_prep.ipynb Step 2b, or set WEATHER_CACHE_ONLY=False.
WEATHER_CACHE_ONLY = True

WEATHER_DAILY_VARS = (
    "temperature_2m_mean,rain_sum,snowfall_sum,precipitation_sum,"
    "weather_code,wind_gusts_10m_max"
)
STORM_WEATHER_CODES = frozenset({65, 67, 75, 77, 82, 85, 86, 95, 96, 99})
MAJOR_STORM_WIND_KMH = 75.0
OPEN_METEO_REQUEST_DELAY_SEC = 10
WEATHER_DATE_CHUNK = 100


def _weather_cache_coverage(weather: pd.DataFrame, order_keys: pd.DataFrame) -> float:
    if weather.empty or order_keys.empty:
        return 0.0
    keys = order_keys.copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
    have = weather[["date", "city", "state"]].drop_duplicates()
    have["date"] = pd.to_datetime(have["date"]).dt.normalize()
    matched = keys.merge(have, on=["date", "city", "state"], how="inner")
    return len(matched) / len(keys)


class OpenMeteoQuotaExceeded(Exception):
    """Open-Meteo rate or daily quota limit hit."""


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f, tqdm(total=int(r.headers.get("content-length", 0)) or None, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def load_superstore() -> pd.DataFrame:
    path = DATA_RAW / "us_superstore" / "superstore_sales.csv"
    if not path.exists():
        raw = _download(US_SUPERSTORE_URL, DATA_RAW / "us_superstore" / "raw.csv")
        df = pd.read_csv(raw)
        df = df.rename(columns={
            "OrderID": "order_id", "OrderDate": "order_date", "ShipDate": "ship_date",
            "ShipMode": "ship_mode", "CustomerID": "customer_id", "CustomerName": "customer_name",
            "Segment": "segment", "Country": "country", "City": "city", "State": "state",
            "Postal Code": "postal_code", "Region": "region", "ProductID": "product_id",
            "Category": "category", "Sub-Category": "sub_category", "ProductName": "product_name",
            "Sales": "sales", "Quantity": "quantity", "Discount": "discount", "Profit": "profit",
        })
        df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=True)
        df["ship_date"] = pd.to_datetime(df["ship_date"], dayfirst=True)
        df = df[df["country"] == "United States"].copy()
        df.to_csv(path, index=False)
        raw.unlink(missing_ok=True)
    return pd.read_csv(path, parse_dates=["order_date", "ship_date"])


def _macro_needs_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    macro = pd.read_csv(path)
    if macro.empty:
        return True
    if not set(MACRO_COLUMNS).issubset(macro.columns):
        return True
    return (
        macro["month"].min() > CONTEXT_MONTH_START
        or macro["month"].max() < CONTEXT_MONTH_END
    )


def _build_macro_frame(raw: pd.DataFrame) -> pd.DataFrame:
    macro = raw.rename(columns={
        "observation_date": "month",
        "CPIAUCSL": "cpi_index",
        "UNRATE": "unemployment_rate",
        "FEDFUNDS": "interest_rate",
    })
    macro["month"] = pd.to_datetime(macro["month"]).dt.to_period("M").astype(str)
    for col in ("cpi_index", "unemployment_rate", "interest_rate"):
        macro[col] = pd.to_numeric(macro[col], errors="coerce")
    macro = macro.dropna(subset=["cpi_index"])
    macro["inflation_mom"] = macro["cpi_index"].pct_change().round(5)
    if macro["month"].min() <= MACRO_FETCH_MONTH_START:
        macro["inflation_yoy"] = macro["cpi_index"].pct_change(12).round(5)
    return macro[
        (macro["month"] >= CONTEXT_MONTH_START) & (macro["month"] <= CONTEXT_MONTH_END)
    ].reset_index(drop=True).copy()


def load_macro() -> pd.DataFrame:
    path = DATA_RAW / "external" / "us_macro_monthly.csv"
    if not _macro_needs_refresh(path):
        macro = pd.read_csv(path)
    else:
        try:
            r = requests.get(US_MACRO_URL, timeout=90)
            r.raise_for_status()
            macro = _build_macro_frame(pd.read_csv(io.StringIO(r.text)))
        except (requests.RequestException, ValueError, KeyError):
            months = pd.period_range(CONTEXT_MONTH_START, CONTEXT_MONTH_END, freq="M").astype(str)
            macro = pd.DataFrame({
                "month": months,
                "cpi_index": 240.0,
                "unemployment_rate": np.nan,
                "interest_rate": np.nan,
            })
            macro["inflation_mom"] = macro["cpi_index"].pct_change().round(5)
        path.parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(path, index=False)
    return macro[[c for c in MACRO_COLUMNS if c in macro.columns]].copy()


def _geocode_city(city: str, state: str) -> dict | None:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 10, "country": "US"},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results") or []
    match = next(
        (x for x in results if x.get("admin1", "").lower() == state.lower()),
        None,
    )
    if match is None and results:
        match = results[0]
    if match is None:
        return None
    return {
        "city": city,
        "state": state,
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "timezone": match["timezone"],
    }


def load_city_coordinates(cities: pd.DataFrame) -> pd.DataFrame:
    path = DATA_RAW / "external" / "city_coordinates.csv"
    expected = cities[["city", "state"]].drop_duplicates()
    if path.exists():
        coords = pd.read_csv(path)
        have = coords[["city", "state"]].drop_duplicates()
        if len(have) >= len(expected):
            return coords

    rows = []
    for city, state in tqdm(
        expected.itertuples(index=False),
        total=len(expected),
        desc="Geocoding cities",
    ):
        row = _geocode_city(city, state)
        if row is not None:
            rows.append(row)
    coords = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords.to_csv(path, index=False)
    return coords


def _add_weather_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.rename(columns={
        "time": "date",
        "temperature_2m_mean": "temp_c",
        "rain_sum": "rain_mm",
        "snowfall_sum": "snow_cm",
        "precipitation_sum": "precip_mm",
        "wind_gusts_10m_max": "wind_gust_kmh",
    })
    out["had_rain"] = out["rain_mm"].fillna(0) > 0
    out["had_snow"] = out["snow_cm"].fillna(0) > 0
    out["had_major_storm"] = (
        out["weather_code"].isin(STORM_WEATHER_CODES)
        | (out["wind_gust_kmh"].fillna(0) >= MAJOR_STORM_WIND_KMH)
    )
    return out


def _order_weather_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["order_date"].dt.normalize()
    in_range = out["order_date"].between(CONTEXT_DATE_START, CONTEXT_DATE_END)
    return out.loc[in_range, ["date", "city", "state"]].drop_duplicates()


def _trim_weather_cache(weather: pd.DataFrame, order_keys: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return weather
    keys = order_keys.copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    return weather.merge(keys, on=["date", "city", "state"], how="inner")


def _city_weather_complete(
    weather: pd.DataFrame, order_keys: pd.DataFrame, city: str, state: str
) -> bool:
    needed = order_keys.loc[
        (order_keys["city"] == city) & (order_keys["state"] == state), "date"
    ].dt.normalize()
    if needed.empty:
        return True
    have = weather.loc[
        (weather["city"] == city) & (weather["state"] == state), "date"
    ].dt.normalize()
    return set(needed) <= set(have)


def _open_meteo_429_reason(response: requests.Response) -> str:
    try:
        return str(response.json().get("reason", ""))
    except ValueError:
        return response.text[:200]


def _open_meteo_daily_frame(payload: dict | list) -> pd.DataFrame:
    # Comma-separated dates return one response object per date (a list).
    if isinstance(payload, list):
        parts = [pd.DataFrame(item["daily"]) for item in payload]
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(payload["daily"])


def _fetch_weather_archive(
    latitude: float, longitude: float, timezone: str, dates: list[str]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for i in range(0, len(dates), WEATHER_DATE_CHUNK):
        chunk = dates[i : i + WEATHER_DATE_CHUNK]
        date_str = ",".join(chunk)
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date_str,
            "end_date": date_str,
            "daily": WEATHER_DAILY_VARS,
            "timezone": timezone,
        }
        attempt = 0
        while True:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=120,
            )
            if r.status_code == 429:
                reason = _open_meteo_429_reason(r)
                if "daily" in reason.lower():
                    raise OpenMeteoQuotaExceeded(
                        reason or "Daily API request limit exceeded. Re-run tomorrow."
                    )
                wait = int(r.headers.get("Retry-After", 0)) or min(60 * (2 ** attempt), 600)
                attempt += 1
                if attempt >= 12:
                    raise OpenMeteoQuotaExceeded(
                        reason or f"Rate limited after {attempt} retries."
                    )
                time.sleep(wait)
                continue
            r.raise_for_status()
            frames.append(_open_meteo_daily_frame(r.json()))
            break
        if i + WEATHER_DATE_CHUNK < len(dates):
            time.sleep(OPEN_METEO_REQUEST_DELAY_SEC)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_weather(
    order_keys: pd.DataFrame | None = None,
    *,
    cache_only: bool = WEATHER_CACHE_ONLY,
) -> pd.DataFrame:
    """Load weather for order dates. Default: read local CSV cache only (no API calls)."""
    path = WEATHER_CACHE_PATH
    if order_keys is None:
        order_keys = _order_weather_keys(load_superstore())

    if cache_only:
        if not path.exists():
            tqdm.write(
                f"Weather: no cache at {path.relative_to(PROJECT_ROOT)} — "
                "weather columns will be null. Run Notebooks/data_prep.ipynb Step 2b once "
                "to build the cache."
            )
            return pd.DataFrame()
        weather = pd.read_csv(path, parse_dates=["date"])
        weather = _trim_weather_cache(weather, order_keys)
        coverage = _weather_cache_coverage(weather, order_keys)
        tqdm.write(
            f"Weather: {len(weather):,} cached rows loaded "
            f"({coverage:.1%} order-key coverage, offline mode)"
        )
        return weather

    # Online fetch path (cache_only=False) — for initial cache population only
    weather = (
        pd.read_csv(path, parse_dates=["date"])
        if path.exists()
        else pd.DataFrame()
    )
    trimmed = _trim_weather_cache(weather, order_keys)
    if len(trimmed) < len(weather):
        path.parent.mkdir(parents=True, exist_ok=True)
        trimmed.to_csv(path, index=False)
    weather = trimmed

    cities = order_keys[["city", "state"]].drop_duplicates()
    coords = load_city_coordinates(cities)
    dates_by_city = (
        order_keys.groupby(["city", "state"])["date"]
        .apply(lambda s: sorted(pd.to_datetime(s).dt.strftime("%Y-%m-%d").unique()))
        .to_dict()
    )
    pending = [
        row
        for row in coords.itertuples(index=False)
        if not _city_weather_complete(weather, order_keys, row.city, row.state)
    ]
    if not pending:
        return weather

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for i, row in enumerate(tqdm(pending, desc="Weather by city"), start=1):
            dates = dates_by_city[(row.city, row.state)]
            frame = _add_weather_flags(
                _fetch_weather_archive(row.latitude, row.longitude, row.timezone, dates)
            )
            frame["city"] = row.city
            frame["state"] = row.state
            weather = pd.concat([weather, frame], ignore_index=True)
            weather["date"] = pd.to_datetime(weather["date"])
            weather = _trim_weather_cache(weather, order_keys)
            weather.to_csv(path, index=False)
            if i < len(pending):
                time.sleep(OPEN_METEO_REQUEST_DELAY_SEC)
    except OpenMeteoQuotaExceeded as exc:
        tqdm.write(f"Weather download paused: {exc}")
        tqdm.write("Progress saved — re-run later to resume.")

    return weather


def download_instacart() -> bool:
    target = DATA_RAW / "instacart"
    target.mkdir(parents=True, exist_ok=True)
    if all((target / f).exists() for f in INSTACART_FILES):
        return True
    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        return False
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", INSTACART_DATASET, "-p", str(target), "--unzip"],
        check=True,
    )
    return all((target / f).exists() for f in INSTACART_FILES)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth weekday (Mon=0 … Sun=6) in a calendar month."""
    first = date(year, month, 1)
    day = 1 + (weekday - first.weekday()) % 7 + 7 * (n - 1)
    return date(year, month, day)


def _retail_spending_days(years: range) -> set[date]:
    days: set[date] = set()
    for year in years:
        days.add(date(year, 2, 14))  # Valentine's Day
        days.add(date(year, 12, 24))  # Christmas Eve
        days.add(_nth_weekday_of_month(year, 5, 6, 2))  # Mother's Day
        days.add(_nth_weekday_of_month(year, 11, 3, 4) + timedelta(days=1))  # Black Friday
        feb1 = date(year, 2, 1)
        days.add(feb1 + timedelta(days=(6 - feb1.weekday()) % 7))  # Super Bowl Sunday
    return days


def add_context(df: pd.DataFrame, macro: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    years = range(2015, 2020)
    us_holidays = holidays.country_holidays("US", years=years)
    retail_days = _retail_spending_days(years)
    out = df.copy()
    out["date"] = out["order_date"].dt.normalize()
    out["month"] = out["order_date"].dt.to_period("M").astype(str)
    out["is_us_federal_holiday"] = out["date"].dt.date.map(lambda d: d in us_holidays)
    out["is_retail_spending_day"] = out["date"].dt.date.isin(retail_days)
    out["day_of_week"] = out["order_date"].dt.dayofweek
    out["is_friday"] = out["day_of_week"] == 4
    out["is_month_start"] = out["order_date"].dt.day <= 5
    out["is_month_end"] = out["order_date"].dt.day >= 25
    out = out.merge(macro[[c for c in MACRO_COLUMNS if c in macro.columns]], on="month", how="left")
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather_cols = [
        "date", "city", "state", "temp_c", "rain_mm", "snow_cm", "precip_mm",
        "weather_code", "wind_gust_kmh", "had_rain", "had_snow", "had_major_storm",
    ]
    return out.merge(weather[weather_cols], on=["date", "city", "state"], how="left")


def _generate_synthetic_crm(customer_ids: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate loyalty CRM attributes with tier-correlated behavioral assumptions."""
    tiers = ["Bronze", "Silver", "Gold", "Platinum"]
    tier_probs = [0.45, 0.30, 0.20, 0.05]
    points_by_tier = {
        "Bronze": (100, 1000),
        "Silver": (500, 3000),
        "Gold": (2000, 6000),
        "Platinum": (5000, 12000),
    }
    app_usage_mean = {"Bronze": 0.30, "Silver": 0.50, "Gold": 0.70, "Platinum": 0.85}
    discount_mean = {"Bronze": 0.75, "Silver": 0.55, "Gold": 0.35, "Platinum": 0.20}
    email_opt_in_p = {"Bronze": 0.55, "Silver": 0.68, "Gold": 0.80, "Platinum": 0.88}

    assigned_tiers = rng.choice(tiers, len(customer_ids), p=tier_probs)
    points = np.empty(len(customer_ids), dtype=int)
    app_usage = np.empty(len(customer_ids))
    discount_sens = np.empty(len(customer_ids))
    email_opt_in = np.empty(len(customer_ids), dtype=bool)

    for i, tier in enumerate(assigned_tiers):
        low, high = points_by_tier[tier]
        points[i] = rng.integers(low, high)
        app_usage[i] = np.clip(rng.normal(app_usage_mean[tier], 0.15), 0, 1)
        discount_sens[i] = np.clip(rng.normal(discount_mean[tier], 0.12), 0, 1)
        email_opt_in[i] = rng.random() < email_opt_in_p[tier]

    return pd.DataFrame({
        "customer_id": customer_ids.values,
        "tier": assigned_tiers,
        "points_balance": points,
        "email_opt_in": email_opt_in,
        "app_usage_score": np.round(app_usage, 2),
        "discount_sensitivity": np.round(discount_sens, 2),
    })


def add_crm(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["customer_id"].drop_duplicates().sort_values()
    crm = _generate_synthetic_crm(ids, rng)
    return df.merge(crm, on="customer_id", how="left")


def build_fact_transactions() -> pd.DataFrame:
    df = load_superstore().reset_index(drop=True)
    df["transaction_id"] = df.index.map(lambda i: f"txn_{i:07d}")
    cities = df[["city", "state"]].drop_duplicates()
    order_keys = _order_weather_keys(df)
    df = add_context(df, load_macro(), load_weather(order_keys))
    return add_crm(df)


def build_dim_customers(fact: pd.DataFrame) -> pd.DataFrame:
    dim = fact.groupby("customer_id", as_index=False).agg(
        customer_name=("customer_name", "first"),
        segment=("segment", "first"),
        home_region=("region", "first"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        total_orders=("order_id", "nunique"),
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        avg_discount=("discount", "mean"),
        tier=("tier", "first"),
        points_balance=("points_balance", "first"),
        email_opt_in=("email_opt_in", "first"),
        app_usage_score=("app_usage_score", "first"),
        discount_sensitivity=("discount_sensitivity", "first"),
    )
    dim["avg_order_value"] = dim["total_sales"] / dim["total_orders"].clip(lower=1)
    return dim


def build_telco_customers() -> pd.DataFrame:
    path = DATA_RAW / "us_telco" / "telco_customer_churn.csv"
    if not path.exists():
        _download(US_TELCO_URL, path)
    telco = pd.read_csv(path).rename(columns={"customerID": "telco_customer_id"})
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
    telco["churn_flag"] = telco["Churn"].eq("Yes").astype(int)
    return telco


def build_instacart_users() -> pd.DataFrame | None:
    orders_path = DATA_RAW / "instacart" / "orders.csv"
    if not orders_path.exists() and not download_instacart():
        return None
    orders = pd.read_csv(orders_path)
    return orders.groupby("user_id", as_index=False).agg(
        total_orders=("order_id", "count"),
        avg_days_since_prior=("days_since_prior_order", "mean"),
        reorder_rate=("days_since_prior_order", lambda s: s.notna().mean()),
    )


def run() -> dict[str, Path]:
    DATA_MODELING.mkdir(parents=True, exist_ok=True)
    fact = build_fact_transactions()
    dim = build_dim_customers(fact)
    telco = build_telco_customers()
    features = build_customer_features(fact, dim)

    outputs = {
        "fact_transactions": DATA_MODELING / "fact_transactions.parquet",
        "dim_customers": DATA_MODELING / "dim_customers.parquet",
        "customer_features": DATA_MODELING / "customer_features.parquet",
        "telco_customers": DATA_MODELING / "telco_customers.parquet",
    }
    fact.to_parquet(outputs["fact_transactions"], index=False)
    dim.to_parquet(outputs["dim_customers"], index=False)
    features.to_parquet(outputs["customer_features"], index=False)
    telco.to_parquet(outputs["telco_customers"], index=False)

    uplift = build_uplift_campaigns(fact, features)
    outputs["uplift_campaigns"] = DATA_MODELING / "uplift_campaigns.parquet"
    uplift.to_parquet(outputs["uplift_campaigns"], index=False)

    nba_catalog = build_nba_catalog()
    outputs["nba_offer_catalog"] = DATA_MODELING / "nba_offer_catalog.parquet"
    nba_catalog.to_parquet(outputs["nba_offer_catalog"], index=False)

    nba_events = build_nba_events(fact, features)
    outputs["nba_offer_events"] = DATA_MODELING / "nba_offer_events.parquet"
    nba_events.to_parquet(outputs["nba_offer_events"], index=False)

    instacart_raw = DATA_RAW / "instacart"
    instacart_orders = build_instacart_orders(instacart_raw)
    if instacart_orders is not None:
        outputs["instacart_orders"] = DATA_MODELING / "instacart_orders.parquet"
        instacart_orders.to_parquet(outputs["instacart_orders"], index=False)
        instacart_users = instacart_orders.groupby("user_id", as_index=False).agg(
            total_orders=("order_id", "count"),
            avg_days_since_prior=("days_since_prior_order", "mean"),
            reorder_rate=("days_since_prior_order", lambda s: s.notna().mean()),
        )
        outputs["instacart_users"] = DATA_MODELING / "instacart_users.parquet"
        instacart_users.to_parquet(outputs["instacart_users"], index=False)
        baskets = build_instacart_baskets(instacart_raw)
        if baskets is not None:
            outputs["instacart_baskets"] = DATA_MODELING / "instacart_baskets.parquet"
            baskets.to_parquet(outputs["instacart_baskets"], index=False)

    try:
        from uci_pipeline import CACHE_ONLY, build_uci_spine
        for name, path in build_uci_spine(cache_only_weather=CACHE_ONLY).items():
            outputs[name] = path
    except Exception as exc:
        tqdm.write(f"UCI spine skipped: {exc}")

    return outputs


if __name__ == "__main__":
    for name, path in run().items():
        df = pd.read_parquet(path)
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}  ({len(df):,} rows)")
