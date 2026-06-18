"""Download raw USA data and build final modeling datasets."""

from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_OUT = PROJECT_ROOT / "data" / "processed"

US_SUPERSTORE_URL = (
    "https://raw.githubusercontent.com/ThigasSantos/BaseSuperStore/main/sales.csv"
)
US_TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv"
)
US_CPI_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    "&cosd=2015-01-01&coed=2018-12-31"
)
INSTACART_DATASET = "yasserh/instacart-online-grocery-basket-analysis-dataset"
INSTACART_FILES = [
    "orders.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
]

REGION_WEATHER = {
    "South": (25.7617, -80.1918, "America/New_York"),
    "West": (34.0522, -118.2437, "America/Los_Angeles"),
    "East": (40.7128, -74.0060, "America/New_York"),
    "Central": (41.8781, -87.6298, "America/Chicago"),
}


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


def load_macro() -> pd.DataFrame:
    path = DATA_RAW / "external" / "us_macro_monthly.csv"
    if path.exists():
        return pd.read_csv(path)
    try:
        r = requests.get(US_CPI_URL, timeout=90)
        r.raise_for_status()
        cpi = pd.read_csv(io.StringIO(r.text))
        cpi = cpi.rename(columns={"observation_date": "month", "CPIAUCSL": "cpi_index"})
        cpi["month"] = pd.to_datetime(cpi["month"]).dt.to_period("M").astype(str)
        cpi["cpi_index"] = pd.to_numeric(cpi["cpi_index"], errors="coerce")
        cpi = cpi.dropna(subset=["cpi_index"])
    except (requests.RequestException, ValueError, KeyError):
        months = pd.period_range("2015-01", "2018-12", freq="M").astype(str)
        cpi = pd.DataFrame({"month": months, "cpi_index": 240.0})
    cpi["inflation_mom"] = cpi["cpi_index"].pct_change().round(5)
    cpi["inflation_yoy"] = cpi["cpi_index"].pct_change(12).round(5)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpi.to_csv(path, index=False)
    return cpi


def load_weather() -> pd.DataFrame:
    path = DATA_RAW / "external" / "us_weather_by_region.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    frames = []
    for region, (lat, lon, tz) in REGION_WEATHER.items():
        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date=2015-01-01&end_date=2018-12-31&"
            "daily=temperature_2m_mean,precipitation_sum&"
            f"timezone={tz.replace('/', '%2F')}"
        )
        daily = requests.get(url, timeout=90).json()["daily"]
        frame = pd.DataFrame(daily).rename(columns={
            "time": "date", "temperature_2m_mean": "temp_c", "precipitation_sum": "precip_mm",
        })
        frame["region"] = region
        frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(path, index=False)
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


def add_context(df: pd.DataFrame, macro: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    us_holidays = holidays.country_holidays("US", years=range(2015, 2020))
    out = df.copy()
    out["date"] = out["order_date"].dt.normalize()
    out["month"] = out["order_date"].dt.to_period("M").astype(str)
    out["is_us_federal_holiday"] = out["date"].dt.date.map(lambda d: d in us_holidays)
    out["day_of_week"] = out["order_date"].dt.dayofweek
    out["is_us_payday_window"] = out["order_date"].dt.day.isin([1, 15])
    out = out.merge(macro[["month", "cpi_index", "inflation_mom", "inflation_yoy"]], on="month", how="left")
    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    return out.merge(weather, on=["date", "region"], how="left")


def add_crm(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = df["customer_id"].drop_duplicates().sort_values()
    crm = pd.DataFrame({
        "customer_id": ids.values,
        "tier": rng.choice(["Bronze", "Silver", "Gold", "Platinum"], len(ids), p=[0.45, 0.30, 0.20, 0.05]),
        "points_balance": rng.integers(100, 8000, len(ids)),
        "email_opt_in": rng.choice([True, False], len(ids), p=[0.72, 0.28]),
        "app_usage_score": rng.uniform(0, 1, len(ids)).round(2),
        "discount_sensitivity": rng.uniform(0, 1, len(ids)).round(2),
    })
    return df.merge(crm, on="customer_id", how="left")


def build_fact_transactions() -> pd.DataFrame:
    df = load_superstore().reset_index(drop=True)
    df["transaction_id"] = df.index.map(lambda i: f"txn_{i:07d}")
    df = add_context(df, load_macro(), load_weather())
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
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    fact = build_fact_transactions()
    dim = build_dim_customers(fact)
    telco = build_telco_customers()

    outputs = {
        "fact_transactions": DATA_OUT / "fact_transactions.csv",
        "dim_customers": DATA_OUT / "dim_customers.csv",
        "telco_customers": DATA_OUT / "telco_customers.csv",
    }
    fact.to_csv(outputs["fact_transactions"], index=False)
    dim.to_csv(outputs["dim_customers"], index=False)
    telco.to_csv(outputs["telco_customers"], index=False)

    instacart = build_instacart_users()
    if instacart is not None:
        outputs["instacart_users"] = DATA_OUT / "instacart_users.csv"
        instacart.to_csv(outputs["instacart_users"], index=False)

    return outputs


if __name__ == "__main__":
    for name, path in run().items():
        df = pd.read_csv(path)
        print(f"{name}: {path.relative_to(PROJECT_ROOT)}  ({len(df):,} rows)")
