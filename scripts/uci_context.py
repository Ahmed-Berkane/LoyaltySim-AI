"""UK/EU macro and weather context for UCI Online Retail pipelines."""

from __future__ import annotations

import io
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import holidays
import numpy as np
import pandas as pd
import requests

# Online Retail country name → Eurostat geo, holidays ISO, weather proxy city
COUNTRY_META: dict[str, dict[str, str]] = {
    "United Kingdom": {"geo": "UK", "holidays": "GB", "city": "London", "country_code": "GB"},
    "Germany": {"geo": "DE", "holidays": "DE", "city": "Berlin", "country_code": "DE"},
    "France": {"geo": "FR", "holidays": "FR", "city": "Paris", "country_code": "FR"},
    "EIRE": {"geo": "IE", "holidays": "IE", "city": "Dublin", "country_code": "IE"},
    "Spain": {"geo": "ES", "holidays": "ES", "city": "Madrid", "country_code": "ES"},
    "Netherlands": {"geo": "NL", "holidays": "NL", "city": "Amsterdam", "country_code": "NL"},
    "Belgium": {"geo": "BE", "holidays": "BE", "city": "Brussels", "country_code": "BE"},
    "Switzerland": {"geo": "CH", "holidays": "CH", "city": "Zurich", "country_code": "CH"},
    "Portugal": {"geo": "PT", "holidays": "PT", "city": "Lisbon", "country_code": "PT"},
    "Norway": {"geo": "NO", "holidays": "NO", "city": "Oslo", "country_code": "NO"},
    "Italy": {"geo": "IT", "holidays": "IT", "city": "Rome", "country_code": "IT"},
    "Finland": {"geo": "FI", "holidays": "FI", "city": "Helsinki", "country_code": "FI"},
    "Cyprus": {"geo": "CY", "holidays": "CY", "city": "Nicosia", "country_code": "CY"},
    "Sweden": {"geo": "SE", "holidays": "SE", "city": "Stockholm", "country_code": "SE"},
    "Austria": {"geo": "AT", "holidays": "AT", "city": "Vienna", "country_code": "AT"},
    "Denmark": {"geo": "DK", "holidays": "DK", "city": "Copenhagen", "country_code": "DK"},
    "Poland": {"geo": "PL", "holidays": "PL", "city": "Warsaw", "country_code": "PL"},
    "Greece": {"geo": "EL", "holidays": "GR", "city": "Athens", "country_code": "GR"},
    "Malta": {"geo": "MT", "holidays": "MT", "city": "Valletta", "country_code": "MT"},
    "Lithuania": {"geo": "LT", "holidays": "LT", "city": "Vilnius", "country_code": "LT"},
    "Czech Republic": {"geo": "CZ", "holidays": "CZ", "city": "Prague", "country_code": "CZ"},
}

EUROZONE_GEOS = frozenset({
    "DE", "FR", "ES", "NL", "BE", "PT", "IE", "FI", "AT", "IT", "CY", "EL", "MT", "LT",
})

EURO_INTEREST_FRED = "IRSTCI01EZM156N"
UK_INTEREST_FRED = "INTGSBGBM193N"

# OECD immediate rates via FRED (monthly). Eurozone geos use EURO_INTEREST_FRED above.
FRED_INTEREST_BY_GEO: dict[str, str] = {
    "UK": UK_INTEREST_FRED,
    "DK": "IRSTCI01DKM156N",
    "NO": "IRSTCI01NOM156N",
    "SE": "IRSTCI01SEM156N",
    "PL": "IRSTCI01PLM156N",
    "CZ": "IRSTCI01CZM156N",
    "CH": "IRSTCI01CHM156N",
}

MACRO_COLUMNS = [
    "month",
    "country",
    "geo",
    "cpi_index",
    "inflation_mom",
    "inflation_yoy",
    "unemployment_rate",
    "interest_rate",
]

WEATHER_DAILY_VARS = (
    "temperature_2m_mean,rain_sum,snowfall_sum,precipitation_sum,"
    "weather_code,wind_gusts_10m_max"
)
STORM_WEATHER_CODES = frozenset({65, 67, 75, 77, 82, 85, 86, 95, 96, 99})
MAJOR_STORM_WIND_KMH = 75.0
OPEN_METEO_REQUEST_DELAY_SEC = 10
WEATHER_DATE_CHUNK = 100
HTTP_HEADERS = {"User-Agent": "LoyaltySim-AI/1.0 (research; uci-context)"}


class OpenMeteoQuotaExceeded(Exception):
    """Open-Meteo rate or daily quota limit hit."""


def country_to_geo(country: str) -> str | None:
    meta = COUNTRY_META.get(country)
    return meta["geo"] if meta else None


def _eurostat_sdmx_csv(dataset: str, key: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
        f"{dataset}/{key}?startPeriod={start}&endPeriod={end}&format=SDMX-CSV"
    )
    r = requests.get(url, headers=HTTP_HEADERS, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return df
    df = df.rename(columns={"TIME_PERIOD": "month", "OBS_VALUE": "value", "geo": "geo"})
    df["month"] = df["month"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _fred_monthly(series_id: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}&cosd={start}-01&coed={end}-28"
    )
    r = requests.get(url, headers=HTTP_HEADERS, timeout=90)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    col = [c for c in raw.columns if c != "observation_date"][0]
    out = raw.rename(columns={"observation_date": "month", col: "interest_rate"})
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").astype(str)
    out["interest_rate"] = pd.to_numeric(out["interest_rate"], errors="coerce")
    return out[["month", "interest_rate"]]


def _interest_series_for_geo(geo: str) -> str | None:
    if geo in FRED_INTEREST_BY_GEO:
        return FRED_INTEREST_BY_GEO[geo]
    if geo in EUROZONE_GEOS:
        return EURO_INTEREST_FRED
    return None


def _load_interest_rates_by_geo(geos: list[str], month_start: str, month_end: str) -> pd.DataFrame:
    """Fetch FRED OECD rates once per series, assign to Eurostat geo codes."""
    series_to_geos: dict[str, list[str]] = {}
    for geo in geos:
        series_id = _interest_series_for_geo(geo)
        if series_id:
            series_to_geos.setdefault(series_id, []).append(geo)

    year_start, year_end = month_start[:4], month_end[:4]
    frames: list[pd.DataFrame] = []
    for series_id, geo_list in series_to_geos.items():
        try:
            rates = _fred_monthly(series_id, year_start, year_end)
        except requests.RequestException:
            continue
        for geo in geo_list:
            frames.append(rates.assign(geo=geo))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["month", "geo", "interest_rate"])


def load_eu_macro(
    geos: list[str],
    month_start: str,
    month_end: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Monthly macro by Eurostat geo code.

    CPI + unemployment from Eurostat; interest rate from FRED (UK + euro area).
    """
    geos = sorted({g for g in geos if g})
    if not geos:
        return pd.DataFrame(columns=MACRO_COLUMNS)

    if cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        geos_complete = (
            not cached.empty
            and "interest_rate" in cached.columns
            and not cached.groupby("geo")["interest_rate"].apply(lambda s: s.isna().all()).reindex(geos).fillna(True).any()
        )
        if (
            not cached.empty
            and set(MACRO_COLUMNS).issubset(cached.columns)
            and cached["month"].min() <= month_start
            and cached["month"].max() >= month_end
            and set(geos).issubset(set(cached["geo"].unique()))
            and geos_complete
        ):
            return cached[cached["month"].between(month_start, month_end)].copy()

    geo_key = "+".join(geos)
    start_p = month_start
    end_p = month_end

    hicp = _eurostat_sdmx_csv("PRC_HICP_MIDX", f"M.I15.CP00.{geo_key}", start_p, end_p)
    unemp = _eurostat_sdmx_csv("UNE_RT_M", f"M.SA..PC_ACT.T.{geo_key}", start_p, end_p)

    cpi = (
        hicp.groupby(["geo", "month"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "cpi_index"})
        .sort_values(["geo", "month"])
    )
    cpi["inflation_mom"] = cpi.groupby("geo")["cpi_index"].pct_change().round(5)
    cpi["inflation_yoy"] = cpi.groupby("geo")["cpi_index"].pct_change(12).round(5)

    un = (
        unemp.groupby(["geo", "month"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "unemployment_rate"})
    )

    macro = cpi.merge(un, on=["geo", "month"], how="left")

    geo_to_country = {v["geo"]: k for k, v in COUNTRY_META.items()}
    macro["country"] = macro["geo"].map(geo_to_country)

    rates = _load_interest_rates_by_geo(geos, month_start, month_end)
    if not rates.empty:
        macro = macro.drop(columns=["interest_rate"], errors="ignore").merge(
            rates, on=["geo", "month"], how="left"
        )
    else:
        macro["interest_rate"] = np.nan

    macro = macro[
        (macro["month"] >= month_start) & (macro["month"] <= month_end)
    ].reset_index(drop=True)
    macro = macro[[c for c in MACRO_COLUMNS if c in macro.columns]]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(cache_path, index=False)
    return macro


def _geocode_proxy(city: str, country_code: str) -> dict | None:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 5, "countryCode": country_code},
        headers=HTTP_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    match = results[0]
    return {
        "latitude": match["latitude"],
        "longitude": match["longitude"],
        "timezone": match["timezone"],
    }


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


def _open_meteo_daily_frame(payload: dict | list) -> pd.DataFrame:
    if isinstance(payload, list):
        parts = [pd.DataFrame(item["daily"]) for item in payload]
        return pd.concat(parts, ignore_index=True)
    return pd.DataFrame(payload["daily"])


def _fetch_weather_range(
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """One archive call per country — full date range (Open-Meteo daily series)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": WEATHER_DAILY_VARS,
        "timezone": timezone,
    }
    attempt = 0
    while True:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            headers=HTTP_HEADERS,
            timeout=120,
        )
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 0)) or min(60 * (2 ** attempt), 600)
            attempt += 1
            if attempt >= 20:
                raise OpenMeteoQuotaExceeded("Rate limited after retries.")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return _open_meteo_daily_frame(r.json())


def _fetch_weather_archive(
    latitude: float, longitude: float, timezone: str, dates: list[str]
) -> pd.DataFrame:
    """Legacy per-day batch fetch — prefer _fetch_weather_range for full country backfill."""
    if not dates:
        return pd.DataFrame()
    return _fetch_weather_range(latitude, longitude, timezone, dates[0], dates[-1])


def order_weather_keys(transactions: pd.DataFrame, date_start: str, date_end: str) -> pd.DataFrame:
    out = transactions.copy()
    out["date"] = out["order_date"].dt.normalize()
    start = pd.Timestamp(date_start).normalize()
    end = pd.Timestamp(date_end).normalize()
    in_range = out["date"].between(start, end)
    return out.loc[in_range, ["date", "country"]].drop_duplicates()


def load_eu_weather(
    order_keys: pd.DataFrame,
    cache_path: Path,
    *,
    cache_only: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Daily weather at country proxy cities; one API call per country (full date range)."""
    if cache_only and not cache_path.exists():
        return pd.DataFrame()

    cache = (
        pd.read_csv(cache_path, parse_dates=["date"])
        if cache_path.exists()
        else pd.DataFrame()
    )
    if not cache.empty:
        cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()

    countries = sorted(order_keys["country"].dropna().unique()) if not order_keys.empty else []
    proxy_coords_path = cache_path.parent / "eu_country_proxy_coordinates.csv"
    coords = (
        pd.read_csv(proxy_coords_path)
        if proxy_coords_path.exists()
        else pd.DataFrame(columns=["country", "latitude", "longitude", "timezone"])
    )
    have_coords = set(coords["country"]) if not coords.empty else set()
    new_rows = []
    for country in countries:
        if country in have_coords:
            continue
        meta = COUNTRY_META.get(country)
        if not meta:
            continue
        geo = _geocode_proxy(meta["city"], meta["country_code"])
        if geo:
            new_rows.append({"country": country, **geo})
    if new_rows:
        coords = pd.concat([coords, pd.DataFrame(new_rows)], ignore_index=True)
        proxy_coords_path.parent.mkdir(parents=True, exist_ok=True)
        coords.to_csv(proxy_coords_path, index=False)

    if cache_only or coords.empty or order_keys.empty:
        return _filter_weather_to_keys(cache, order_keys)

    dates_by_country = (
        order_keys.groupby("country")["date"]
        .apply(lambda s: sorted(pd.to_datetime(s).dt.strftime("%Y-%m-%d").unique()))
        .to_dict()
    )

    pending: list[tuple] = []
    for row in coords.itertuples(index=False):
        if row.country not in dates_by_country:
            continue
        needed = set(dates_by_country[row.country])
        have: set[str] = set()
        if not cache.empty:
            have = set(
                cache.loc[cache["country"] == row.country, "date"]
                .dt.strftime("%Y-%m-%d")
            )
        if needed.issubset(have):
            continue
        dates = dates_by_country[row.country]
        pending.append((row, dates[0], dates[-1], len(needed)))

    if verbose and pending:
        print(f"Fetching weather for {len(pending)} countries (1 API call each)...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for i, (row, start_date, end_date, n_keys) in enumerate(pending, start=1):
            if verbose:
                print(f"  [{i}/{len(pending)}] {row.country}: {start_date} -> {end_date} ({n_keys} order dates)")
            frame = _add_weather_flags(
                _fetch_weather_range(row.latitude, row.longitude, row.timezone, start_date, end_date)
            )
            frame["country"] = row.country
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            if not cache.empty:
                cache = cache[cache["country"] != row.country]
            cache = pd.concat([cache, frame], ignore_index=True)
            cache = cache.drop_duplicates(subset=["date", "country"], keep="last")
            cache.to_csv(cache_path, index=False)
            if i < len(pending):
                time.sleep(OPEN_METEO_REQUEST_DELAY_SEC)
    except OpenMeteoQuotaExceeded:
        if not cache.empty:
            cache.to_csv(cache_path, index=False)
        raise

    if verbose and not pending:
        print("Weather cache already complete for all order keys.")

    return _filter_weather_to_keys(cache, order_keys)


def _filter_weather_to_keys(weather: pd.DataFrame, order_keys: pd.DataFrame) -> pd.DataFrame:
    if weather.empty or order_keys.empty:
        return weather
    keys = order_keys.copy()
    keys["date"] = pd.to_datetime(keys["date"]).dt.normalize()
    out = weather.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.merge(keys, on=["date", "country"], how="inner")


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return date(year, month, 1 + (weekday - first.weekday()) % 7 + 7 * (n - 1))


def _thanksgiving(year: int) -> date:
    nov1 = date(year, 11, 1)
    return date(year, 11, 1 + (3 - nov1.weekday()) % 7 + 21)


def _black_friday(year: int) -> date:
    return _thanksgiving(year) + timedelta(days=1)


def _cyber_monday(year: int) -> date:
    return _thanksgiving(year) + timedelta(days=4)


def _week_bounds(anchor: date) -> tuple[date, date]:
    start = anchor - timedelta(days=anchor.weekday())
    return start, start + timedelta(days=6)


def _eu_retail_spending_days(years: range) -> set[date]:
    days: set[date] = set()
    for year in years:
        days.add(date(year, 2, 14))
        days.add(date(year, 12, 24))
        days.add(_nth_weekday_of_month(year, 5, 6, 2))
        days.add(_cyber_monday(year))
    return days


def _anchor_ts_by_year(
    years: range,
    anchor_fn: Callable[[int], date],
) -> dict[int, pd.Timestamp]:
    return {year: pd.Timestamp(anchor_fn(year)) for year in years}


def _is_in_anchor_week(order_ts: pd.Series, anchor_by_year: dict[int, pd.Timestamp]) -> pd.Series:
    anchor = order_ts.dt.year.map(anchor_by_year)
    week_start = anchor - pd.to_timedelta(anchor.dt.dayofweek, unit="D")
    week_end = week_start + pd.Timedelta(days=6)
    return (order_ts >= week_start) & (order_ts <= week_end)


def _days_to_next_annual_event(order_ts: pd.Series, month: int, day: int) -> pd.Series:
    event = pd.to_datetime(
        order_ts.dt.year.astype(str) + f"-{month:02d}-{day:02d}"
    )
    event = event.where(order_ts <= event, event + pd.DateOffset(years=1))
    return (event - order_ts).dt.days


def _days_to_next_black_friday(order_ts: pd.Series, bf_by_year: dict[int, pd.Timestamp]) -> pd.Series:
    bf = order_ts.dt.year.map(bf_by_year)
    bf = bf.where(order_ts <= bf, order_ts.dt.year.add(1).map(bf_by_year))
    return (bf - order_ts).dt.days


def add_shopping_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pan-EU retail events + country-specific sale days (no nulls)."""
    out = df.copy()
    ts = out["order_date"]
    month, day = ts.dt.month, ts.dt.day
    country = out["country"]

    out["is_christmas_season"] = ((month == 11) & (day >= 15)) | ((month == 12) & (day <= 24))
    out["is_back_to_school"] = ((month == 8) & (day >= 15)) | ((month == 9) & (day <= 15))

    year_min, year_max = ts.dt.year.min(), ts.dt.year.max()
    years = range(year_min - 1, year_max + 2)
    bf_by_year = _anchor_ts_by_year(years, _black_friday)
    cm_by_year = _anchor_ts_by_year(years, _cyber_monday)

    out["is_black_friday_week"] = _is_in_anchor_week(ts, bf_by_year)
    out["is_cyber_monday_week"] = _is_in_anchor_week(ts, cm_by_year)
    out["is_amazon_prime_day"] = (month == 7) & (day >= 10) & (day <= 20) & (ts.dt.year >= 2015)

    out["days_to_christmas"] = _days_to_next_annual_event(ts, 12, 25)
    out["days_to_black_friday"] = _days_to_next_black_friday(ts, bf_by_year)

    out["is_sinterklaas"] = (country == "Netherlands") & (month == 12) & (day == 5)
    out["is_three_kings_day"] = (country == "Spain") & (month == 1) & (day == 6)
    out["is_italian_epiphany"] = (country == "Italy") & (month == 1) & (day == 6)
    out["is_boxing_day"] = country.isin(["United Kingdom", "EIRE"]) & (month == 12) & (day == 26)
    out["is_polish_childrens_day"] = (country == "Poland") & (month == 6) & (day == 1)
    out["is_french_winter_sale"] = (country == "France") & (
        ((month == 1) & (day >= 10)) | ((month == 2) & (day <= 10))
    )
    out["is_french_summer_sale"] = (country == "France") & (
        ((month == 6) & (day >= 20)) | (month == 7)
    )

    out["is_major_sale_period"] = (
        out["is_black_friday_week"]
        | out["is_cyber_monday_week"]
        | out["is_christmas_season"]
        | out["is_back_to_school"]
        | out["is_amazon_prime_day"]
        | out["is_sinterklaas"]
        | out["is_three_kings_day"]
        | out["is_italian_epiphany"]
        | out["is_boxing_day"]
        | out["is_polish_childrens_day"]
        | out["is_french_winter_sale"]
        | out["is_french_summer_sale"]
    )
    return out


def add_context_uci(
    df: pd.DataFrame,
    macro: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Calendar + country holidays + EU macro + country-proxy weather."""
    out = df.copy()
    out["date"] = out["order_date"].dt.normalize()
    out["month"] = out["order_date"].dt.to_period("M").astype(str)
    out["geo"] = out["country"].map(country_to_geo)

    years = range(out["order_date"].dt.year.min() - 1, out["order_date"].dt.year.max() + 2)
    holiday_sets: dict[str, set] = {}
    for country, meta in COUNTRY_META.items():
        try:
            holiday_sets[country] = set(holidays.country_holidays(meta["holidays"], years=years).keys())
        except NotImplementedError:
            holiday_sets[country] = set()

    retail_days = _eu_retail_spending_days(range(years.start, years.stop))

    out["is_public_holiday"] = [
        d in holiday_sets.get(c, set())
        for d, c in zip(out["date"].dt.date, out["country"])
    ]
    out["is_retail_spending_day"] = out["date"].dt.date.isin(retail_days)
    out["day_of_week"] = out["order_date"].dt.dayofweek
    out["is_friday"] = out["day_of_week"] == 4
    out["is_month_start"] = out["order_date"].dt.day <= 5
    out["is_month_end"] = out["order_date"].dt.day >= 25
    out = add_shopping_calendar_features(out)

    if not macro.empty:
        macro_join = macro.copy()
        out = out.merge(
            macro_join[[c for c in MACRO_COLUMNS if c in macro_join.columns and c != "country"]],
            on=["geo", "month"],
            how="left",
        )

    if not weather.empty:
        weather = weather.copy()
        weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
        weather = weather.drop_duplicates(subset=["date", "country"], keep="last")
        weather_cols = [
            "date", "country", "temp_c", "rain_mm", "snow_cm", "precip_mm",
            "weather_code", "wind_gust_kmh", "had_rain", "had_snow", "had_major_storm",
        ]
        out = out.merge(weather[[c for c in weather_cols if c in weather.columns]], on=["date", "country"], how="left")

    return out


def generate_synthetic_crm(customer_ids: pd.Series, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
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

    rows = []
    for cid in customer_ids:
        tier = rng.choice(tiers, p=tier_probs)
        low, high = points_by_tier[tier]
        rows.append({
            "customer_id": cid,
            "tier": tier,
            "points_balance": int(rng.integers(low, high)),
            "email_opt_in": bool(rng.random() < email_opt_in_p[tier]),
            "app_usage_score": round(float(np.clip(rng.normal(app_usage_mean[tier], 0.15), 0, 1)), 2),
            "discount_sensitivity": round(float(np.clip(rng.normal(discount_mean[tier], 0.12), 0, 1)), 2),
        })
    return pd.DataFrame(rows)


def build_uci_dim_customers(fact: pd.DataFrame) -> pd.DataFrame:
    as_of = fact["order_date"].max()
    dim = fact.groupby("customer_id", as_index=False).agg(
        country=("country", "first"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        total_orders=("order_id", "nunique"),
        total_lines=("order_id", "count"),
        total_revenue=("line_total", "sum"),
        avg_unit_price=("unit_price", "mean"),
        tier=("tier", "first"),
        points_balance=("points_balance", "first"),
        email_opt_in=("email_opt_in", "first"),
        app_usage_score=("app_usage_score", "first"),
        discount_sensitivity=("discount_sensitivity", "first"),
    )
    dim["avg_order_value"] = dim["total_revenue"] / dim["total_orders"].clip(lower=1)
    dim["recency_days"] = (as_of - dim["last_order_date"]).dt.days
    dim["tenure_days"] = (dim["last_order_date"] - dim["first_order_date"]).dt.days
    return dim
