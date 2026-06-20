# LoyaltySim-AI

AI-powered loyalty intelligence platform — churn prediction, CLV, uplift modeling, and campaign simulation.

## Build modeling data (Parquet)

All **final modeling datasets** are written to `data/modeling/*.parquet`.  
Raw downloads, caches, and intermediates stay in `data/raw/` — both are **gitignored** (regenerate locally).

### One-time setup

```powershell
cd C:\Users\ahmed\Desktop\Projects\LoyaltySim-AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Optional — full Instacart outputs** (~32M basket rows): configure Kaggle, then re-run the build.

```powershell
# one-time: place kaggle.json in %USERPROFILE%\.kaggle\
pip install kaggle
```

Without Kaggle, Superstore + Telco + Online Retail + feature store + uplift + NBA still build.

### Build (every time you want fresh Parquet)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/build_datasets.py
```

Expected output:

```
fact_transactions:        data\modeling\fact_transactions.parquet
dim_customers:            data\modeling\dim_customers.parquet
customer_features:        data\modeling\customer_features.parquet
telco_customers:          data\modeling\telco_customers.parquet
uplift_campaigns:         data\modeling\uplift_campaigns.parquet
nba_offer_catalog:        data\modeling\nba_offer_catalog.parquet
nba_offer_events:         data\modeling\nba_offer_events.parquet
instacart_*:              (if Kaggle configured)
online_retail_*:          (UCI download on first run)
```

**Alternative:** run `Notebooks/data_prep.ipynb` top-to-bottom — same files, step-by-step.

> **Weather:** `build_datasets.py` reads **`data/raw/external/us_weather_by_city.csv` only** (no Open-Meteo calls). Populate that file once via `Notebooks/data_prep.ipynb` Step 2b if it is missing.

## Pipeline

```
Download raw (Superstore, Telco, Instacart, CPI, weather)
  → add US holidays + macro + weather to Superstore
  → add synthetic loyalty CRM (tier, points, …)
  → save modeling datasets as Parquet to data/modeling/
```

**Join rule:** Superstore + context + CRM → `fact_transactions`. Group by `customer_id` → `dim_customers`. Telco and Instacart are separate (different customer IDs).

## Output files

| File | Rows | Used by |
|------|------|---------|
| `fact_transactions.parquet` | ~10k | Segmentation, CLV, uplift, profit optimization |
| `dim_customers.parquet` | ~800 | Churn (inactivity), RFM, next-best-action |
| `telco_customers.parquet` | ~7k | Churn model (XGBoost/LightGBM) |
| `instacart_users.parquet` | ~206k | Basket/reorder model (optional) |
| `instacart_orders.parquet` | ~3M | Order-sequence / cadence models |
| `instacart_baskets.parquet` | ~32M | Item-level reorder & market-basket |
| `customer_features.parquet` | ~800 | **Feature store** — RFM, segments, engineered features |
| `uplift_campaigns.parquet` | ~1.6k | Synthetic RCT uplift modeling |
| `nba_offer_catalog.parquet` | 8 | Next-best-action offer definitions |
| `nba_offer_events.parquet` | ~800 | NBA show/click/convert log (simulated) |
| `online_retail_transactions.parquet` | ~400k | CLV / seasonality at scale (UK, UCI) |
| `online_retail_customers.parquet` | ~4k | Online Retail customer rollups |

## Columns

### fact_transactions (one row = one order line)

| Column | Source | Model use |
|--------|--------|-----------|
| transaction_id, order_id, order_date | Superstore | Time series, seasonality |
| customer_id, segment | Superstore | Segmentation, CLV |
| sales, quantity, discount, profit | Superstore | CLV, profit optimization |
| region, state, city | Superstore | Geo effects |
| category, sub_category, product_name | Superstore | Product affinity |
| is_us_federal_holiday, day_of_week, is_us_payday_window | Derived | Temporal features |
| cpi_index, inflation_mom, inflation_yoy, unemployment_rate, interest_rate | FRED | Macro context |
| temp_c, rain_mm, snow_cm, precip_mm, weather_code, wind_gust_kmh, had_rain, had_snow, had_major_storm | Open-Meteo | Weather context (by city) |
| tier, points_balance, email_opt_in, app_usage_score, discount_sensitivity | Synthetic CRM | Loyalty simulation, uplift |

### dim_customers (one row = one customer)

| Column | Model use |
|--------|-----------|
| total_orders, total_sales, total_profit, avg_order_value | CLV, segmentation |
| first_order_date, last_order_date | Recency, churn |
| avg_discount, discount_sensitivity | Uplift, discount waste |
| tier, points_balance, app_usage_score | Next-best-action, RL agent |

### telco_customers (standalone)

| Column | Model use |
|--------|-----------|
| tenure, MonthlyCharges, TotalCharges, Contract | Churn prediction |
| PhoneService, InternetService, … (service flags) | Feature importance |
| churn_flag | Target label |

### instacart_users (standalone)

| Column | Model use |
|--------|-----------|
| total_orders, avg_days_since_prior, reorder_rate | Sequence/reorder model |

## Project layout

```
scripts/build_datasets.py   ← single entry point (+ feature store, uplift, NBA, Instacart depth)
scripts/feature_store.py  ← customer feature engineering
scripts/modeling_extensions.py  ← uplift, NBA, Online Retail, Instacart baskets
Notebooks/data_prep.ipynb   ← step-by-step walkthrough
Notebooks/EDA.ipynb         ← exploratory analysis
MODELING.md                 ← modeling readiness & file guide
data/raw/                   ← cached downloads (gitignored)
data/modeling/              ← final modeling Parquet (gitignored)
```
