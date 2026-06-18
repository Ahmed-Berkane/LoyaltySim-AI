# LoyaltySim-AI

AI-powered loyalty intelligence platform — churn prediction, CLV, uplift modeling, and campaign simulation.

## Run

```powershell
pip install -r requirements.txt
python scripts/build_datasets.py
```

Optional: Instacart needs `~/.kaggle/kaggle.json`. Without it, the other 3 outputs still build.

Walk through the logic step-by-step in `data_prep.ipynb`.

## Pipeline

```
Download raw (Superstore, Telco, Instacart, CPI, weather)
  → add US holidays + macro + weather to Superstore
  → add synthetic loyalty CRM (tier, points, …)
  → save final CSVs to data/processed/
```

**Join rule:** Superstore + context + CRM → `fact_transactions`. Group by `customer_id` → `dim_customers`. Telco and Instacart are separate (different customer IDs).

## Output files

| File | Rows | Used by |
|------|------|---------|
| `fact_transactions.csv` | ~10k | Segmentation, CLV, uplift, profit optimization |
| `dim_customers.csv` | ~800 | Churn (inactivity), RFM, next-best-action |
| `telco_customers.csv` | ~7k | Churn model (XGBoost/LightGBM) |
| `instacart_users.csv` | ~206k | Basket/reorder model (optional) |

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
| cpi_index, inflation_mom, inflation_yoy | FRED | Macro context |
| temp_c, precip_mm | Open-Meteo | Weather context |
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
scripts/build_datasets.py   ← single entry point
data_prep.ipynb             ← step-by-step walkthrough
data/raw/                   ← cached downloads
data/processed/             ← final modeling CSVs
```
