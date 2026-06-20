# Modeling Data Guide

Answers to common readiness questions and how to use each `data/modeling/*.parquet` file.

## Quick answers

| Question | Answer |
|----------|--------|
| **Can we integrate Instacart?** | Yes — pipeline now exports `instacart_orders` (~3M orders) and `instacart_baskets` (~32M line items with `reordered` flag). Requires Kaggle raw files in `data/raw/instacart/`. |
| **Is RFM/segmentation blocked?** | No data gap — 793 Superstore customers is small but valid. Use `customer_features.parquet` (`R_score`, `F_score`, `M_score`, `rfm_segment`). |
| **More CLV / seasonality history?** | Yes — **UCI Online Retail I** added as a separate universe: ~400k transactions, ~4.3k customers, Dec 2010–Dec 2011 (UK). Use `online_retail_transactions.parquet`. |
| **Uplift modeling?** | Yes (synthetic RCT) — `uplift_campaigns.parquet` has 2 waves × customers with `treatment`, `offer_discount_pct`, `responded`, `incremental_revenue`. Seeded from `discount_sensitivity`. |
| **Next best action?** | Partial → **prototype-ready** — `nba_offer_catalog.parquet` + `nba_offer_events.parquet` (shown / clicked / converted). Simulated from category affinity + RFM. |
| **Feature store?** | Yes — `customer_features.parquet` is the single customer feature table (16 engineered features + labels). Built by `scripts/feature_store.py`, called from `build_datasets.py`. |

---

## Modeling universes (still not cross-joined)

```
Superstore spine     → fact_transactions, dim_customers, customer_features, uplift, nba
Online Retail (UK)   → online_retail_transactions, online_retail_customers  [CLV scale-up]
Telco                → telco_customers  [labeled churn]
Instacart            → instacart_orders, instacart_baskets, instacart_users
```

Customer IDs are **not** shared across universes by design.

---

## File reference

### Superstore loyalty spine

| File | Grain | Use |
|------|-------|-----|
| `fact_transactions.parquet` | Line item | CLV, seasonality, profit |
| `dim_customers.parquet` | Customer | Aggregates |
| `customer_features.parquet` | Customer | **Feature store** — RFM, behavioral, temporal, loyalty features + `inactivity_churn_label` |
| `uplift_campaigns.parquet` | Customer × campaign wave | Uplift / CATE models (`treatment`, `responded`, `incremental_revenue`) |
| `nba_offer_catalog.parquet` | Offer | Static offer definitions |
| `nba_offer_events.parquet` | Customer × offer | Ranking / conversion models (`shown`, `clicked`, `converted`) |

### CLV scale-up (Online Retail I)

| File | Grain | Use |
|------|-------|-----|
| `online_retail_transactions.parquet` | Line item | BG/NBD, Gamma-Gamma CLV, seasonality |
| `online_retail_customers.parquet` | Customer | RFM at 4k+ customer scale |

### Instacart (optional)

| File | Grain | Use |
|------|-------|-----|
| `instacart_orders.parquet` | Order | Sequence / cadence |
| `instacart_baskets.parquet` | Order line | Reorder prediction, market basket |
| `instacart_users.parquet` | User | User-level aggregates |

### Telco benchmark

| File | Grain | Use |
|------|-------|-----|
| `telco_customers.parquet` | Account | Supervised churn (`churn_flag`) |

---

## Labels & synthetic data honesty

| Field | Type |
|-------|------|
| `telco.churn_flag` | **Real** labeled outcome |
| `customer_features.inactivity_churn_label` | **Derived** (180d since last order) |
| `uplift_campaigns.*` | **Synthetic RCT** (reproducible, seed=42) |
| `nba_offer_events.*` | **Simulated** CRM log |
| CRM tier / points on Superstore | **Synthetic** (seed=42) |

Swap synthetic tables for production CRM exports when available.

---

## Rebuild everything

```powershell
pip install -r requirements.txt
python scripts/build_datasets.py
```

Feature logic lives in `scripts/feature_store.py` — keep EDA and training in sync by always reading `customer_features.parquet`, not re-deriving in notebooks.

---

## Modeling notebooks (`Notebooks/01`–`05`)

Run after `build_datasets.py`. Each notebook loads Parquet, explores correlation / feature selection, splits train/val/test where applicable, compares models, and saves the best artifact to `models/` (gitignored).

| Notebook | Data | Models | Saved artifact |
|----------|------|--------|----------------|
| `01-segmentation.ipynb` | `customer_features.parquet` | KMeans, GMM, Hierarchical | `01_segmentation_best.joblib` |
| `02-churn-prediction.ipynb` | `telco_customers.parquet` | XGBoost, LightGBM, MLP | `02_churn_best.joblib` |
| `03-clv.ipynb` | `online_retail_transactions.parquet` | BG/NBD + Gamma-Gamma, MLP | `03_clv_best.joblib` + `03_clv_bgf` / `03_clv_ggf` |
| `04-uplift.ipynb` | `uplift_campaigns.parquet` | T-learner, X-learner, Uplift RF | `04_uplift_best.joblib` |
| `05-next-best-action.ipynb` | `nba_offer_events` + `customer_features` | XGBoost / RF + tabular Q-learning | `05_nba_supervised_best.joblib`, `05_nba_rl_qtable.joblib` |

Regenerate notebook templates: `python scripts/generate_modeling_notebooks.py`

Quick smoke test (no plots): `python scripts/smoke_test_modeling.py`
