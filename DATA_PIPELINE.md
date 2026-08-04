# Data pipeline (CLV)

**Entry point:** `python scripts/uci_pipeline.py`

**Setting:** noncontractual B2C retail (EU filter on UCI Online Retail II). No subscription cancel date. Revenue-only.

## Outputs

### From `uci_pipeline.py`

| File | Grain | Role |
|------|-------|------|
| `uci_fact_transactions.parquet` | invoice line | Clean UCI columns only (`customer_id`, `order_id`, `order_date`, `line_total`, …) |
| `uci_dim_customers.parquet` | customer | Lifetime rollups for sanity checks |

### From notebook `00` (`customer_base_audit.export_audit_summaries`)

| File | Grain | Role |
|------|-------|------|
| `audit/uci_orders.parquet` | **order** | Primary weekly CLV / audit spine (`spend = sum(line_total)`) |
| `audit/uci_customer_master.parquet` | customer | Consulting master table (tenure, recency, AOV, gaps) |
| `audit/uci_customer_lifetime.parquet` | customer | Lifetime stats + cohort |
| `audit/uci_monthly_acquisition.parquet` | month | Acquisition trend |
| `audit/uci_*` | panels / whale | Five Lenses + consulting exports |

### From notebook `01` (weekly Hardie-style split)

| Artifact | Role |
|----------|------|
| `models/01_clv_bgf`, `01_clv_ggf` | Fitted lifetimes models (`freq="W"`) |
| `models/01_clv_best.joblib` | Bundle + metrics + horizons |
| `models/01_clv_customer_scores.parquet` | `p_alive`, multi-horizon `expected_purchases_*w` / `expected_clv_*w`, `segment` |

## Raw inputs (`data/raw/`)

| Path | Source |
|------|--------|
| `online_retail_ii/online_retail_II.xlsx` | [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (auto-download) |
| `online_retail_ii/eu_transactions.parquet` | Cached clean EU lines |

No holiday / macro / weather / synthetic CRM enrichment.

## Build steps

1. `load_online_retail_ii` — download xlsx if needed, clean, EU filter, cache  
2. Write fact (source columns only) + dim rollups  

### Order → CLV

```
fact lines → orders (sum line_total) → cal/holdout RFM
  → BG/NBD + Gamma-Gamma → expected_clv
```

## Notebooks

| Notebook | Role |
|----------|------|
| `get_UCI_data.ipynb` | Interactive rebuild (same as pipeline) |
| `EDA.ipynb` | Overview / concentration |
| `00-customer-base-audit.ipynb` | Five Lenses audit |
| `01-clv.ipynb` | Probabilistic CLV |

## Rebuild

```powershell
pip install -r requirements.txt
python scripts/uci_pipeline.py
python scripts/smoke_test_modeling.py
```

```powershell
python scripts/generate_audit_notebook.py
python scripts/generate_clv_notebook.py
python scripts/generate_eda_notebook.py
```
