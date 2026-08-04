# CLV

Consulting-style **Customer Value Assessment** on [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): **audit first**, then **weekly** BG/NBD + Gamma-Gamma CLV (Hardie / CDNOW time unit).

Silent attrition (no cancel date). Revenue-only (no COGS on this dataset).

## Setup

```powershell
cd C:\Users\ahmed\Desktop\Projects\CLV
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build data

```powershell
python scripts/uci_pipeline.py
```

| Output | Grain |
|--------|--------|
| `data/modeling/uci_fact_transactions.parquet` | Invoice line (clean UCI only) |
| `data/modeling/uci_dim_customers.parquet` | Customer rollups |

Optional walkthrough: `Notebooks/get_UCI_data.ipynb`.

## Run order

```
uci_pipeline.py
  → EDA.ipynb                         (optional overview)
  → 00-customer-base-audit.ipynb      ★ Dual lenses + master table → audit/
  → 01-clv.ipynb                      ★ Weekly BG/NBD + GG → models/01_clv_*
  → 02-clv-enterprise.ipynb           (optional) rich-feature HGB foil → models/02_clv_*
```

```powershell
python scripts/generate_audit_notebook.py
python scripts/generate_clv_notebook.py
python scripts/generate_enterprise_clv_notebook.py
python scripts/smoke_test_modeling.py
```

## What you get

| Deliverable | Notes |
|-------------|--------|
| Customer master | `audit/uci_customer_master.parquet` |
| Dual audit | Consulting views (A1–A5) + Hardie/Ross lenses (1–5) |
| Weekly CLV | 45/20/35 fit/val/test; classic BG/NBD + GG; **test aggregate £ error must be ≤5%** |
| Segments | Predicted CLV × expected purchases (activity) |
| Scenarios | Illustrative equity what-ifs (not causal) |
| Enterprise foil (`02`) | Feature-rich HGB + EB pooling vs Hardie; same gate; no clickstreams/transformers |

See [MODELING.md](MODELING.md) for the full consulting phase map.

## Layout

```
scripts/uci_pipeline.py
scripts/customer_base_audit.py
scripts/generate_audit_notebook.py
scripts/generate_clv_notebook.py
scripts/generate_enterprise_clv_notebook.py
scripts/clv_weekly.py
scripts/clv_enterprise.py
scripts/modeling_extensions.py
Notebooks/00-customer-base-audit.ipynb
Notebooks/01-clv.ipynb
Notebooks/02-clv-enterprise.ipynb
Notebooks/EDA.ipynb
data/modeling/uci_*.parquet
data/modeling/audit/
models/01_clv_*
models/02_clv_*
```

See [DATA_PIPELINE.md](DATA_PIPELINE.md) and [MODELING.md](MODELING.md).
