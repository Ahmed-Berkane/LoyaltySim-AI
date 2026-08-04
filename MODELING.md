# Modeling (CLV)

**Sequence:** customer-base audit → weekly noncontractual CLV. Do not train CLV before the audit.

**Spine:** UCI Online Retail II (EU), `python scripts/uci_pipeline.py`.

**Consulting map:** Customer Value Assessment (notebooks-first).

| Phase | Where |
|-------|--------|
| 1–2 Business / data | Pipeline + EDA |
| 3 Customer master | `00` → `audit/uci_customer_master.parquet` |
| 4 Dual descriptive lenses | `00` Part A (Acquisition…Heterogeneity) + Part B (Hardie/Ross) |
| 5 Cohorts | `00` Lenses 3–5 |
| 6 Access data + explore | `01` §§ 1–2 |
| 7 Protocol + RFM | `01` §§ 3–4 |
| 8 Val BG/NBD + GG → £ scale | `01` §§ 5–6 |
| 9 Test BG/NBD + GG | `01` §§ 7–8 |
| 10 CLV + £ gate + ranking | `01` §9 |
| 11 Segments × scenarios | `01` §§ 10–11 |
| 12 Artifacts | `01` §12 |
| 13 Enterprise foil (optional) | `02` — HGB / Hardie+HGB blend vs Hardie |

`01` is **stepwise** (granular `clv_weekly` API in-notebook). Smoke tests may still call `run_three_way` for the same protocol.

---

## 0 · Customer-base audit

| | |
|---|---|
| **Data** | Order spine from `uci_fact_transactions` |
| **Goal** | Descriptive health on the **customer × time** face |
| **Notebook** | `Notebooks/00-customer-base-audit.ipynb` |
| **Module** | `scripts/customer_base_audit.py` |

**Part A (consulting views):** Acquisition, Frequency, Monetary, Duration, Heterogeneity.

**Part B (Fader / Hardie / Ross):** single-period heterogeneity, period vs period, cohort evolution, cohort comparison, overall base health.

Revenue-only. Exports under `data/modeling/audit/` (orders, master, lifetime, acquisition, panels, whale).

---

## 1 · CLV (noncontractual, weekly)

| | |
|---|---|
| **Data** | `audit/uci_orders.parquet` (order spend) |
| **Time unit** | **Weeks** (`freq="W"`) |
| **Split** | **45% fit / 20% validation / 35% test** |
| **Acceptance gate** | Test aggregate \|pred − actual\| / actual **≤ 5%** |
| **Scale** | `val_actual / val_pred` only — **test labels never used** |
| **Leakage audit** | `clv_weekly.audit_no_leakage()` (cal span, known customers, scale identity) |
| **Models** | Classic BG/NBD (Fader–Hardie–Lee) + Gamma-Gamma; MLP trained on **val**, scored on **test** |
| **Purchase path** | Stationary weekly BG/NBD — **no** seasonality overlays (Hardie default). |
| **Monetary** | Winsorized at p99 on **calibration** monetary only (separate from purchases) |
| **CLV formula** | `E[purchases]_BG/NBD × E[avg spend]_GG × val_scale` |
| **Module** | `scripts/clv_weekly.py` |
| **Notebook** | `Notebooks/01-clv.ipynb` |
| **Artifacts** | `models/01_clv_*` (saved only if gate passes) |

If the gate fails, treat the model as **not shippable** for £ totals (`assert_accepted`).

**Not in scope:** Streamlit, margin/discounting, Dunnhumby, coupons, uplift/NBA.

---

## 2 · Enterprise foil (optional)

| | |
|---|---|
| **Keeps** | `00` + `01` unchanged (Hardie remains default equity) |
| **Notebook** | `Notebooks/02-clv-enterprise.ipynb` |
| **Module** | `scripts/clv_enterprise.py` |
| **Idea** | Marketplace-style tabular CLV: rich order/line features + HistGradientBoosting + Empirical-Bayes country/cohort pooling |
| **Protocol** | Same 45/20/35, val-only scale, ≤5% £ gate, no test leakage |
| **Target** | Gross holdout £ (not contribution margin — UCI has no COGS/shipping) |
| **Out of scope here** | PyMC Bayesian BG/NBD MCMC, Transformers/CASPR embeddings, ad bidding APIs |

Compare head-to-head with `01` in the notebook. Ship `01` for consulting narratives unless `02` clearly wins ranking *and* clears the gate.

---

## Smoke test

```powershell
python scripts/generate_audit_notebook.py
python scripts/generate_clv_notebook.py
python scripts/generate_enterprise_clv_notebook.py
python scripts/smoke_test_modeling.py
```

Runs audit export (incl. master) + weekly Hardie-faithful BG/NBD + GG fit/score/segments. Enterprise foil is optional (`02`).
