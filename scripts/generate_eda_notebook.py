"""Generate Notebooks/EDA.ipynb — overview EDA for the CLV project (UCI spine)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Notebooks" / "EDA.ipynb"


def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }


def _normalize_cell_source(text: str) -> str:
    lines = text.strip("\n").splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return "\n"
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in nonempty]
    if indents[0] == 0 and len(indents) > 1 and min(indents[1:]) > 0:
        pad = min(indents[1:])
    else:
        pad = min(indents)
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
        elif len(ln) >= pad and ln[:pad].strip() == "":
            out.append(ln[pad:])
        else:
            out.append(ln.lstrip(" ") if ln.startswith(" ") else ln)
    return "\n".join(out).strip("\n") + "\n"


def md(text: str, cid: str = "") -> dict:
    cleaned = _normalize_cell_source(text)
    return {"cell_type": "markdown", "id": cid or None, "metadata": {}, "source": cleaned.splitlines(keepends=True)}


def code(text: str, cid: str = "") -> dict:
    cleaned = _normalize_cell_source(text)
    return {
        "cell_type": "code",
        "id": cid or None,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": cleaned.splitlines(keepends=True),
    }


SETUP = r'''import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)


def find_project_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in (path, *path.parents):
        if (candidate / "scripts" / "uci_pipeline.py").exists():
            return candidate
    return path


PROJECT_ROOT = find_project_root()
DATA = PROJECT_ROOT / "data" / "modeling"
PALETTE = {"primary": "#1f4e79", "accent": "#c0392b", "neutral": "#7f8c8d"}
print("Config loaded ·", DATA)
'''

LOAD = r'''def load(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run: python scripts/uci_pipeline.py")
    return pd.read_parquet(path)

fact = load("uci_fact_transactions.parquet")
fact["order_date"] = pd.to_datetime(fact["order_date"])
dim = load("uci_dim_customers.parquet")
orders = (
    fact.groupby(["order_id", "customer_id", "country"], as_index=False)
    .agg(order_revenue=("line_total", "sum"), order_lines=("line_total", "size"), order_date=("order_date", "min"))
)
print(f"Lines: {len(fact):,} · Orders: {orders['order_id'].nunique():,} · Customers: {fact['customer_id'].nunique():,}")
print(f"Span: {fact['order_date'].min().date()} → {fact['order_date'].max().date()}")
print(f"Total revenue: £{fact['line_total'].sum():,.0f}")
fact.head(3)
'''

cells = [
    md("""# CLV — UCI Online Retail II EDA

**Prerequisite:** `python scripts/uci_pipeline.py`

Noncontractual retail spine (~5.8k customers, ~800k EU invoice lines).

| Parquet | Grain | Use |
|---------|-------|-----|
| `uci_fact_transactions` | Invoice line | Source for **order** spine |
| `uci_dim_customers` | Customer | Lifetime rollups / sanity checks |

**Next:** `00-customer-base-audit.ipynb` → `01-clv.ipynb` (BG/NBD + Gamma-Gamma on orders).
""", "intro"),
    code(SETUP, "setup"),
    code(LOAD, "load"),
    md("## 1 · Concentration (orders)", "h1"),
    code("""cust = orders.groupby("customer_id")["order_revenue"].sum().sort_values(ascending=False)
cum = cust.cumsum() / cust.sum()
pct_cust = np.arange(1, len(cust) + 1) / len(cust)
half = float(pct_cust[np.searchsorted(cum.values, 0.5)])
print(f"Customers for 50% of revenue: {half:.1%}")
print(f"One-order customers (lifetime): {(dim['total_orders'] == 1).mean():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(cust.clip(upper=cust.quantile(0.99)), bins=40, color=PALETTE["primary"], edgecolor="white")
axes[0].set_title("Customer lifetime spend")
axes[0].set_xlabel("£")
axes[1].plot(pct_cust * 100, cum * 100, color=PALETTE["accent"])
axes[1].axhline(50, ls=":", color="gray")
axes[1].set_title("Whale / Lorenz curve")
axes[1].set_xlabel("% customers (richest first)")
axes[1].set_ylabel("% revenue")
plt.tight_layout()
plt.show()
""", "conc"),
    md("## 2 · Next steps", "h2"),
    md("""| Notebook | Role |
|----------|------|
| `00-customer-base-audit.ipynb` | Five Lenses customer×time audit |
| `01-clv.ipynb` | BG/NBD + Gamma-Gamma expected CLV |

Do not model CLV at invoice-line grain.
""", "next"),
]

NB.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
print(f"Wrote {NB.relative_to(ROOT)}")
