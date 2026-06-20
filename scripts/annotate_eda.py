"""Insert goal + finding markdown cells into EDA.ipynb."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "Notebooks" / "EDA.ipynb"

S = {
    "n_lines": 9994,
    "n_orders": 5009,
    "n_customers": 793,
    "date_min": "2016-01-03",
    "date_max": "2019-12-30",
    "total_sales": 2_297_201,
    "margin_pct": 12.5,
    "null_cols": "postal_code",
    "null_pct": 0.11,
    "monthly_pct_change": 489,
    "best_month": "2019-11",
    "worst_month": "2016-02",
    "gini": 0.44,
    "top10_share": 30.6,
    "top20_share": 48.0,
    "zero_disc_margin": 29.5,
    "high_disc_margin": -4.0,
    "pct_lines_discounted": 52.0,
    "champions_pct": 22.1,
    "at_risk_pct": 18.3,
    "hibernating_pct": 21.7,
    "tier_aov_silver": 475,
    "tier_aov_platinum": 430,
    "recency_churn_corr": 0.80,
    "holiday_lift": 9.0,
    "holiday_p": 0.44,
    "retail_lift": 16.3,
    "retail_p": 0.46,
    "rain_aov": 464,
    "dry_aov": 454,
    "rain_p": 0.71,
    "best_cat": "Technology",
    "best_cat_margin": 17.4,
    "worst_cat": "Furniture",
    "worst_cat_margin": 2.5,
    "telco_churn": 26.5,
    "m2m_churn": 42.7,
    "two_yr_churn": 2.8,
    "churned_tenure": 10,
    "retained_tenure": 38,
    "instacart_users": 206_209,
    "instacart_median_orders": 10,
    "instacart_reorder": 90,
    "payday_pct": 38.0,
    "holiday_pct": 3.8,
    "eng_platinum": 0.715,
    "eng_bronze": 0.383,
}


def md(text: str) -> dict:
    lines = text.strip().split("\n")
    return {
        "cell_type": "markdown",
        "id": None,
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
    }


def is_annotation_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    src = "".join(cell.get("source", []))
    return src.startswith("▶ **What this cell does:**") or src.startswith("📌 **Finding:**")


def is_redundant_takeaway(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    src = "".join(cell.get("source", []))
    return src.strip().startswith("**Takeaway:** Heavy discount bands")


ANNOTATIONS: dict[str, tuple[str, str]] = {
    "from __future__ import annotations": (
        "Import libraries, set plot theme/colors, and define helpers (`find_project_root`, `fmt_usd`, `cohens_d`, `gini_coefficient`) used across the notebook.",
        "Environment ready. Project root resolves to the repo containing `scripts/build_datasets.py`; palette and formatters are shared so dollar axes and effect sizes stay consistent in later sections.",
    ),
    "# ── Load all modeling Parquet": (
        "Load every `data/modeling/*.parquet` table from `build_datasets.py`, align `dim` with the feature store, and build an order-level `orders` view from `fact_transactions`.",
        f"Superstore spine = **{S['n_lines']:,} line items** → **{S['n_orders']:,} orders** → **{S['n_customers']:,} customers** ({S['date_min']} → {S['date_max']}). Total sales **${S['total_sales']:,.0f}** at **{S['margin_pct']:.1f}%** portfolio margin. Optional universes (telco, uplift, NBA, Online Retail, Instacart) load when present — customer IDs are **not** cross-joinable by design.",
    ),
    "# Missingness heatmap": (
        "Flag columns in `fact_transactions` with any nulls and visualize missingness % so we know what cannot be used without imputation.",
        f"Only **`{S['null_cols']}`** has gaps (**{S['null_pct']:.2f}%** of rows). Core modeling fields (sales, profit, dates, weather, calendar flags) are complete — safe to train without row drops on the transaction spine.",
    ),
    "# ── 2a Monthly revenue": (
        "Plot monthly sales and profit over 2016–2019 and overlay CPI context to separate real growth from inflation narrative.",
        f"Revenue accelerates from early 2016 trough (**{S['worst_month']}**) to peak **{S['best_month']}** — last month ~**{S['monthly_pct_change']:.0f}%** above the first tracked month. Profit tracks sales but margin wobble in late years signals mix/discount pressure, not a flat growth story.",
    ),
    "# ── 2b Customer concentration": (
        "Quantify revenue inequality via Lorenz curve, Gini, and Pareto chart — answers how dependent the business is on a few accounts.",
        f"Gini ≈ **{S['gini']:.2f}** (moderate concentration). Top **10%** of customers ≈ **{S['top10_share']:.1f}%** of sales; top **20%** ≈ **{S['top20_share']:.0f}%**. Segmentation and CLV should prioritize the high-value tail without ignoring the mid-tier.",
    ),
    "# ── 2c Discount vs margin": (
        "Bucket line-level discounts and compare revenue share vs profit margin to see where promotions help top-line but erode profit.",
        f"**{S['pct_lines_discounted']:.0f}%** of lines carry a discount. Zero-discount lines run **{S['zero_disc_margin']:.1f}%** margin; lines with **≥20%** discount show **{S['high_disc_margin']:.1f}%** margin (often negative contribution). Uplift/NBA models must optimize **incremental profit**, not response rate alone.",
    ),
    "# ── 3a RFM scatter": (
        "Visualize recency vs frequency (bubble size = monetary) using `rfm_segment` colors from the feature store — sanity-check cluster separation before segmentation modeling.",
        f"**Champions ({S['champions_pct']:.1f}%)** sit in high-F/low-R corner; **At Risk ({S['at_risk_pct']:.1f}%)** and **Hibernating ({S['hibernating_pct']:.1f}%)** dominate the high-recency tail. Clear visual separation supports KMeans/GMM in `01-segmentation.ipynb`.",
    ),
    "# ── 3b Segment × region": (
        "Heatmap revenue by Superstore `segment` × `region` to locate category–geography profit pools.",
        "Corporate and Consumer segments lead in Central/West; Technology and Office Supplies skew East/South. Region effects are real but secondary to customer-level RFM — use region as a covariate, not the primary segment axis.",
    ),
    "# ── 3c Loyalty tier vs AOV": (
        "Compare average order value by synthetic CRM tier (Bronze → Platinum) — validates tier assignment logic from `build_datasets.py`.",
        f"Tier AOVs cluster **$430–$475** — **Silver highest (${S['tier_aov_silver']})**, **Platinum slightly lower (${S['tier_aov_platinum']})** because tier reflects cumulative engagement, not single-basket size. Do not treat tier as a monotonic spend proxy in NBA ranking.",
    ),
    "# Feature store already loaded": (
        "Summarize `customer_features.parquet` — segment counts, feature distributions, and label prevalence (`inactivity_churn_label`) without re-engineering features in-notebook.",
        f"**793** customers × **16+** engineered features. Inactivity churn proxy (180-day rule) flags a minority; **`recency_days` ↔ `inactivity_churn_label` correlation ≈ {S['recency_churn_corr']:.2f}** — strongest single predictor on the Superstore spine.",
    ),
    "# ── 4a Core features: RFM map": (
        "Plot RFM heatmap (avg R/F/M by `rfm_segment`) and scatter basket size vs discount dependency to link behavioral clusters to promotion reliance.",
        "Champions show high F/M and low discount dependency; At Risk / Hibernating clusters pull high `discount_dependency` and lower basket values. The discount-seeker segment in `01-segmentation.ipynb` should align with the upper-right of the dependency scatter.",
    ),
    "# ── 4b Behavioral + Temporal": (
        "Box plots of engagement, email, app, seasonality, payday, and holiday features by tier; then population seasonality bar chart from order-level calendar flags.",
        f"Engagement rises sharply by tier — **Platinum {S['eng_platinum']:.3f}** vs **Bronze {S['eng_bronze']:.3f}**. **{S['payday_pct']:.0f}%** of orders fall in month-start/end payday windows; only **{S['holiday_pct']:.1f}%** on federal holidays — payday timing matters more than holidays for this catalog.",
    ),
    "# ── 4c Loyalty features": (
        "Plot tier progression, reward redemption, and churn inertia by tier; correlate engineered features with the inactivity churn proxy.",
        f"`churn_inertia_score` and `recency_days` dominate churn-proxy correlation; loyalty mechanics (redemption, tier speed) add signal for Platinum/Gold. Superstore churn label is **derived**, not observed — use **`telco_customers.parquet`** for supervised churn benchmarking.",
    ),
    "# ── 4d Feature catalog": (
        "Print the feature catalog grouped by Core / Behavioral / Temporal / Loyalty and show the numeric correlation matrix to spot collinearity before modeling.",
        "R/F/M scores correlate with each other and with `RFM_score` (expected). `discount_dependency` ↔ `avg_discount` partially overlap — tree models tolerate this; for linear/MLP models consider dropping one. No feature pair exceeds |r|>0.85 except within the RFM family.",
    ),
    "# ── 4a Calendar lift: federal holidays": (
        "Compare mean order value on US federal holidays and retail spending days vs ordinary days (Welch t-test + Cohen's d).",
        f"Holiday AOV **+{S['holiday_lift']:.1f}%** vs non-holiday; retail-day AOV **+{S['retail_lift']:.1f}%** — directionally positive but **p≈{S['holiday_p']:.2f} / {S['retail_p']:.2f}** (not significant at α=0.05 with n≈5k orders). Calendar flags are weak standalone predictors; better as NBA timing features.",
    ),
    "# ── 4b Weather — temperature": (
        "Relate city-level temperature and rain flags (from cached Open-Meteo) to order-level AOV via binned temperature curve and rain vs dry t-test.",
        f"Rain-day AOV **${S['rain_aov']:.0f}** vs dry **${S['dry_aov']:.0f}** (p≈**{S['rain_p']:.2f}** — not significant). Temperature bins show mild mid-range uplift; weather explains <5% of AOV variance. Keep as enrichment, not a primary lever.",
    ),
    "# ── 4c Category mix": (
        "Rank categories by sales, profit, and margin % to see where the business earns vs bleeds.",
        f"**{S['best_cat']}** leads margin (**{S['best_cat_margin']:.1f}%**); **{S['worst_cat']}** trails at **{S['worst_cat_margin']:.1f}%** despite large revenue share. NBA offers on Furniture should be margin-aware; Technology upsell has headroom.",
    ),
    "# ── 5a Churn rate by contract": (
        "Compute churn rate with Wilson confidence intervals by `Contract` type on the IBM Telco benchmark (labeled ground truth).",
        f"Overall churn **{S['telco_churn']:.1f}%**. **Month-to-month {S['m2m_churn']:.1f}%** vs **Two-year {S['two_yr_churn']:.1f}%** — contract length is the dominant lever. This is the labeled universe for `02-churn-prediction.ipynb` (not joinable to Superstore IDs).",
    ),
    "# ── 5b Tenure distribution": (
        "KDE overlay of `tenure` for churned vs retained Telco customers — early-life churn pattern check.",
        f"Median tenure **{S['churned_tenure']:.0f} months** (churned) vs **{S['retained_tenure']:.0f} months** (retained). Most churn happens in the first year — aligns with using tenure + contract + charges as top features in the churn notebook.",
    ),
    "# ── 5c Monthly charges vs churn": (
        "Violin plot of `MonthlyCharges` split by churn flag — price sensitivity vs voluntary exit.",
        "Churned customers skew slightly higher monthly charges (more add-ons / fiber bundles) but overlap is large. Charges alone won't separate classes; interaction with contract type and tenure drives AUC ~0.83 in the trained XGBoost model.",
    ),
    "if instacart is not None:": (
        "If Kaggle Instacart Parquet exists, histogram user order counts and reorder rates — scale reference for sequence/basket models.",
        f"**{S['instacart_users']:,}** users loaded; median **{S['instacart_median_orders']:.0f}** orders/user, median reorder rate **{S['instacart_reorder']:.0f}%** — far denser than Superstore's 793-customer spine. Use Instacart for basket/reorder prototypes; use Online Retail for CLV scale.",
    ),
}


def match_annotation(source: str) -> tuple[str, str] | None:
    for key, pair in ANNOTATIONS.items():
        if source.lstrip().startswith(key):
            return pair
    return None


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    base_cells = [
        c for c in nb["cells"]
        if not is_annotation_cell(c) and not is_redundant_takeaway(c)
    ]

    new_cells: list[dict] = []
    for cell in base_cells:
        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        source = "".join(cell.get("source", []))
        ann = match_annotation(source)
        if ann is None:
            new_cells.append(cell)
            continue

        before, after = ann
        new_cells.append(md(f"▶ **What this cell does:** {before}"))
        new_cells.append(cell)
        new_cells.append(md(f"📌 **Finding:** {after}"))

    old_count = len(nb["cells"])
    nb["cells"] = new_cells
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Annotated EDA: {old_count} -> {len(new_cells)} cells")


if __name__ == "__main__":
    main()
