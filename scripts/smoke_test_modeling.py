"""Smoke-test: audit + Hardie-faithful weekly BG/NBD CLV with ≤5% £ gate."""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "modeling"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def ok(name: str) -> None:
    print(f"  OK  {name}")


def run_00_audit() -> None:
    from customer_base_audit import build_orders, export_audit_summaries, load_fact_transactions

    fact = load_fact_transactions(DATA / "uci_fact_transactions.parquet")
    orders = build_orders(fact)
    paths = export_audit_summaries(ROOT, orders=orders)
    assert (DATA / "audit" / "uci_customer_master.parquet").exists()
    ok(f"00-audit ({len(paths)} summaries)")


def run_01_clv() -> None:
    from clv_weekly import AGG_ERROR_MAX, assert_accepted, assign_segments, audit_no_leakage, run_three_way
    from customer_base_audit import attach_acquisition, build_orders, load_fact_transactions

    audit_orders = DATA / "audit" / "uci_orders.parquet"
    if audit_orders.exists():
        orders = pd.read_parquet(audit_orders)
        orders["order_date"] = pd.to_datetime(orders["order_date"])
    else:
        fact = load_fact_transactions(DATA / "uci_fact_transactions.parquet")
        orders = attach_acquisition(build_orders(fact))

    result = run_three_way(orders)
    assert_accepted(result)
    leak = audit_no_leakage(result, orders)
    assert leak["passed"]
    assert leak["customers_first_seen_in_test"] == 0
    assert leak["test_labels_in_fit"] is False
    assert leak["test_labels_in_scale"] is False
    assert leak.get("seasonality_overlays") is False
    assert result.get("model_name") == "BG/NBD"

    summary = assign_segments(result["summary"], "expected_clv")
    assert result["agg_error"] <= AGG_ERROR_MAX
    assert summary["segment"].nunique() >= 3
    assert "val_summary" in result
    assert abs(result["val_actual"] - result["test_actual"]) > 1.0
    assert "pred_cum_purchases" in result
    assert len(result["pred_cum_purchases"]) == result["t_horizon"]

    result["bgf"].save_model(str(MODELS / "01_clv_bgf"))
    result["ggf"].save_model(str(MODELS / "01_clv_ggf"))
    joblib.dump(
        {
            "accepted": True,
            "purchase_model": "BG/NBD",
            "agg_error_test": float(result["agg_error"]),
            "orders_agg_error_test": float(result["orders_agg_error"]),
            "agg_error_max": AGG_ERROR_MAX,
            "val_scale": float(result["scale"]),
            "horizon_weeks": result["t_horizon"],
            "fit_end": str(result["fit_end"].date()),
            "val_end": str(result["val_end"].date()),
            "obs_end": str(result["obs_end"].date()),
            "seasonality_overlays": False,
            "leakage_audit": leak,
        },
        MODELS / "01_clv_best.joblib",
    )
    summary.reset_index().to_parquet(MODELS / "01_clv_customer_scores.parquet", index=False)
    ok(
        f"01-clv ACCEPTED (BG/NBD · £|error|={result['agg_error']:.2%} · "
        f"purch|error|={result['orders_agg_error']:.1%} · scale={result['scale']:.3f}, n={len(summary):,})"
    )


def main() -> int:
    failed = []
    for fn in (run_00_audit, run_01_clv):
        try:
            fn()
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {exc}")
            failed.append(fn.__name__)
    if failed:
        print(f"\nFailed: {failed}")
        return 1
    print(f"\nAll {2 - len(failed)} CLV smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
