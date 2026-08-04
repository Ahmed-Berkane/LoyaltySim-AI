"""Weekly BG/NBD (Fader–Hardie–Lee) + separate Gamma-Gamma monetary layer.

Hardie-faithful purchase model:
  - time unit = weeks
  - inputs (x, t_x, T) via lifetimes calibration_and_holdout_data
  - conditional expected purchases E[Y(t) | history]
  - aggregate transaction tracking (stationary path — no seasonality overlays)

Monetary / £ CLV is a **separate** step (Gamma-Gamma), then optional val-only
revenue scale for the consulting ≤5% aggregate £ gate. Test labels never enter
fit or scaling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data

FREQ = "W"
FIT_FRAC = 0.45
VAL_FRAC = 0.20
# test = remainder (~0.35)
MONETARY_WINSOR_Q = 0.99
GG_PENALIZER = 0.01
BG_PENALIZERS = (0.0, 0.001, 0.01, 0.05, 0.1)
AGG_ERROR_MAX = 0.05  # 5% on unseen test aggregate revenue
HORIZON_WEEKS_CANDIDATES = (4, 13, 26, 52)


def split_dates(orders: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    min_d = pd.Timestamp(orders["order_date"].min())
    max_d = pd.Timestamp(orders["order_date"].max())
    span = int((max_d - min_d).days)
    fit_end = min_d + pd.Timedelta(days=int(span * FIT_FRAC))
    val_end = min_d + pd.Timedelta(days=int(span * (FIT_FRAC + VAL_FRAC)))
    return min_d, fit_end, val_end, max_d


def weeks_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(int(np.ceil((end - start).days / 7.0)), 1)


def _fit_bgnbd(frequency, recency, T) -> BetaGeoFitter:
    """Classic BG/NBD (Fader–Hardie–Lee), with light penalizer fallback."""
    import contextlib
    import io

    last_err: Exception | None = None
    freq = frequency.reset_index(drop=True)
    rec = recency.reset_index(drop=True)
    age = T.reset_index(drop=True)
    for pen in BG_PENALIZERS:
        try:
            bgf = BetaGeoFitter(penalizer_coef=pen)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                bgf.fit(freq, rec, age, verbose=False)
            n = min(50, len(freq))
            probe = bgf.conditional_expected_number_of_purchases_up_to_time(
                12, freq.iloc[:n], rec.iloc[:n], age.iloc[:n]
            )
            if not np.isfinite(probe).all() or float(np.nanmean(probe)) <= 1e-8:
                last_err = RuntimeError(f"degenerate BG/NBD fit at pen={pen}")
                continue
            return bgf
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"BG/NBD failed to converge: {last_err}")


def build_summary(
    orders: pd.DataFrame,
    cal_end: pd.Timestamp,
    obs_end: pd.Timestamp,
) -> pd.DataFrame:
    summary = calibration_and_holdout_data(
        orders[["customer_id", "order_date", "spend"]],
        customer_id_col="customer_id",
        datetime_col="order_date",
        calibration_period_end=cal_end,
        observation_period_end=obs_end,
        monetary_value_col="spend",
        freq=FREQ,
    )
    hold = orders[(orders["order_date"] > cal_end) & (orders["order_date"] <= obs_end)]
    actual_rev = hold.groupby("customer_id")["spend"].sum().rename("actual_holdout_revenue")
    actual_n = hold.groupby("customer_id")["order_id"].nunique().rename("actual_holdout_orders")
    summary = summary.join(actual_rev, how="left").join(actual_n, how="left")
    summary["actual_holdout_revenue"] = summary["actual_holdout_revenue"].fillna(0.0)
    summary["actual_holdout_orders"] = summary["actual_holdout_orders"].fillna(0.0)
    return summary


def fit_purchase_model(summary: pd.DataFrame) -> BetaGeoFitter:
    """Step 1 — BG/NBD on calibration RFM only."""
    return _fit_bgnbd(summary["frequency_cal"], summary["recency_cal"], summary["T_cal"])


def score_purchases(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    *,
    horizon_weeks: int,
) -> pd.DataFrame:
    """Hardie conditional expected purchases over the holdout horizon."""
    out = summary.copy()
    out["p_alive"] = bgf.conditional_probability_alive(
        out["frequency_cal"], out["recency_cal"], out["T_cal"]
    )
    horizons = [h for h in HORIZON_WEEKS_CANDIDATES if h <= horizon_weeks]
    if horizon_weeks not in horizons:
        horizons = sorted(set(horizons + [horizon_weeks]))
    for h in horizons:
        out[f"expected_purchases_{h}w"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            h, out["frequency_cal"], out["recency_cal"], out["T_cal"]
        )
    out["expected_purchases"] = out[f"expected_purchases_{horizon_weeks}w"]
    return out


def fit_score_monetary(
    summary: pd.DataFrame,
    *,
    ggf: GammaGammaFitter | None = None,
) -> tuple[pd.DataFrame, GammaGammaFitter, float]:
    """Step 2 — Gamma-Gamma average order value (separate from purchase process)."""
    out = summary.copy()
    gg = out[out["frequency_cal"] > 0].copy()
    mon_cap = float(gg["monetary_value_cal"].quantile(MONETARY_WINSOR_Q))
    gg["monetary_fit"] = gg["monetary_value_cal"].clip(upper=mon_cap)
    if ggf is None:
        ggf = GammaGammaFitter(penalizer_coef=GG_PENALIZER)
        ggf.fit(gg["frequency_cal"], gg["monetary_fit"])
    out["expected_avg_spend"] = np.nan
    out.loc[gg.index, "expected_avg_spend"] = ggf.conditional_expected_average_profit(
        gg["frequency_cal"], gg["monetary_fit"]
    )
    pop = float(gg["monetary_fit"].median())
    out["expected_avg_spend"] = out["expected_avg_spend"].fillna(pop)
    out["monetary_winsor_cap"] = mon_cap
    out["expected_clv"] = out["expected_purchases"] * out["expected_avg_spend"]
    for c in list(out.columns):
        if c.startswith("expected_purchases_") and c.endswith("w"):
            h = c[len("expected_purchases_") : -1]
            out[f"expected_clv_{h}w"] = out[c] * out["expected_avg_spend"]
    return out, ggf, mon_cap


def score_summary(
    summary: pd.DataFrame,
    *,
    horizon_weeks: int,
    bgf: BetaGeoFitter | None = None,
    ggf: GammaGammaFitter | None = None,
) -> tuple[pd.DataFrame, BetaGeoFitter, GammaGammaFitter, float]:
    """Convenience: purchases then monetary (Hardie then Fader/Hardie GG)."""
    if bgf is None:
        bgf = fit_purchase_model(summary)
    out = score_purchases(summary, bgf, horizon_weeks=horizon_weeks)
    out, ggf, mon_cap = fit_score_monetary(out, ggf=ggf)
    return out, bgf, ggf, mon_cap


def cumulative_expected_purchases(
    bgf: BetaGeoFitter,
    summary: pd.DataFrame,
    horizon_weeks: int,
) -> np.ndarray:
    """Paper-style aggregate path: sum_i E[Y(t) | history_i] for t = 1..H."""
    rfm = summary[["frequency_cal", "recency_cal", "T_cal"]].dropna()
    return np.array(
        [
            float(
                bgf.conditional_expected_number_of_purchases_up_to_time(
                    t, rfm["frequency_cal"], rfm["recency_cal"], rfm["T_cal"]
                ).sum()
            )
            for t in range(1, horizon_weeks + 1)
        ],
        dtype=float,
    )


def holdout_weekly_actual_orders(
    orders: pd.DataFrame,
    summary: pd.DataFrame,
    cal_end: pd.Timestamp,
    obs_end: pd.Timestamp,
    horizon_weeks: int,
) -> np.ndarray:
    """Weekly order counts for the fixed calibration cohort in the holdout window."""
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    cal_ids = summary.index
    hold = orders[
        (orders["order_date"] > cal_end)
        & (orders["order_date"] <= obs_end)
        & (orders["customer_id"].isin(cal_ids))
    ].copy()
    hold["hold_week"] = ((hold["order_date"] - cal_end).dt.days // 7) + 1
    weeks = np.arange(1, horizon_weeks + 1)
    return (
        hold.groupby("hold_week")["order_id"]
        .nunique()
        .reindex(weeks)
        .fillna(0)
        .to_numpy(dtype=float)
    )


def holdout_week_calendar_months(
    cal_end: pd.Timestamp,
    horizon_weeks: int,
) -> np.ndarray:
    """Calendar month (1–12) for each holdout week index 1..H (week mid-point)."""
    months = []
    for t in range(1, horizon_weeks + 1):
        mid = cal_end + pd.Timedelta(days=7 * (t - 1) + 3)
        months.append(int(mid.month))
    return np.asarray(months, dtype=int)


def fit_month_seasonality(
    actual_weekly: np.ndarray,
    pred_weekly: np.ndarray,
    months: np.ndarray,
    *,
    clip: tuple[float, float] = (0.5, 2.5),
) -> dict[int, float]:
    """Month multipliers = Σ actual / Σ stationary pred within each calendar month.

    Fit only on a validation holdout (no test labels). Missing months → 1.0.
    Clipped to avoid extreme sparse-month ratios.
    """
    actual_weekly = np.asarray(actual_weekly, dtype=float)
    pred_weekly = np.asarray(pred_weekly, dtype=float)
    months = np.asarray(months, dtype=int)
    factors: dict[int, float] = {m: 1.0 for m in range(1, 13)}
    for m in range(1, 13):
        mask = months == m
        if not mask.any():
            continue
        pred_sum = float(pred_weekly[mask].sum())
        act_sum = float(actual_weekly[mask].sum())
        if pred_sum <= 1e-8:
            continue
        factors[m] = float(np.clip(act_sum / pred_sum, clip[0], clip[1]))
    return factors


def apply_month_seasonality(
    pred_weekly: np.ndarray,
    months: np.ndarray,
    factors: dict[int, float],
) -> np.ndarray:
    """Scale stationary weekly expected flow by calendar-month factors."""
    pred_weekly = np.asarray(pred_weekly, dtype=float)
    months = np.asarray(months, dtype=int)
    mult = np.array([factors.get(int(m), 1.0) for m in months], dtype=float)
    return pred_weekly * mult


def seasonal_purchase_lift(pred_weekly: np.ndarray, seasonal_weekly: np.ndarray) -> float:
    """Aggregate lift so customer E[purchases] can be rescaled without changing ranks."""
    base = float(np.asarray(pred_weekly, dtype=float).sum())
    seas = float(np.asarray(seasonal_weekly, dtype=float).sum())
    if base <= 1e-8:
        return 1.0
    return seas / base


def evaluate_seasonal_overlay(
    orders: pd.DataFrame,
    *,
    bgf_val: BetaGeoFitter,
    s_val: pd.DataFrame,
    bgf: BetaGeoFitter,
    s_test: pd.DataFrame,
    fit_end: pd.Timestamp,
    val_end: pd.Timestamp,
    obs_end: pd.Timestamp,
    hw_val: int,
    hw_test: int,
    scale_stationary: float,
) -> dict[str, Any]:
    """Val-fit month seasonality; score test purchases + £ vs stationary baseline.

    Classic BG/NBD stays stationary; this is a *post-hoc overlay* (Hardie-style
    seasonal covariates idea, simplified to month multipliers on the aggregate path).
    Factors are estimated on the validation window only.
    """
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    pred_cum_val = cumulative_expected_purchases(bgf_val, s_val, hw_val)
    pred_weekly_val = np.diff(pred_cum_val, prepend=0.0)
    actual_weekly_val = holdout_weekly_actual_orders(orders, s_val, fit_end, val_end, hw_val)
    months_val = holdout_week_calendar_months(fit_end, hw_val)
    factors = fit_month_seasonality(actual_weekly_val, pred_weekly_val, months_val)
    seasonal_weekly_val = apply_month_seasonality(pred_weekly_val, months_val, factors)
    lift_val = seasonal_purchase_lift(pred_weekly_val, seasonal_weekly_val)

    # Val £ scale under seasonal purchase totals (same GG spend)
    val_pred_seas = float((s_val["expected_purchases"] * lift_val * s_val["expected_avg_spend"]).sum())
    val_actual = float(s_val["actual_holdout_revenue"].sum())
    if val_pred_seas <= 0 or val_actual <= 0:
        raise RuntimeError("Seasonal validation window has zero predicted or actual revenue")
    scale_seas = val_actual / val_pred_seas

    pred_cum_test = cumulative_expected_purchases(bgf, s_test, hw_test)
    pred_weekly_test = np.diff(pred_cum_test, prepend=0.0)
    actual_weekly_test = holdout_weekly_actual_orders(orders, s_test, val_end, obs_end, hw_test)
    months_test = holdout_week_calendar_months(val_end, hw_test)
    seasonal_weekly_test = apply_month_seasonality(pred_weekly_test, months_test, factors)
    lift_test = seasonal_purchase_lift(pred_weekly_test, seasonal_weekly_test)

    orders_pred_stat = float(pred_weekly_test.sum())
    orders_pred_seas = float(seasonal_weekly_test.sum())
    orders_actual = float(actual_weekly_test.sum())
    orders_err_stat = abs(orders_pred_stat - orders_actual) / orders_actual if orders_actual else np.inf
    orders_err_seas = abs(orders_pred_seas - orders_actual) / orders_actual if orders_actual else np.inf

    test_actual = float(s_test["actual_holdout_revenue"].sum())
    # Stationary £ already scaled outside; recompute seasonal £ from raw purchases × spend
    raw_purch = s_test["expected_purchases"]
    spend = s_test["expected_avg_spend"]
    clv_stat = raw_purch * spend * scale_stationary
    clv_seas = raw_purch * lift_test * spend * scale_seas
    pound_err_stat = abs(float(clv_stat.sum()) - test_actual) / test_actual if test_actual else np.inf
    pound_err_seas = abs(float(clv_seas.sum()) - test_actual) / test_actual if test_actual else np.inf

    rank_df = pd.DataFrame(
        {
            "clv_stat": clv_stat,
            "clv_seas": clv_seas,
            "actual": s_test["actual_holdout_revenue"],
        }
    )
    sp_stat = float(rank_df["clv_stat"].corr(rank_df["actual"], method="spearman"))
    sp_seas = float(rank_df["clv_seas"].corr(rank_df["actual"], method="spearman"))

    better_orders = bool(orders_err_seas < orders_err_stat - 1e-12)
    better_pounds = bool(pound_err_seas < pound_err_stat - 1e-12)
    better_rank = bool(
        (sp_seas if sp_seas == sp_seas else -1) > (sp_stat if sp_stat == sp_stat else -1)
    )

    return {
        "factors": factors,
        "lift_val": lift_val,
        "lift_test": lift_test,
        "scale_stationary": float(scale_stationary),
        "scale_seasonal": float(scale_seas),
        "pred_weekly_test": pred_weekly_test,
        "seasonal_weekly_test": seasonal_weekly_test,
        "actual_weekly_test": actual_weekly_test,
        "months_test": months_test,
        "pred_cum_seasonal": np.cumsum(seasonal_weekly_test),
        "orders_agg_error_stationary": float(orders_err_stat),
        "orders_agg_error_seasonal": float(orders_err_seas),
        "pound_agg_error_stationary": float(pound_err_stat),
        "pound_agg_error_seasonal": float(pound_err_seas),
        "spearman_stationary": float(sp_stat) if sp_stat == sp_stat else None,
        "spearman_seasonal": float(sp_seas) if sp_seas == sp_seas else None,
        "better_orders": better_orders,
        "better_pounds": better_pounds,
        "better_rank": better_rank,
        "seasonal_wins": bool(better_orders or better_pounds),
        "expected_clv_seasonal": clv_seas,
    }


def run_three_way(orders: pd.DataFrame) -> dict[str, Any]:
    """Fit / validate-scale / test.

    Purchase model follows Fader–Hardie–Lee BG/NBD (weekly, no seasonality).
    £ CLV = E[purchases] × E[spend]; val-only revenue scale for the ≤5% gate.
    """
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    min_d, fit_end, val_end, max_d = split_dates(orders)

    hw_val = weeks_between(fit_end, val_end)
    hw_test = weeks_between(val_end, max_d)

    s_val = build_summary(orders, fit_end, val_end)
    s_val, bgf_val, ggf_val, _ = score_summary(s_val, horizon_weeks=hw_val)
    val_actual = float(s_val["actual_holdout_revenue"].sum())
    val_pred = float(s_val["expected_clv"].sum())
    if val_pred <= 0 or val_actual <= 0:
        raise RuntimeError("Validation window has zero predicted or actual revenue")
    scale = val_actual / val_pred

    s_test = build_summary(orders, val_end, max_d)
    s_test, bgf, ggf, mon_cap = score_summary(s_test, horizon_weeks=hw_test)
    s_test["expected_clv_raw"] = s_test["expected_clv"]
    s_test["expected_clv"] = s_test["expected_clv_raw"] * scale
    for h in HORIZON_WEEKS_CANDIDATES:
        c = f"expected_clv_{h}w"
        if c in s_test.columns:
            s_test[c] = s_test[c] * scale
    s_test[f"expected_clv_{hw_test}w"] = s_test["expected_clv"]
    s_test["expected_clv_calibrated"] = s_test["expected_clv"]

    test_actual = float(s_test["actual_holdout_revenue"].sum())
    test_pred = float(s_test["expected_clv"].sum())
    agg_error = abs(test_pred - test_actual) / test_actual if test_actual else np.inf
    accepted = bool(agg_error <= AGG_ERROR_MAX)

    # Hardie transaction metrics on test window (unscaled purchase expectations)
    test_orders_actual = float(s_test["actual_holdout_orders"].sum())
    test_orders_pred = float(s_test["expected_purchases"].sum())
    orders_agg_error = (
        abs(test_orders_pred - test_orders_actual) / test_orders_actual
        if test_orders_actual
        else np.inf
    )
    pred_cum_purchases = cumulative_expected_purchases(bgf, s_test, hw_test)

    out = {
        "orders": orders,
        "min_date": min_d,
        "fit_end": fit_end,
        "val_end": val_end,
        "obs_end": max_d,
        "hw_val": hw_val,
        "hw_test": hw_test,
        "t_horizon": hw_test,
        "scale": scale,
        "val_actual": val_actual,
        "val_pred": val_pred,
        "val_summary": s_val,
        "test_actual": test_actual,
        "test_pred": test_pred,
        "agg_error": agg_error,
        "accepted": accepted,
        "summary": s_test,
        "bgf": bgf,
        "ggf": ggf,
        "bgf_val": bgf_val,
        "ggf_val": ggf_val,
        "monetary_winsor_cap": mon_cap,
        "agg_error_max": AGG_ERROR_MAX,
        "test_orders_actual": test_orders_actual,
        "test_orders_pred": test_orders_pred,
        "orders_agg_error": orders_agg_error,
        "pred_cum_purchases": pred_cum_purchases,
        "model_name": "BG/NBD",
    }
    out["leakage_audit"] = audit_no_leakage(out, orders)
    return out


def assert_accepted(result: dict[str, Any]) -> None:
    if not result["accepted"]:
        raise AssertionError(
            f"MODEL REJECTED: unseen test aggregate |error|={result['agg_error']:.1%} "
            f"> {result['agg_error_max']:.0%} "
            f"(pred=GBP {result['test_pred']:,.0f}, actual=GBP {result['test_actual']:,.0f}, "
            f"val_scale={result['scale']:.3f}). "
            "Do not ship ranking equity totals."
        )


def assign_segments(summary: pd.DataFrame, seg_clv_col: str) -> pd.DataFrame:
    out = summary.copy()
    act_cut = float(out["expected_purchases"].median())
    clv_band = pd.qcut(out[seg_clv_col].rank(method="first"), 3, labels=["Low", "Mid", "High"])
    activity_band = np.where(out["expected_purchases"] >= act_cut, "High", "Low")
    labels = []
    for c, a in zip(clv_band, activity_band):
        if c == "High" and a == "High":
            labels.append("VIP retention")
        elif c == "High" and a == "Low":
            labels.append("Win-back")
        elif c == "Mid" and a == "High":
            labels.append("Upsell")
        elif c == "Low" and a == "High":
            labels.append("Low-cost nurture")
        else:
            labels.append("Minimize spend")
    out["clv_band"] = clv_band.astype(str)
    out["activity_band"] = activity_band
    out["segment"] = labels
    return out


def audit_no_leakage(result: dict[str, Any], orders: pd.DataFrame) -> dict[str, Any]:
    """Checks that test outcomes never enter fit or £ scaling."""
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    fit_end = pd.Timestamp(result["fit_end"])
    val_end = pd.Timestamp(result["val_end"])
    obs_end = pd.Timestamp(result["obs_end"])
    summary = result["summary"]

    assert fit_end < val_end < obs_end, "fit_end < val_end < obs_end required"
    assert result["val_pred"] > 0 and result["val_actual"] > 0
    scale_recomputed = result["val_actual"] / result["val_pred"]
    assert abs(scale_recomputed - result["scale"]) < 1e-9, "scale must equal val_actual/val_pred"
    assert abs(result["test_actual"] - result["val_actual"]) > 1.0, "val/test actuals look identical"

    cal = orders[orders["order_date"] <= val_end]
    hold = orders[(orders["order_date"] > val_end) & (orders["order_date"] <= obs_end)]
    assert len(hold) > 0, "test window has no orders"
    max_weeks = weeks_between(orders["order_date"].min(), val_end) + 1
    assert float(summary["T_cal"].max()) <= max_weeks + 1e-6, (
        f"T_cal max {summary['T_cal'].max()} exceeds cal span weeks {max_weeks}"
    )

    first = orders.groupby("customer_id")["order_date"].min()
    first_ok = summary.index.map(first)
    n_future = int((pd.Series(first_ok, index=summary.index) > val_end).sum())
    assert n_future == 0, f"{n_future} customers first appear after val_end (leakage risk)"

    raw = float(summary["expected_clv_raw"].sum()) if "expected_clv_raw" in summary.columns else float("nan")
    if raw == raw:
        assert abs(raw * result["scale"] - result["test_pred"]) / max(result["test_pred"], 1) < 1e-6

    assert result["monetary_winsor_cap"] > 0

    return {
        "fit_end": str(fit_end.date()),
        "val_end": str(val_end.date()),
        "obs_end": str(obs_end.date()),
        "n_cal_orders": int(len(cal)),
        "n_test_orders": int(len(hold)),
        "n_scored_customers": int(len(summary)),
        "customers_first_seen_in_test": n_future,
        "scale_from_val_only": True,
        "test_labels_in_fit": False,
        "test_labels_in_scale": False,
        "purchase_model": "BG/NBD",
        "seasonality_overlays": False,
        "agg_error_test": float(result["agg_error"]),
        "orders_agg_error_test": float(result.get("orders_agg_error", np.nan)),
        "passed": True,
    }
