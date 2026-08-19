"""End-to-end demo.

Generates synthetic data, trains the invoice-level model, produces a 13-week
forecast with confidence bands, backtests it against the naive baseline, and
saves two charts to ./outputs/.

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cashflow_forecast import (
    SyntheticFinancials,
    PaymentBehaviourModel,
    CashFlowForecaster,
    run_backtest,
)
from cashflow_forecast.backtest import _open_invoices_asof

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def main() -> None:
    print("Generating synthetic financials...")
    gen = SyntheticFinancials()
    data = gen.generate_all()
    inv, pay, led = data["invoices"], data["payments"], data["ledger"]
    recurring = gen.config.recurring_costs
    led["date"] = pd.to_datetime(led["date"])

    # ---- train + single forecast -------------------------------------- #
    asof = pd.Timestamp("2024-06-01")
    print(f"Training payment model as of {asof.date()}...")
    # report held-out metrics on the full invoice set (80/20 chronological);
    # includes defaulted invoices so the paid/unpaid classifier can be scored
    eval_report = PaymentBehaviourModel().fit(inv)
    print("  ", eval_report)
    # then train the deployment model on everything known before asof
    train = inv[(inv["paid_date"].notna()) & (inv["paid_date"] < asof)]
    pm = PaymentBehaviourModel()
    pm.fit(train, cutoff=asof)

    opening = float(led[led["date"] <= asof].iloc[-1]["balance"])
    open_inv = _open_invoices_asof(inv, asof)
    inv_hist = inv[inv["issue_date"] < asof].copy()
    fut = inv_hist["paid_date"] >= asof
    inv_hist.loc[fut, "paid_date"] = pd.NaT
    inv_hist.loc[fut, "days_late"] = np.nan

    fc = CashFlowForecaster(pm, weeks=13).forecast(
        open_invoices=open_inv,
        payments_history=pay[pay["date"] < asof],
        recurring_costs=recurring,
        opening_balance=opening,
        start=asof,
        invoice_history=inv_hist,
    )

    # realised balances for overlay
    act = led.set_index("date")["balance"]
    fc.weekly["actual"] = [
        float(act[act.index <= (ws + pd.Timedelta(days=6))].iloc[-1])
        if (ws + pd.Timedelta(days=6)) <= act.index.max()
        else np.nan
        for ws in fc.weekly["week_start"]
    ]

    low = fc.min_balance_week()
    print(
        f"  Projected trough: ${low['balance']:,.0f} "
        f"in week of {pd.Timestamp(low['week_start']).date()}"
    )

    # ---- chart 1: forecast with bands + actuals ----------------------- #
    fig, ax = plt.subplots(figsize=(11, 6))
    w = fc.weekly
    ax.plot(w["week_start"], w["balance"], "-o", label="Forecast balance", color="#2563eb")
    ax.fill_between(
        w["week_start"], w["lower"], w["upper"], alpha=0.15, color="#2563eb",
        label="95% confidence",
    )
    ax.plot(
        w["week_start"], w["actual"], "--s", label="Actual (realised)",
        color="#16a34a",
    )
    ax.axhline(0, color="#dc2626", lw=1, ls=":", label="Zero cash")
    ax.set_title("13-Week Cash Forecast vs. Actual", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cash balance ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p1 = os.path.join(OUT, "forecast.png")
    fig.savefig(p1, dpi=130)
    print(f"  saved {p1}")

    # ---- backtest: model vs baseline ---------------------------------- #
    print("Backtesting model vs naive baseline...")
    run_dates = pd.date_range("2023-06-01", "2024-08-01", freq="MS")
    result = run_backtest(inv, pay, led, recurring, list(run_dates), weeks=13)
    imp = result.improvement()
    print(
        f"  Model MAE ${imp['model_mae']:,.0f} vs "
        f"baseline ${imp['baseline_mae']:,.0f} "
        f"({imp['mae_improvement']:.0%} better)"
    )

    # ---- chart 2: error by horizon ------------------------------------ #
    m_h = result.model_ledger.accuracy_by_horizon()
    b_h = result.baseline_ledger.accuracy_by_horizon()
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    ax2.plot(m_h["horizon_weeks"], m_h["mae"], "-o", label="Invoice-level model", color="#2563eb")
    ax2.plot(b_h["horizon_weeks"], b_h["mae"], "-s", label="Naive avg-terms baseline", color="#9ca3af")
    ax2.set_title("Forecast Error by Horizon (lower is better)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Weeks ahead")
    ax2.set_ylabel("Mean absolute error ($)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(OUT, "error_by_horizon.png")
    fig2.savefig(p2, dpi=130)
    print(f"  saved {p2}")


if __name__ == "__main__":
    main()
