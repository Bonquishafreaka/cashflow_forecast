"""Walk-forward backtest.

Proves the invoice-level forecaster beats the naive 'flat terms' baseline by
replaying history: at each run-date we train only on the past, forecast the
next 13 weeks, then score both models against what actually happened.

This is what produces the headline "our model reduces 13-week forecast error by
X% vs. average-terms" number for a demo or write-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .forecast import CashFlowForecaster, flat_terms_forecast
from .payment_model import PaymentBehaviourModel
from .variance import VarianceLedger


@dataclass
class BacktestResult:
    model_ledger: VarianceLedger
    baseline_ledger: VarianceLedger

    def improvement(self) -> dict:
        m = self.model_ledger.summary()
        b = self.baseline_ledger.summary()
        mae_impr = (b.mae - m.mae) / b.mae if b.mae else np.nan
        mape_impr = (b.mape - m.mape) / b.mape if b.mape else np.nan
        return {
            "model_mae": m.mae,
            "baseline_mae": b.mae,
            "mae_improvement": mae_impr,
            "model_mape": m.mape,
            "baseline_mape": b.mape,
            "mape_improvement": mape_impr,
            "n_comparisons": m.n,
        }


def _open_invoices_asof(invoices: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Invoices issued on/before `asof` that are still open at `asof`
    (i.e. not yet paid as of that date). Payment info after asof is hidden."""
    issued = invoices[invoices["issue_date"] <= asof].copy()
    open_mask = issued["paid_date"].isna() | (issued["paid_date"] > asof)
    open_inv = issued[open_mask].copy()
    # hide the future: null out paid info the forecaster shouldn't see
    open_inv["paid_date"] = pd.NaT
    open_inv["days_late"] = np.nan
    return open_inv


def run_backtest(
    invoices: pd.DataFrame,
    payments: pd.DataFrame,
    ledger: pd.DataFrame,
    recurring_costs: dict,
    run_dates: list[pd.Timestamp],
    weeks: int = 13,
) -> BacktestResult:
    model_ledger = VarianceLedger()
    baseline_ledger = VarianceLedger()

    ledger = ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])

    for asof in run_dates:
        asof = pd.Timestamp(asof)

        # --- train payment model on invoices resolved before asof --- #
        # Include both paid (paid before asof) and likely-defaulted invoices
        # (issued long enough ago that non-payment is now observable), so the
        # classifier sees both classes.
        resolved_paid = (invoices["paid_date"].notna()) & (
            invoices["paid_date"] < asof
        )
        long_unpaid = (invoices["paid_date"].isna()) & (
            invoices["issue_date"] < asof - pd.Timedelta(days=120)
        )
        train_inv = invoices[
            (invoices["issue_date"] < asof) & (resolved_paid | long_unpaid)
        ].copy()
        if len(train_inv) < 100:
            continue  # not enough history yet

        pm = PaymentBehaviourModel()
        pm.fit(train_inv, cutoff=asof)

        # opening balance = realised balance as of run-date
        bal_asof = ledger[ledger["date"] <= asof]
        if bal_asof.empty:
            continue
        opening = float(bal_asof.iloc[-1]["balance"])

        open_inv = _open_invoices_asof(invoices, asof)
        hist_payments = payments[payments["date"] < asof]

        # invoice history for run-rate: issued before asof, with payment info
        # censored at asof (payments after asof are hidden)
        inv_hist = invoices[invoices["issue_date"] < asof].copy()
        future_pay = inv_hist["paid_date"] >= asof
        inv_hist.loc[future_pay, "paid_date"] = pd.NaT
        inv_hist.loc[future_pay, "days_late"] = np.nan

        # --- model forecast --- #
        fc = CashFlowForecaster(payment_model=pm, weeks=weeks)
        model_forecast = fc.forecast(
            open_invoices=open_inv,
            payments_history=hist_payments,
            recurring_costs=recurring_costs,
            opening_balance=opening,
            start=asof,
            invoice_history=inv_hist,
        )
        model_weekly = model_forecast.weekly.copy()

        # --- baseline forecast --- #
        base_forecast = flat_terms_forecast(
            open_invoices=open_inv,
            payments_history=hist_payments,
            recurring_costs=recurring_costs,
            opening_balance=opening,
            start=asof,
            weeks=weeks,
        )

        # --- score both against realised ledger --- #
        model_ledger.record_forecast(model_weekly, ledger, run_date=asof)
        baseline_ledger.record_forecast(
            base_forecast.weekly, ledger, run_date=asof
        )

    return BacktestResult(
        model_ledger=model_ledger, baseline_ledger=baseline_ledger
    )
