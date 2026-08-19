"""Forecast service.

Glue between a `DataSource` and the forecasting engine: takes invoices +
payments, trains the payment model on resolved invoices, projects the open
book, and returns a forecast plus headline figures in plain dicts ready to be
JSON-serialised for the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import CashFlowForecaster
from .payment_model import PaymentBehaviourModel
from .sources import DataSource

# default recurring costs used when an upload has no recurring-flagged payments
DEFAULT_RECURRING: dict = {}


def _infer_recurring(payments: pd.DataFrame) -> dict:
    """Pull recurring costs out of payment history if the 'kind' column marks
    them; otherwise return empty (variable AP baseline still covers outflows).
    """
    if payments.empty or "kind" not in payments.columns:
        return {}
    rec = payments[payments["kind"] == "recurring"]
    if rec.empty:
        return {}
    out = {}
    for name, grp in rec.groupby("category"):
        amount = float(grp["amount"].median())
        dom = int(pd.to_datetime(grp["date"]).dt.day.median())
        out[str(name)] = (amount, dom)
    return out


def build_forecast(
    source: DataSource,
    weeks: int = 13,
    opening_balance: float | None = None,
    start: pd.Timestamp | None = None,
) -> dict:
    """Run the full pipeline on a data source and return a JSON-ready dict."""
    invoices = source.get_invoices()
    payments = source.get_payments()

    if invoices.empty:
        raise ValueError("No usable invoices found in the uploaded data.")

    # forecast start: day after the latest known activity
    if start is None:
        last_dates = [invoices["issue_date"].max()]
        if not payments.empty:
            last_dates.append(payments["date"].max())
        start = pd.Timestamp(max(last_dates)).normalize() + pd.Timedelta(days=1)

    # train the payment model on invoices resolved before start
    resolved = invoices[
        invoices["paid_date"].notna() & (invoices["paid_date"] < start)
    ]
    model = PaymentBehaviourModel()
    trained = False
    if len(resolved) >= 50:
        model.fit(resolved, cutoff=start)
        trained = True
    else:
        # not enough paid history: fit on whatever we have so predict works
        fit_on = invoices if invoices["paid_date"].notna().any() else None
        if fit_on is not None and fit_on["paid_date"].notna().sum() >= 5:
            model.fit(fit_on)
            trained = True

    # open book = invoices not yet paid as of start
    open_inv = invoices[
        invoices["paid_date"].isna() | (invoices["paid_date"] >= start)
    ].copy()
    open_inv["paid_date"] = pd.NaT
    open_inv["days_late"] = np.nan

    # opening balance: user-supplied, else derive from paid history net of payments
    if opening_balance is None:
        inflow = invoices.dropna(subset=["paid_date"])
        inflow = inflow[inflow["paid_date"] < start]["amount"].sum()
        outflow = payments[payments["date"] < start]["amount"].sum() if not payments.empty else 0.0
        opening_balance = float(inflow - outflow)

    recurring = _infer_recurring(payments)

    if not trained:
        raise ValueError(
            "Not enough paid-invoice history to forecast. Include some invoices "
            "with a paid_date so the model can learn payment behaviour."
        )

    forecaster = CashFlowForecaster(model, weeks=weeks)
    forecast = forecaster.forecast(
        open_invoices=open_inv,
        payments_history=payments,
        recurring_costs=recurring,
        opening_balance=opening_balance,
        start=start,
        invoice_history=invoices,
    )

    w = forecast.weekly
    trough = forecast.min_balance_week()
    shortfalls = forecast.shortfall_weeks()

    return {
        "start": start.strftime("%Y-%m-%d"),
        "weeks": weeks,
        "opening_balance": round(opening_balance, 2),
        "series": [
            {
                "week_start": pd.Timestamp(r["week_start"]).strftime("%Y-%m-%d"),
                "inflow": round(float(r["inflow"]), 2),
                "outflow": round(float(r["outflow"]), 2),
                "balance": round(float(r["balance"]), 2),
                "lower": round(float(r["lower"]), 2),
                "upper": round(float(r["upper"]), 2),
            }
            for _, r in w.iterrows()
        ],
        "trough": {
            "week_start": pd.Timestamp(trough["week_start"]).strftime("%Y-%m-%d"),
            "balance": round(float(trough["balance"]), 2),
        },
        "shortfall_weeks": [
            pd.Timestamp(r["week_start"]).strftime("%Y-%m-%d")
            for _, r in shortfalls.iterrows()
        ],
        "n_open_invoices": int(len(open_inv)),
        "n_invoices_total": int(len(invoices)),
    }
