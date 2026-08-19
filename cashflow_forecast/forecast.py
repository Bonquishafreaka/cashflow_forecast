"""Cash-flow forecast engine.

Assembles a forward daily/weekly cash forecast from three components:

  1. AR inflows  -- from the invoice-level PaymentBehaviourModel: each open
     invoice contributes its probability-weighted amount on its predicted
     paid-date.
  2. Recurring AP -- known fixed outflows (payroll, rent, ...) projected
     forward on their known schedule.
  3. Variable AP  -- a seasonal baseline model for supplier bills / lumpy
     costs that don't have a fixed schedule.

The headline output is a 13-week rolling forecast with a projected running
balance and confidence bands derived from historical forecast error.

A deliberately simple "flat average terms" forecaster is included as a
baseline so a demo can show the invoice-level model beating it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .payment_model import PaymentBehaviourModel


# --------------------------------------------------------------------------- #
# AP projection
# --------------------------------------------------------------------------- #
def project_recurring_ap(
    recurring_costs: dict,
    start: pd.Timestamp,
    weeks: int,
) -> pd.DataFrame:
    """Project fixed recurring outflows forward on their day-of-month schedule."""
    rows = []
    end = start + pd.Timedelta(weeks=weeks)
    # iterate month by month across the horizon
    month = pd.Timestamp(year=start.year, month=start.month, day=1)
    while month <= end:
        for name, (amount, dom) in recurring_costs.items():
            day = min(dom, 28)
            d = pd.Timestamp(year=month.year, month=month.month, day=day)
            if start <= d <= end:
                rows.append({"date": d, "amount": float(amount), "category": name})
        month = month + pd.offsets.MonthBegin(1)
    return pd.DataFrame(rows)


def project_variable_ap(
    payments: pd.DataFrame,
    start: pd.Timestamp,
    weeks: int,
    lumpy_threshold: float = 12_000.0,
) -> pd.DataFrame:
    """Project variable outflows.

    Splits history into a *regular* variable stream (supplier bills) and
    *lumpy* one-offs. The regular stream is projected at its recent median
    weekly rate (robust to outliers); lumpy events are spread as a small
    expected-value-per-week amount rather than projected as spikes. This
    avoids the systematic over-forecasting that a naive mean-of-weeks baseline
    produces when rare large payments inflate the average.
    """
    hist = payments[payments["kind"] != "recurring"].copy()
    if hist.empty:
        return pd.DataFrame(columns=["date", "amount", "category"])

    regular = hist[hist["amount"] < lumpy_threshold]
    lumpy = hist[hist["amount"] >= lumpy_threshold]

    # regular weekly rate: median of realised weekly totals (robust)
    reg_weekly = regular.set_index("date")["amount"].resample("W").sum()
    reg_rate = float(reg_weekly.median()) if not reg_weekly.empty else 0.0

    # lumpy expected value per week = (total lumpy / weeks observed)
    if not lumpy.empty:
        span_weeks = max(
            1,
            (hist["date"].max() - hist["date"].min()).days / 7.0,
        )
        lumpy_per_week = float(lumpy["amount"].sum() / span_weeks)
    else:
        lumpy_per_week = 0.0

    rows = []
    for w in range(weeks):
        wk_start = start + pd.Timedelta(weeks=w)
        d = wk_start + pd.Timedelta(days=3)
        rows.append(
            {
                "date": d,
                "amount": round(reg_rate + lumpy_per_week, 2),
                "category": "variable_ap",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# forecast result container
# --------------------------------------------------------------------------- #
@dataclass
class Forecast:
    weekly: pd.DataFrame  # week_start, inflow, outflow, net, balance, lower, upper
    opening_balance: float
    generated_at: pd.Timestamp

    def min_balance_week(self) -> pd.Series:
        return self.weekly.loc[self.weekly["balance"].idxmin()]

    def shortfall_weeks(self, threshold: float = 0.0) -> pd.DataFrame:
        return self.weekly[self.weekly["balance"] < threshold]


# --------------------------------------------------------------------------- #
# main engine
# --------------------------------------------------------------------------- #
class CashFlowForecaster:
    def __init__(
        self,
        payment_model: PaymentBehaviourModel | None = None,
        weeks: int = 13,
    ):
        self.payment_model = payment_model or PaymentBehaviourModel()
        self.weeks = weeks

    # -- AR inflow projection ------------------------------------------- #
    def _project_ar(
        self, open_invoices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Inflows from the current open AR book, via invoice-level prediction."""
        if open_invoices.empty:
            return pd.DataFrame(columns=["date", "amount"])
        preds = self.payment_model.predict_invoice(open_invoices)
        preds = preds[
            (preds["pred_paid_date"] >= start) & (preds["pred_paid_date"] <= end)
        ]
        return preds.rename(columns={"pred_paid_date": "date"})[
            ["date", "expected_amount"]
        ].rename(columns={"expected_amount": "amount"})

    def _project_future_invoicing(
        self,
        recent_invoices: pd.DataFrame,
        start: pd.Timestamp,
        lookback_weeks: int = 13,
    ) -> pd.DataFrame:
        """Project cash from invoices *not yet issued* at forecast time.

        Uses the recent weekly billing run-rate (last `lookback_weeks`) and
        assumes new invoices are collected after the historical average
        collection lag. Without this, the forecast starves in later weeks
        because the current open book empties out.
        """
        if recent_invoices.empty:
            return pd.DataFrame(columns=["date", "amount"])

        window_start = start - pd.Timedelta(weeks=lookback_weeks)
        recent = recent_invoices[
            (recent_invoices["issue_date"] >= window_start)
            & (recent_invoices["issue_date"] < start)
        ]
        if recent.empty:
            return pd.DataFrame(columns=["date", "amount"])

        weekly_billing = recent["amount"].sum() / lookback_weeks
        # average collection lag from issue to payment (fall back to net terms)
        paid = recent[recent["paid_date"].notna()]
        if not paid.empty:
            avg_lag = int(
                (paid["paid_date"] - paid["issue_date"]).dt.days.mean()
            )
        else:
            avg_lag = int(recent["net_terms"].mean())
        avg_lag = max(avg_lag, 7)

        # collection rate (share of invoiced amount actually collected)
        collect_rate = (
            paid["amount"].sum() / recent["amount"].sum()
            if recent["amount"].sum() > 0
            else 0.95
        )

        rows = []
        for w in range(self.weeks):
            issue_week = start + pd.Timedelta(weeks=w)
            pay_date = issue_week + pd.Timedelta(days=avg_lag)
            rows.append(
                {
                    "date": pay_date,
                    "amount": round(weekly_billing * collect_rate, 2),
                }
            )
        return pd.DataFrame(rows)

    # -- assemble weekly forecast --------------------------------------- #
    def forecast(
        self,
        open_invoices: pd.DataFrame,
        payments_history: pd.DataFrame,
        recurring_costs: dict,
        opening_balance: float,
        start: pd.Timestamp,
        invoice_history: pd.DataFrame | None = None,
        error_std: float | None = None,
    ) -> Forecast:
        end = start + pd.Timedelta(weeks=self.weeks)

        ar_open = self._project_ar(open_invoices, start, end)
        # cash from invoices not yet issued (recent billing run-rate)
        hist_for_runrate = (
            invoice_history if invoice_history is not None else open_invoices
        )
        ar_future = self._project_future_invoicing(hist_for_runrate, start)
        ar = pd.concat([ar_open, ar_future], ignore_index=True)

        rec_ap = project_recurring_ap(recurring_costs, start, self.weeks)
        var_ap = project_variable_ap(payments_history, start, self.weeks)

        # bucket everything by week
        def weekly_sum(df: pd.DataFrame, col: str) -> pd.Series:
            if df.empty:
                return pd.Series(dtype=float)
            s = df.copy()
            s["week_start"] = (
                s["date"] - pd.to_timedelta(s["date"].dt.dayofweek, unit="D")
            ).dt.normalize()
            return s.groupby("week_start")[col].sum()

        week_index = pd.date_range(
            start - pd.Timedelta(days=start.dayofweek),
            periods=self.weeks,
            freq="W-MON",
        ).normalize()

        inflow = weekly_sum(ar, "amount").reindex(week_index).fillna(0.0)
        out_rec = weekly_sum(rec_ap, "amount").reindex(week_index).fillna(0.0)
        out_var = weekly_sum(var_ap, "amount").reindex(week_index).fillna(0.0)
        outflow = out_rec + out_var

        weekly = pd.DataFrame(
            {
                "week_start": week_index,
                "inflow": inflow.values,
                "outflow": outflow.values,
            }
        )
        weekly["net"] = weekly["inflow"] - weekly["outflow"]
        weekly["balance"] = opening_balance + weekly["net"].cumsum()

        # confidence bands: error grows with sqrt(horizon) (random-walk-like)
        if error_std is None:
            error_std = 0.15 * abs(weekly["net"]).mean()
        horizon = np.arange(1, len(weekly) + 1)
        band = error_std * np.sqrt(horizon)
        weekly["lower"] = weekly["balance"] - 1.96 * band
        weekly["upper"] = weekly["balance"] + 1.96 * band

        return Forecast(
            weekly=weekly,
            opening_balance=opening_balance,
            generated_at=start,
        )


# --------------------------------------------------------------------------- #
# baseline for comparison in demos
# --------------------------------------------------------------------------- #
def flat_terms_forecast(
    open_invoices: pd.DataFrame,
    payments_history: pd.DataFrame,
    recurring_costs: dict,
    opening_balance: float,
    start: pd.Timestamp,
    weeks: int = 13,
) -> Forecast:
    """Naive baseline: assume every invoice is paid exactly on its due date and
    ignore default risk. This is the 'average terms' approach the invoice-level
    model is meant to beat.
    """
    end = start + pd.Timedelta(weeks=weeks)
    ar = open_invoices.copy()
    ar = ar[(ar["due_date"] >= start) & (ar["due_date"] <= end)]
    ar = ar.rename(columns={"due_date": "date"})[["date", "amount"]]

    rec_ap = project_recurring_ap(recurring_costs, start, weeks)
    var_ap = project_variable_ap(payments_history, start, weeks)

    def weekly_sum(df: pd.DataFrame, col: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        s = df.copy()
        s["week_start"] = (
            s["date"] - pd.to_timedelta(s["date"].dt.dayofweek, unit="D")
        ).dt.normalize()
        return s.groupby("week_start")[col].sum()

    week_index = pd.date_range(
        start - pd.Timedelta(days=start.dayofweek), periods=weeks, freq="W-MON"
    ).normalize()

    inflow = weekly_sum(ar, "amount").reindex(week_index).fillna(0.0)
    outflow = (
        weekly_sum(rec_ap, "amount").reindex(week_index).fillna(0.0)
        + weekly_sum(var_ap, "amount").reindex(week_index).fillna(0.0)
    )
    weekly = pd.DataFrame(
        {"week_start": week_index, "inflow": inflow.values, "outflow": outflow.values}
    )
    weekly["net"] = weekly["inflow"] - weekly["outflow"]
    weekly["balance"] = opening_balance + weekly["net"].cumsum()
    weekly["lower"] = weekly["balance"]
    weekly["upper"] = weekly["balance"]
    return Forecast(weekly=weekly, opening_balance=opening_balance, generated_at=start)
