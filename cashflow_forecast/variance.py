"""Actual-vs-forecast variance tracking.

This is the self-improvement loop: every forecast, once reality catches up,
produces a labelled error signal. Persisting these (forecast, actual, error)
triples is the data exhaust that (a) lets you report forecast accuracy honestly
over time and (b) can be fed back to recalibrate the models' confidence bands.

`VarianceLedger` is a thin, storage-agnostic append log. In production you'd
back it with a database table; here it round-trips through a DataFrame / CSV so
the demo can show the accuracy trend building up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


VARIANCE_COLUMNS = [
    "forecast_run_date",
    "week_start",
    "horizon_weeks",
    "forecast_balance",
    "actual_balance",
    "abs_error",
    "pct_error",
]


@dataclass
class AccuracySummary:
    mae: float
    mape: float
    n: int
    bias: float  # mean signed error (actual - forecast); +ve = we under-forecast

    def __str__(self) -> str:
        return (
            f"Accuracy  |  MAE: {self.mae:,.0f}  |  MAPE: {self.mape:.1%}  |  "
            f"bias: {self.bias:,.0f}  |  n={self.n}"
        )


class VarianceLedger:
    def __init__(self, records: pd.DataFrame | None = None):
        self.records = (
            records
            if records is not None
            else pd.DataFrame(columns=VARIANCE_COLUMNS)
        )

    # ------------------------------------------------------------------ #
    def record_forecast(
        self,
        forecast_weekly: pd.DataFrame,
        actual_ledger: pd.DataFrame,
        run_date: pd.Timestamp,
    ) -> int:
        """Compare a forecast's weekly balances to realised balances.

        actual_ledger must have columns ['date', 'balance'] at daily grain.
        Only weeks whose end has actually occurred are recorded.
        Returns the number of new rows appended.
        """
        actual = actual_ledger.set_index("date")["balance"].sort_index()
        new_rows = []
        for i, row in forecast_weekly.reset_index(drop=True).iterrows():
            wk = pd.Timestamp(row["week_start"])
            wk_end = wk + pd.Timedelta(days=6)
            # find the realised balance at week-end (if it exists yet)
            realised = actual[actual.index <= wk_end]
            if realised.empty or wk_end > actual.index.max():
                continue
            actual_bal = float(realised.iloc[-1])
            fc_bal = float(row["balance"])
            abs_err = abs(fc_bal - actual_bal)
            pct_err = abs_err / abs(actual_bal) if actual_bal != 0 else np.nan
            new_rows.append(
                {
                    "forecast_run_date": run_date,
                    "week_start": wk,
                    "horizon_weeks": i + 1,
                    "forecast_balance": fc_bal,
                    "actual_balance": actual_bal,
                    "abs_error": abs_err,
                    "pct_error": pct_err,
                }
            )
        if new_rows:
            self.records = pd.concat(
                [self.records, pd.DataFrame(new_rows)], ignore_index=True
            )
        return len(new_rows)

    # ------------------------------------------------------------------ #
    def summary(self) -> AccuracySummary:
        if self.records.empty:
            return AccuracySummary(mae=np.nan, mape=np.nan, n=0, bias=np.nan)
        r = self.records
        bias = float((r["actual_balance"] - r["forecast_balance"]).mean())
        return AccuracySummary(
            mae=float(r["abs_error"].mean()),
            mape=float(r["pct_error"].mean()),
            n=len(r),
            bias=bias,
        )

    def accuracy_by_horizon(self) -> pd.DataFrame:
        """MAE / MAPE broken out by forecast horizon -- error should grow with
        horizon, which validates the widening confidence bands."""
        if self.records.empty:
            return pd.DataFrame(columns=["horizon_weeks", "mae", "mape", "n"])
        return (
            self.records.groupby("horizon_weeks")
            .agg(
                mae=("abs_error", "mean"),
                mape=("pct_error", "mean"),
                n=("abs_error", "size"),
            )
            .reset_index()
        )

    def recalibrated_error_std(self) -> float:
        """Estimate a per-week error std from realised errors, for feeding back
        into the forecaster's confidence bands."""
        if self.records.empty:
            return np.nan
        # de-trend by horizon: error ~ std * sqrt(h)  ->  std ~ error / sqrt(h)
        per_week = self.records["abs_error"] / np.sqrt(
            self.records["horizon_weeks"]
        )
        return float(per_week.mean() / 0.8)  # 0.8 ~ E|N(0,1)| adjustment

    # ------------------------------------------------------------------ #
    def to_csv(self, path: str) -> None:
        self.records.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str) -> "VarianceLedger":
        return cls(pd.read_csv(path, parse_dates=["forecast_run_date", "week_start"]))
