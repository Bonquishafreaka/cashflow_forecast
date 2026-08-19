"""Synthetic SMB financial data generator.

Produces the messy, realistic data the forecasting models are designed to
handle: lumpy revenue, seasonal swings, customers who pay *late* relative to
their invoice due dates, recurring fixed costs, and irregular one-off events.

The goal is a dataset where a naive "average payment terms" forecast does
poorly and an invoice-level payment-behaviour model does well -- so the
difference is visible in a demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass
class CustomerProfile:
    """A single customer's invariant paying behaviour.

    pay_delay_mean / pay_delay_std describe how many days *after* the due date
    this customer actually pays (positive = habitually late). This is the
    signal the payment-behaviour model learns per customer.
    """

    customer_id: str
    size: str  # "small" | "mid" | "large"
    pay_delay_mean: float
    pay_delay_std: float
    default_prob: float  # chance an invoice is never paid within horizon
    invoices_per_month: float


@dataclass
class GeneratorConfig:
    start: date = date(2022, 1, 1)
    end: date = date(2025, 1, 1)
    n_customers: int = 40
    seed: int = 7
    base_monthly_revenue: float = 120_000.0
    seasonal_amplitude: float = 0.25  # +/- 25% seasonal swing
    opening_balance: float = 85_000.0
    # recurring fixed outflows (name -> (amount, day_of_month))
    recurring_costs: dict = field(
        default_factory=lambda: {
            "payroll_1": (28_000.0, 15),
            "payroll_2": (28_000.0, 30),
            "rent": (9_500.0, 1),
            "saas_stack": (4_200.0, 5),
            "insurance": (2_100.0, 10),
        }
    )


class SyntheticFinancials:
    """Generates invoices (AR), bill payments (AP) and a derived cash ledger."""

    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.customers = self._make_customers()

    # ------------------------------------------------------------------ #
    # customers
    # ------------------------------------------------------------------ #
    def _make_customers(self) -> list[CustomerProfile]:
        sizes = ["small", "mid", "large"]
        size_weights = [0.55, 0.35, 0.10]
        customers: list[CustomerProfile] = []
        for i in range(self.config.n_customers):
            size = self.rng.choice(sizes, p=size_weights)
            if size == "small":
                delay_mean = self.rng.normal(6, 3)
                inv_pm = self.rng.uniform(0.3, 1.0)
                default_p = 0.03
            elif size == "mid":
                delay_mean = self.rng.normal(11, 4)
                inv_pm = self.rng.uniform(0.5, 1.5)
                default_p = 0.02
            else:  # large: slow but reliable
                delay_mean = self.rng.normal(18, 5)
                inv_pm = self.rng.uniform(1.0, 2.5)
                default_p = 0.01
            customers.append(
                CustomerProfile(
                    customer_id=f"CUST-{i:03d}",
                    size=size,
                    pay_delay_mean=float(max(delay_mean, -2)),
                    pay_delay_std=float(self.rng.uniform(2, 6)),
                    default_prob=float(default_p),
                    invoices_per_month=float(inv_pm),
                )
            )
        return customers

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _seasonal_factor(self, d: date) -> float:
        # peak late in the calendar year (month 11), trough in month 5
        month_angle = 2 * np.pi * (d.month - 1) / 12.0
        return 1.0 + self.config.seasonal_amplitude * np.sin(month_angle - np.pi / 2)

    def _months(self) -> list[date]:
        months = []
        d = self.config.start
        while d < self.config.end:
            months.append(d)
            # advance one month
            if d.month == 12:
                d = date(d.year + 1, 1, 1)
            else:
                d = date(d.year, d.month + 1, 1)
        return months

    @staticmethod
    def _clamp_day(year: int, month: int, day: int) -> date:
        """Clamp a day-of-month to a valid date (handles 30 -> Feb, etc.)."""
        for dd in (day, 28):
            try:
                return date(year, month, min(dd, 28) if dd == 28 else dd)
            except ValueError:
                continue
        return date(year, month, 28)

    # ------------------------------------------------------------------ #
    # AR: invoices
    # ------------------------------------------------------------------ #
    def generate_invoices(self) -> pd.DataFrame:
        rows = []
        inv_counter = 0
        for month_start in self._months():
            seasonal = self._seasonal_factor(month_start)
            for cust in self.customers:
                # number of invoices this customer issues this month (Poisson)
                lam = cust.invoices_per_month * seasonal
                n_inv = self.rng.poisson(lam)
                for _ in range(n_inv):
                    inv_counter += 1
                    issue_day = int(self.rng.integers(1, 28))
                    issue_date = date(month_start.year, month_start.month, issue_day)
                    net_terms = int(self.rng.choice([15, 30, 45], p=[0.2, 0.6, 0.2]))
                    due_date = issue_date + timedelta(days=net_terms)

                    # invoice amount scales with customer size + revenue base
                    size_mult = {"small": 0.4, "mid": 1.0, "large": 2.6}[cust.size]
                    base = self.config.base_monthly_revenue / (
                        self.config.n_customers * 1.2
                    )
                    amount = float(
                        max(
                            250.0,
                            self.rng.lognormal(
                                mean=np.log(base * size_mult), sigma=0.5
                            ),
                        )
                    )

                    # actual payment behaviour
                    defaulted = self.rng.random() < cust.default_prob
                    if defaulted:
                        paid_date = pd.NaT
                    else:
                        delay = self.rng.normal(
                            cust.pay_delay_mean, cust.pay_delay_std
                        )
                        # occasional early payers, occasional very late
                        if self.rng.random() < 0.05:
                            delay += self.rng.uniform(20, 45)
                        paid_offset = int(round(max(delay, -net_terms + 1)))
                        paid_date = due_date + timedelta(days=paid_offset)

                    rows.append(
                        {
                            "invoice_id": f"INV-{inv_counter:05d}",
                            "customer_id": cust.customer_id,
                            "customer_size": cust.size,
                            "issue_date": pd.Timestamp(issue_date),
                            "due_date": pd.Timestamp(due_date),
                            "net_terms": net_terms,
                            "amount": round(amount, 2),
                            "paid_date": pd.Timestamp(paid_date)
                            if paid_date is not pd.NaT
                            else pd.NaT,
                        }
                    )
        df = pd.DataFrame(rows).sort_values("issue_date").reset_index(drop=True)
        # derived label: days late relative to due date (NaN if unpaid)
        df["days_late"] = (df["paid_date"] - df["due_date"]).dt.days
        return df

    # ------------------------------------------------------------------ #
    # AP: recurring + variable outflows
    # ------------------------------------------------------------------ #
    def generate_payments(self) -> pd.DataFrame:
        rows = []
        pay_counter = 0
        for month_start in self._months():
            # recurring fixed costs
            for name, (amount, dom) in self.config.recurring_costs.items():
                pay_counter += 1
                pay_date = self._clamp_day(month_start.year, month_start.month, dom)
                jitter = self.rng.normal(1.0, 0.01)  # tiny variation
                rows.append(
                    {
                        "payment_id": f"PAY-{pay_counter:05d}",
                        "category": name,
                        "kind": "recurring",
                        "date": pd.Timestamp(pay_date),
                        "amount": round(float(amount) * jitter, 2),
                    }
                )
            # variable outflows (supplier bills, scale with seasonality)
            seasonal = self._seasonal_factor(month_start)
            n_var = self.rng.poisson(8 * seasonal)
            for _ in range(n_var):
                pay_counter += 1
                day = int(self.rng.integers(1, 28))
                pay_date = date(month_start.year, month_start.month, day)
                amount = float(self.rng.lognormal(mean=np.log(3_500), sigma=0.7))
                rows.append(
                    {
                        "payment_id": f"PAY-{pay_counter:05d}",
                        "category": "supplier_bill",
                        "kind": "variable",
                        "date": pd.Timestamp(pay_date),
                        "amount": round(amount, 2),
                    }
                )
            # occasional lumpy one-off (equipment, tax) ~15% of months
            if self.rng.random() < 0.15:
                pay_counter += 1
                day = int(self.rng.integers(1, 28))
                pay_date = date(month_start.year, month_start.month, day)
                amount = float(self.rng.uniform(15_000, 45_000))
                rows.append(
                    {
                        "payment_id": f"PAY-{pay_counter:05d}",
                        "category": "one_off",
                        "kind": "lumpy",
                        "date": pd.Timestamp(pay_date),
                        "amount": round(amount, 2),
                    }
                )
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # derived daily cash ledger (actuals)
    # ------------------------------------------------------------------ #
    def build_ledger(
        self, invoices: pd.DataFrame, payments: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine paid invoices (inflow) and payments (outflow) into a daily
        cash ledger with a running balance."""
        inflows = (
            invoices.dropna(subset=["paid_date"])
            .groupby("paid_date")["amount"]
            .sum()
            .rename("inflow")
        )
        outflows = payments.groupby("date")["amount"].sum().rename("outflow")

        idx = pd.date_range(self.config.start, self.config.end, freq="D")
        ledger = pd.DataFrame(index=idx)
        ledger["inflow"] = inflows.reindex(idx).fillna(0.0)
        ledger["outflow"] = outflows.reindex(idx).fillna(0.0)
        ledger["net"] = ledger["inflow"] - ledger["outflow"]
        ledger["balance"] = self.config.opening_balance + ledger["net"].cumsum()
        ledger.index.name = "date"
        return ledger.reset_index()

    def generate_all(self) -> dict[str, pd.DataFrame]:
        invoices = self.generate_invoices()
        payments = self.generate_payments()
        ledger = self.build_ledger(invoices, payments)
        return {"invoices": invoices, "payments": payments, "ledger": ledger}


if __name__ == "__main__":
    gen = SyntheticFinancials()
    data = gen.generate_all()
    for name, df in data.items():
        print(f"\n=== {name} ({len(df)} rows) ===")
        print(df.head())
