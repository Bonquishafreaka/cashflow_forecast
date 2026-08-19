"""Data sources.

A `DataSource` returns invoices and payments in the standard schema the
forecasting engine expects. CSV upload is the implementation used by the
dashboard; adding a QuickBooks / Xero / Plaid source later is a matter of
implementing this same interface, so nothing downstream changes.

Standard schemas
----------------
invoices: invoice_id, customer_id, customer_size, issue_date, due_date,
          net_terms, amount, paid_date, days_late
payments: payment_id, category, kind, date, amount

`customer_size` and `net_terms` are optional in uploaded data and are filled
with sensible defaults when absent, so a user's export doesn't have to match
the synthetic schema exactly.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod

import pandas as pd


REQUIRED_INVOICE_COLS = {"invoice_id", "customer_id", "issue_date", "amount"}
REQUIRED_PAYMENT_COLS = {"date", "amount"}


class DataSourceError(ValueError):
    """Raised when incoming data can't be coerced into the standard schema."""


class DataSource(ABC):
    """Returns invoices and payments in the engine's standard schema."""

    @abstractmethod
    def get_invoices(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_payments(self) -> pd.DataFrame: ...


# --------------------------------------------------------------------------- #
# normalisation helpers (shared by every source)
# --------------------------------------------------------------------------- #
def normalise_invoices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_INVOICE_COLS - set(df.columns)
    if missing:
        raise DataSourceError(
            f"Invoice data is missing required column(s): {', '.join(sorted(missing))}. "
            f"Required: {', '.join(sorted(REQUIRED_INVOICE_COLS))}."
        )

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["issue_date", "amount"])

    # optional columns with defaults
    if "net_terms" not in df.columns:
        df["net_terms"] = 30
    df["net_terms"] = pd.to_numeric(df["net_terms"], errors="coerce").fillna(30).astype(int)

    if "due_date" in df.columns:
        df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
        df["due_date"] = df["due_date"].fillna(
            df["issue_date"] + pd.to_timedelta(df["net_terms"], unit="D")
        )
    else:
        df["due_date"] = df["issue_date"] + pd.to_timedelta(df["net_terms"], unit="D")

    if "customer_size" not in df.columns:
        df["customer_size"] = "mid"
    df["customer_size"] = (
        df["customer_size"].astype(str).str.lower().where(
            df["customer_size"].isin(["small", "mid", "large"]), "mid"
        )
    )

    if "paid_date" in df.columns:
        df["paid_date"] = pd.to_datetime(df["paid_date"], errors="coerce")
    else:
        df["paid_date"] = pd.NaT

    df["days_late"] = (df["paid_date"] - df["due_date"]).dt.days
    return df.sort_values("issue_date").reset_index(drop=True)


def normalise_payments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_PAYMENT_COLS - set(df.columns)
    if missing:
        raise DataSourceError(
            f"Payment data is missing required column(s): {', '.join(sorted(missing))}. "
            f"Required: {', '.join(sorted(REQUIRED_PAYMENT_COLS))}."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    if "category" not in df.columns:
        df["category"] = "uncategorised"
    if "kind" not in df.columns:
        # inferring recurring vs variable is out of scope; default to variable
        df["kind"] = "variable"
    if "payment_id" not in df.columns:
        df["payment_id"] = [f"PAY-{i:05d}" for i in range(len(df))]

    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CSV source (used by the dashboard)
# --------------------------------------------------------------------------- #
class CSVDataSource(DataSource):
    """Build a data source from two uploaded CSVs (invoices, payments).

    Accepts raw bytes (as received from a file upload) or file paths.
    """

    def __init__(
        self,
        invoices_csv: bytes | str,
        payments_csv: bytes | str | None = None,
    ):
        self._invoices_raw = invoices_csv
        self._payments_raw = payments_csv

    @staticmethod
    def _read(src: bytes | str) -> pd.DataFrame:
        if isinstance(src, bytes):
            return pd.read_csv(io.BytesIO(src))
        return pd.read_csv(src)

    def get_invoices(self) -> pd.DataFrame:
        return normalise_invoices(self._read(self._invoices_raw))

    def get_payments(self) -> pd.DataFrame:
        if self._payments_raw is None:
            # payments are optional; return an empty, correctly-typed frame
            return pd.DataFrame(
                columns=["payment_id", "category", "kind", "date", "amount"]
            )
        return normalise_payments(self._read(self._payments_raw))


# --------------------------------------------------------------------------- #
# in-memory source (used for the built-in sample / demo)
# --------------------------------------------------------------------------- #
class DataFrameSource(DataSource):
    """Wrap already-loaded DataFrames (e.g. the synthetic generator's output)."""

    def __init__(self, invoices: pd.DataFrame, payments: pd.DataFrame):
        self._invoices = invoices
        self._payments = payments

    def get_invoices(self) -> pd.DataFrame:
        return self._invoices

    def get_payments(self) -> pd.DataFrame:
        return self._payments


# --------------------------------------------------------------------------- #
# The seam for future integrations:
#
# class QuickBooksSource(DataSource):
#     def __init__(self, oauth_token): ...
#     def get_invoices(self):  # fetch, map QBO schema -> normalise_invoices
#     def get_payments(self):  # fetch, map QBO schema -> normalise_payments
#
# Nothing in forecast.py / payment_model.py changes: they consume the standard
# schema this module guarantees.
# --------------------------------------------------------------------------- #
