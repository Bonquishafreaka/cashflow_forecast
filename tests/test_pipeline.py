"""Tests for the cash-flow forecasting pipeline.

Run with:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from cashflow_forecast import (
    SyntheticFinancials,
    GeneratorConfig,
    PaymentBehaviourModel,
    CashFlowForecaster,
    flat_terms_forecast,
    VarianceLedger,
    run_backtest,
)


@pytest.fixture(scope="module")
def data():
    gen = SyntheticFinancials(GeneratorConfig(seed=1))
    return gen.generate_all(), gen.config.recurring_costs


def test_synthetic_shapes(data):
    d, _ = data
    assert not d["invoices"].empty
    assert not d["payments"].empty
    assert not d["ledger"].empty
    assert "balance" in d["ledger"].columns
    assert d["ledger"]["balance"].notna().all()


def test_synthetic_has_late_and_defaults(data):
    d, _ = data
    inv = d["invoices"]
    assert (inv["days_late"] > 0).any()
    assert inv["paid_date"].isna().any()


def test_payment_model_trains(data):
    d, _ = data
    m = PaymentBehaviourModel()
    report = m.fit(d["invoices"])
    assert report.late_mae < 20
    assert report.paid_auc > 0.6


def test_payment_model_no_leakage_features(data):
    d, _ = data
    m = PaymentBehaviourModel()
    feat = m.build_features(d["invoices"])
    first_rows = feat.sort_values("issue_date").groupby("customer_id").head(1)
    assert (first_rows["cust_hist_count"] == 0).any()


def test_predict_invoice_on_open_book(data):
    d, _ = data
    m = PaymentBehaviourModel()
    m.fit(d["invoices"])
    open_inv = d["invoices"].head(20).copy()
    open_inv["paid_date"] = pd.NaT
    open_inv["days_late"] = np.nan
    preds = m.predict_invoice(open_inv)
    assert (preds["pred_paid_prob"] >= 0).all()
    assert (preds["pred_paid_prob"] <= 1).all()
    assert preds["pred_paid_date"].notna().all()


def test_forecast_structure(data):
    d, recurring = data
    m = PaymentBehaviourModel()
    m.fit(d["invoices"])
    start = pd.Timestamp("2024-01-01")
    open_inv = d["invoices"][d["invoices"]["issue_date"] < start].copy()
    open_inv["paid_date"] = pd.NaT
    fc = CashFlowForecaster(m, weeks=13).forecast(
        open_invoices=open_inv,
        payments_history=d["payments"],
        recurring_costs=recurring,
        opening_balance=100_000.0,
        start=start,
        invoice_history=d["invoices"][d["invoices"]["issue_date"] < start],
    )
    assert len(fc.weekly) == 13
    widths = fc.weekly["upper"] - fc.weekly["lower"]
    assert widths.iloc[-1] > widths.iloc[0]


def test_model_beats_baseline(data):
    d, recurring = data
    run_dates = pd.date_range("2023-06-01", "2024-06-01", freq="MS")
    result = run_backtest(
        d["invoices"], d["payments"], d["ledger"], recurring, list(run_dates)
    )
    imp = result.improvement()
    assert imp["mae_improvement"] > 0.15


def test_variance_ledger_records_and_summarises(data):
    d, recurring = data
    m = PaymentBehaviourModel()
    m.fit(d["invoices"])
    start = pd.Timestamp("2023-06-01")
    open_inv = d["invoices"][d["invoices"]["issue_date"] < start].copy()
    open_inv["paid_date"] = pd.NaT
    fc = CashFlowForecaster(m, weeks=13).forecast(
        open_invoices=open_inv,
        payments_history=d["payments"],
        recurring_costs=recurring,
        opening_balance=100_000.0,
        start=start,
        invoice_history=d["invoices"][d["invoices"]["issue_date"] < start],
    )
    ledger = VarianceLedger()
    n = ledger.record_forecast(fc.weekly, d["ledger"], run_date=start)
    assert n > 0
    summary = ledger.summary()
    assert summary.n == n
    assert summary.mae >= 0
