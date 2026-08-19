"""cashflow_forecast: invoice-level cash-flow forecasting for SMBs.

Public API:
    SyntheticFinancials   -- generate realistic messy financial data
    PaymentBehaviourModel -- invoice-level payment-timing prediction
    CashFlowForecaster    -- 13-week rolling cash forecast
    flat_terms_forecast   -- naive baseline for comparison
    VarianceLedger        -- actual-vs-forecast tracking / data exhaust
    run_backtest          -- walk-forward evaluation
"""

from .synthetic import SyntheticFinancials, GeneratorConfig
from .payment_model import PaymentBehaviourModel, PaymentModelReport
from .forecast import CashFlowForecaster, Forecast, flat_terms_forecast
from .variance import VarianceLedger, AccuracySummary
from .backtest import run_backtest, BacktestResult

__version__ = "0.1.0"

__all__ = [
    "SyntheticFinancials",
    "GeneratorConfig",
    "PaymentBehaviourModel",
    "PaymentModelReport",
    "CashFlowForecaster",
    "Forecast",
    "flat_terms_forecast",
    "VarianceLedger",
    "AccuracySummary",
    "run_backtest",
    "BacktestResult",
]
