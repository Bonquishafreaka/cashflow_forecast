# Cash-Flow Forecast

Invoice-level cash-flow forecasting for small and mid-sized businesses.

Most SMB cash-flow tools forecast inflows by applying **average payment terms**
across the whole receivables book — "everyone pays in 30 days." Real customers
don't. They pay *late*, by amounts that differ per customer, and some don't pay
at all. This project forecasts cash at the **individual-invoice level**: it
learns each customer's actual paying behaviour and predicts *when* and *whether*
each open invoice will be collected, then rolls those predictions into a 13-week
cash forecast with confidence bands.

On synthetic-but-realistic data, the invoice-level model cuts 13-week forecast
error roughly in half versus the average-terms baseline.

| Metric (13-week horizon) | Invoice-level model | Naive avg-terms baseline |
| --- | --- | --- |
| Mean absolute error | **~$39k** | ~$90k |
| MAPE | **~40%** | ~84% |

![13-week forecast vs actual](outputs/forecast.png)

![Forecast error by horizon](outputs/error_by_horizon.png)

*(Charts are produced by `scripts/demo.py`. Numbers vary slightly with the
random seed.)*

## How it works

The forecast is assembled from three streams:

1. **AR inflows — invoice-level prediction.** A gradient-boosted model predicts,
   for each open invoice, how many days after its due date it will be paid
   (regression) and the probability it is paid at all (classification). Each
   invoice contributes its probability-weighted amount on its predicted paid
   date. Features include invoice amount and terms, and — most importantly —
   each customer's *historical* lateness statistics, computed with an expanding
   window shifted by one so a customer's own future never leaks into its past.

2. **Recurring AP.** Known fixed outflows (payroll, rent, SaaS, insurance) are
   projected forward on their day-of-month schedule.

3. **Variable AP.** Supplier bills are projected at their recent *median* weekly
   rate (robust to outliers), and lumpy one-offs are spread as an expected value
   per week rather than modelled as spikes — which avoids the systematic
   over-forecasting a naive mean-of-weeks baseline produces.

Because the current open book empties out over a 13-week horizon, the forecaster
also projects **expected new invoicing** from the recent billing run-rate and
collection lag; without this the forecast starves in later weeks.

Confidence bands widen with the square root of the horizon and can be
recalibrated from realised errors (see below).

### The self-improvement loop

`VarianceLedger` records every `(forecast, actual, error)` triple once reality
catches up to a forecast week. This is both an honest accuracy report over time
and the mechanism for recalibrating the confidence bands — the "data exhaust"
that makes the forecast better the longer it runs.

## Repository layout

```
cashflow_forecast/
  synthetic.py      generate realistic messy SMB financials
  payment_model.py  invoice-level payment-timing prediction (the core model)
  forecast.py       13-week forecast engine + naive baseline
  variance.py       actual-vs-forecast tracking / accuracy reporting
  backtest.py       walk-forward evaluation (model vs baseline)
scripts/
  demo.py           end-to-end run; saves charts to outputs/
tests/
  test_pipeline.py  pytest suite
```

## Quickstart

```bash
pip install -r requirements.txt

# run the end-to-end demo (writes charts to ./outputs/)
python scripts/demo.py

# run the tests
pytest -q
```

### Using it in code

```python
from cashflow_forecast import (
    SyntheticFinancials, PaymentBehaviourModel, CashFlowForecaster,
)

data = SyntheticFinancials().generate_all()
model = PaymentBehaviourModel()
model.fit(data["invoices"])

forecaster = CashFlowForecaster(model, weeks=13)
forecast = forecaster.forecast(
    open_invoices=open_book,          # invoices not yet paid
    payments_history=data["payments"],
    recurring_costs=recurring_costs,
    opening_balance=100_000.0,
    start=pd.Timestamp("2024-06-01"),
    invoice_history=data["invoices"],
)

print(forecast.min_balance_week())    # projected cash trough
print(forecast.shortfall_weeks())     # any weeks below zero
```

## Design notes / honest limitations

- **The data is synthetic.** It's designed to reproduce the behaviours that make
  real forecasting hard — per-customer late-payment distributions, defaults,
  seasonality, lumpy costs — so the model's advantage over the baseline is
  meaningful. It is not a claim about any real business.
- **No leakage.** Customer history features use a shifted expanding window; the
  backtest trains only on invoices resolved before each forecast date and hides
  all post-forecast payment information.
- **The model is intentionally simple** (gradient boosting, engineered
  features). The point is the *problem framing* — invoice-level prediction
  feeding a cash forecast with a variance loop — not squeezing the last point of
  accuracy. Obvious extensions: an ensemble across model families, quantile
  regression for the bands, and calendar/holiday features.

## License

MIT
