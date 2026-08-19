"""Invoice-level payment-timing prediction.

This is the core differentiator: instead of applying a single average payment
term across the whole AR book, we predict *per invoice* when (and whether) it
will actually be paid, using each customer's learned paying behaviour plus
invoice features.

Two heads:
  * a regressor  -> predicted days-late relative to due date
  * a classifier -> probability the invoice is paid at all (vs. defaulted)

The expected paid-date for an open invoice is then
    due_date + predicted_days_late,
weighted by the payment probability when we roll invoices up into a forecast.

We deliberately avoid leaking the future: a customer's historical statistics
are computed only from invoices that were *already paid before* the invoice in
question was issued (see `_add_customer_history`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


FEATURE_COLUMNS = [
    "amount",
    "net_terms",
    "issue_month",
    "issue_dow",
    "cust_hist_mean_late",
    "cust_hist_std_late",
    "cust_hist_count",
    "cust_hist_default_rate",
    "size_small",
    "size_mid",
    "size_large",
]


@dataclass
class PaymentModelReport:
    late_mae: float
    paid_auc: float
    n_train: int
    n_test: int

    def __str__(self) -> str:
        return (
            f"PaymentModel  |  days-late MAE: {self.late_mae:.2f}  |  "
            f"paid AUC: {self.paid_auc:.3f}  |  "
            f"train/test: {self.n_train}/{self.n_test}"
        )


class PaymentBehaviourModel:
    def __init__(self, horizon_days: int = 90, random_state: int = 0):
        self.horizon_days = horizon_days
        self.random_state = random_state
        self.reg = GradientBoostingRegressor(random_state=random_state)
        self.clf = GradientBoostingClassifier(random_state=random_state)
        self._clf_constant: float | None = None
        self._fitted = False

    def _paid_prob(self, X: pd.DataFrame) -> np.ndarray:
        """Payment probability, robust to a classifier trained on one class."""
        if self._clf_constant is not None:
            return np.full(len(X), self._clf_constant)
        return self.clf.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # feature engineering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
        """For each invoice, attach the paying stats of that customer computed
        ONLY from invoices issued earlier (expanding, shifted) to avoid leakage.
        """
        df = df.sort_values("issue_date").copy()
        df["_paid_flag"] = df["paid_date"].notna().astype(float)
        # per-invoice observed lateness (NaN when unpaid)
        df["_late"] = df["days_late"].astype(float)

        out = []
        for _cust, g in df.groupby("customer_id", sort=False):
            g = g.sort_values("issue_date").copy()
            # expanding stats shifted by 1 so the current row is excluded
            g["cust_hist_mean_late"] = (
                g["_late"].expanding().mean().shift(1)
            )
            g["cust_hist_std_late"] = (
                g["_late"].expanding().std().shift(1)
            )
            g["cust_hist_count"] = (
                g["_paid_flag"].expanding().count().shift(1)
            )
            g["cust_hist_default_rate"] = (
                (1.0 - g["_paid_flag"]).expanding().mean().shift(1)
            )
            out.append(g)
        df = pd.concat(out).sort_values("issue_date")

        # cold-start fallbacks for a customer's first invoices
        df["cust_hist_mean_late"] = df["cust_hist_mean_late"].fillna(7.0)
        df["cust_hist_std_late"] = df["cust_hist_std_late"].fillna(5.0)
        df["cust_hist_count"] = df["cust_hist_count"].fillna(0.0)
        df["cust_hist_default_rate"] = df["cust_hist_default_rate"].fillna(0.02)
        return df

    @staticmethod
    def _add_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["issue_month"] = df["issue_date"].dt.month
        df["issue_dow"] = df["issue_date"].dt.dayofweek
        for size in ("small", "mid", "large"):
            df[f"size_{size}"] = (df["customer_size"] == size).astype(int)
        return df

    def build_features(self, invoices: pd.DataFrame) -> pd.DataFrame:
        df = self._add_customer_history(invoices)
        df = self._add_invoice_features(df)
        return df

    # ------------------------------------------------------------------ #
    # fit / evaluate
    # ------------------------------------------------------------------ #
    def fit(
        self, invoices: pd.DataFrame, cutoff: pd.Timestamp | None = None
    ) -> PaymentModelReport:
        """Train on invoices issued before `cutoff`, evaluate on those after.

        If cutoff is None, uses a chronological 80/20 split.
        """
        feat = self.build_features(invoices)

        if cutoff is None:
            cutoff = feat["issue_date"].quantile(0.8)

        train = feat[feat["issue_date"] < cutoff].copy()
        test = feat[feat["issue_date"] >= cutoff].copy()

        # classifier target: was it paid within horizon of due date?
        def paid_within_horizon(frame: pd.DataFrame) -> pd.Series:
            paid = frame["paid_date"].notna() & (
                (frame["paid_date"] - frame["due_date"]).dt.days
                <= self.horizon_days
            )
            return paid.astype(int)

        y_train_paid = paid_within_horizon(train)
        y_test_paid = paid_within_horizon(test)

        # regressor trains only on invoices actually paid (observed lateness)
        train_paid = train[train["paid_date"].notna()]
        test_paid = test[test["paid_date"].notna()]

        # classifier needs >=2 classes; fall back to a constant rate otherwise
        if y_train_paid.nunique() > 1:
            self.clf.fit(train[FEATURE_COLUMNS], y_train_paid)
            self._clf_constant = None
        else:
            self._clf_constant = float(y_train_paid.iloc[0])
        self.reg.fit(
            train_paid[FEATURE_COLUMNS], train_paid["days_late"].astype(float)
        )
        self._fitted = True

        # evaluation (may be empty in walk-forward mode where cutoff == asof)
        if len(test_paid) > 0:
            late_pred = self.reg.predict(test_paid[FEATURE_COLUMNS])
            late_mae = float(
                mean_absolute_error(
                    test_paid["days_late"].astype(float), late_pred
                )
            )
        else:
            late_mae = float("nan")
        # AUC needs both classes present and a non-empty test set
        if len(test) > 0 and y_test_paid.nunique() > 1:
            paid_prob = self._paid_prob(test[FEATURE_COLUMNS])
            paid_auc = float(roc_auc_score(y_test_paid, paid_prob))
        else:
            paid_auc = float("nan")

        return PaymentModelReport(
            late_mae=late_mae,
            paid_auc=paid_auc,
            n_train=len(train),
            n_test=len(test),
        )

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #
    def predict_invoice(self, invoices: pd.DataFrame) -> pd.DataFrame:
        """Return per-invoice predicted paid-date and payment probability.

        Accepts raw invoice rows (must include the columns used in feature
        construction). Safe to call on open/unpaid invoices.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before predict_invoice().")
        feat = self.build_features(invoices)
        pred_late = self.reg.predict(feat[FEATURE_COLUMNS])
        pred_paid_prob = self._paid_prob(feat[FEATURE_COLUMNS])

        result = feat[["invoice_id", "customer_id", "amount", "due_date"]].copy()
        result["pred_days_late"] = pred_late
        result["pred_paid_prob"] = pred_paid_prob
        result["pred_paid_date"] = result["due_date"] + pd.to_timedelta(
            np.round(pred_late), unit="D"
        )
        # expected (probability-weighted) cash landing on the predicted date
        result["expected_amount"] = result["amount"] * result["pred_paid_prob"]
        return result

    def feature_importance(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Model must be fit first.")
        clf_imp = (
            self.clf.feature_importances_
            if self._clf_constant is None
            else np.full(len(FEATURE_COLUMNS), np.nan)
        )
        return (
            pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "reg_importance": self.reg.feature_importances_,
                    "clf_importance": clf_imp,
                }
            )
            .sort_values("reg_importance", ascending=False)
            .reset_index(drop=True)
        )
