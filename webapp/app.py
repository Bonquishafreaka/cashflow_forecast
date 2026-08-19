"""Cash-flow forecast dashboard (Flask).

A minimal web app on top of the forecasting engine: upload an invoices CSV
(and optionally a payments CSV), get a 13-week cash forecast with the projected
low point and any weeks the balance goes negative.

Run:
    python webapp/app.py
then open http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, request, send_from_directory

# allow running from repo root or from webapp/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cashflow_forecast.service import build_forecast  # noqa: E402
from cashflow_forecast.sources import (  # noqa: E402
    CSVDataSource,
    DataFrameSource,
    DataSourceError,
)
from cashflow_forecast.synthetic import SyntheticFinancials  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(HERE, "sample_data")

app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """Forecast from uploaded CSV(s). Expects multipart form with an
    'invoices' file and an optional 'payments' file."""
    if "invoices" not in request.files or request.files["invoices"].filename == "":
        return jsonify({"error": "Upload an invoices CSV to forecast."}), 400

    invoices_bytes = request.files["invoices"].read()
    payments_bytes = None
    if "payments" in request.files and request.files["payments"].filename:
        payments_bytes = request.files["payments"].read()

    weeks = int(request.form.get("weeks", 13))
    opening = request.form.get("opening_balance")
    opening_balance = float(opening) if opening not in (None, "") else None

    try:
        source = CSVDataSource(invoices_bytes, payments_bytes)
        result = build_forecast(
            source, weeks=weeks, opening_balance=opening_balance
        )
    except DataSourceError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:  # pragma: no cover - defensive
        return jsonify({"error": "Could not process this file. Check the format and try again."}), 400

    return jsonify(result)


@app.route("/api/sample", methods=["POST"])
def api_sample():
    """Forecast on the built-in sample business, so the app is explorable with
    one click and no upload."""
    data = SyntheticFinancials().generate_all()
    source = DataFrameSource(data["invoices"], data["payments"])
    weeks = int(request.form.get("weeks", 13))
    result = build_forecast(source, weeks=weeks)
    result["is_sample"] = True
    return jsonify(result)


@app.route("/sample_data/<path:name>")
def sample_file(name):
    """Let users download the sample CSVs to see the expected format."""
    return send_from_directory(SAMPLE_DIR, name, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
