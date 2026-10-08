# Freight Rate Prediction

Predicts `posted_rate` (USD) for freight loads. LightGBM on log(rate), time-based validation.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py   # validation experiments -> reports/experiment_results.json
python train.py             # final fit -> validation_predictions.csv + filled data/december_chart_inputs.csv
python make_report.py       # rebuilds reports/Freight_Rate_Report.pdf (needs reportlab)
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
Data goes in `data/` (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Approach
- **Split:** labeled data is Jan-Oct 2025, scoring data is Nov-Dec 2025, so I validate forward in time:
  expanding-window folds (train Jan-Apr -> test May-Jun, Jan-Jun -> Jul-Aug) plus an untouched holdout
  (train Jan-Aug -> test Sep-Oct). Random splits are not used: they leak the date-level market state.
- **Cold-start check:** validation contains 8 cities never seen in training, so I also hold out 8 whole cities.
- **Cleaning (row-preserving):** negative weights -> abs (sign errors), missing weights left as NaN + flag.
- **Features:** distance (raw, log, inverse), haversine, road/air ratio, weight, equipment, coordinates, direction.
- **Excluded on evidence:** `quote_signal` (regime-dependent; uninformative in Nov/Dec) and
  `market_index` (helped on May-Jun, hurt on Sep-Oct). See `reports/experiment_results.json`.
- **Model:** LightGBM, Huber loss on log(rate), 60-day recency half-life weights.
