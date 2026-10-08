"""Final fit on ALL labeled data, then score validation.csv and the December chart inputs.
Usage: python train.py   -> validation_predictions.csv, data/december_chart_inputs.csv (filled), model.joblib
"""
import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.data import load, clean, daily_market
from src.features import build_features
from src.modeling import metrics, fit_model, predict_rate

tr, va = clean(load(C.TRAIN_FILE)), clean(load(C.VALID_FILE))
daily = daily_market(tr, va)
y = tr[C.TARGET].values
X = build_features(tr, daily)

age = (tr.date.max() - tr.date).dt.days.values
model = fit_model(X, y, sample_weight=0.5 ** (age / C.RECENCY_HALF_LIFE_DAYS))
joblib.dump({"model": model, "features": list(X.columns)}, C.ROOT / "model.joblib")
print("in-sample fit (sanity only):", metrics(y, predict_rate(model, X)))

# ---- validation.csv: every row scored, template order preserved -----------------------------
Xv = build_features(va, daily)[X.columns]
assert not Xv.drop(columns=["weight"]).isna().any().any()
pred = predict_rate(model, Xv)
tmpl = pd.read_csv(C.VALID_TEMPLATE)
out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": va.load_id, "predicted_rate": pred}), on="load_id", how="left")
assert len(out) == 12000 and out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
out["predicted_rate"] = out["predicted_rate"].round(2)
out.to_csv(C.ROOT / "validation_predictions.csv", index=False)

# ---- December chart: fixed lane, only the date changes ---------------------------------------
dec = pd.read_csv(C.DEC_FILE, parse_dates=["date"])
cols = list(dec.columns)
pk = tr.loc[tr.pickup == "Lexington", ["pickup_lat", "pickup_lon"]].iloc[0]
dl = tr.loc[tr.delivery == "Fort Wayne", ["delivery_lat", "delivery_lon"]].iloc[0]
d = dec.copy()
for k, v in {**pk.to_dict(), **dl.to_dict()}.items():
    d[k] = v
d["weight_was_negative"], d["weight_missing"], d["market_index"], d["quote_signal"] = 0, 0, np.nan, np.nan
dec["predicted_rate"] = predict_rate(model, build_features(d, daily)[X.columns]).round(2)
dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
dec[cols].to_csv(C.DEC_FILE, index=False)
print(dec[["date", "predicted_rate"]].describe())
print("validation pred: mean rate/mile =", round(float((pred / va.distance).mean()), 3),
      "| train mean rate/mile =", round(float((y / tr.distance).mean()), 3))
