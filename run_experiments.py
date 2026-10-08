"""All split / validation experiments behind the final model choice.
Usage: python run_experiments.py   -> writes reports/experiment_results.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src import config as C
from src.data import load, clean, daily_market
from src.features import build_features
from src.modeling import metrics, fit_model, predict_rate

tr, va = clean(load(C.TRAIN_FILE)), clean(load(C.VALID_FILE))
daily = daily_market(tr, va)
y = tr[C.TARGET].values
X_final = build_features(tr, daily)                                   # production feature set
X_mkt = build_features(tr, daily, use_market=True)                    # + market_index features
X_qs = build_features(tr, daily, use_market=True, use_quote_signal=True)
month = tr.date.dt.month.values
FOLDS = C.TIME_FOLDS + [C.HOLDOUT]
res = {}

def win(a, b):
    return ((tr.date >= a) & (tr.date <= b)).values

def weights(tri, anchor, hl):
    if not hl:
        return None
    return 0.5 ** ((pd.Timestamp(anchor) - tr.date[tri]).dt.days.values / hl)

def run(X, hl=None, params=None):
    out = []
    for a, b, c, d in FOLDS:
        tri, tei = win(a, b), win(c, d)
        p = predict_rate(fit_model(X[tri], y[tri], params, weights(tri, b, hl)), X[tei])
        out.append({**metrics(y[tei], p), "test": f"{c}..{d}"})
    return out

# 0. Label-free regime diagnostic for quote_signal
res["quote_signal_corr_with_log_distance"] = {
    f"{n}-{m:02d}": round(float(np.corrcoef(g.quote_signal, np.log(g.distance))[0, 1]), 3)
    for n, df in (("train", tr), ("validation", va)) for m, g in df.groupby(df.date.dt.month)}

# 1. Feature-set comparison across forward-chaining folds
res["folds_final_features(no_market,no_qs)"] = run(X_final)
res["folds_plus_market_index"] = run(X_mkt)
res["folds_plus_market_and_quote_signal"] = run(X_qs)

# 1b. Regime-matched check: Aug is the only month whose quote_signal regime resembles Nov/Dec
tri, tei = win("2025-01-01", "2025-07-31"), win("2025-08-01", "2025-08-31")
res["regime_matched_Aug"] = {n: metrics(y[tei], predict_rate(fit_model(X[tri], y[tri]), X[tei]))
                             for n, X in (("final", X_final), ("with_market_and_qs", X_qs))}

# 2. Recency weighting + hyper-parameter sensitivity (final features)
sweep = {}
for hl in (None, 30, 60, 120):
    r = run(X_final, hl=hl); sweep[f"half_life={hl}"] = {"fold_MAE": [x["MAE"] for x in r],
                                                         "mean_MAE": round(float(np.mean([x["MAE"] for x in r])), 1)}
for n, prm in {"num_leaves=15": {"num_leaves": 15}, "num_leaves=63": {"num_leaves": 63},
               "objective=l1": {"objective": "regression_l1"}, "n_estimators=600": {"n_estimators": 600}}.items():
    r = run(X_final, params=prm); sweep[n] = {"fold_MAE": [x["MAE"] for x in r],
                                              "mean_MAE": round(float(np.mean([x["MAE"] for x in r])), 1)}
res["sweep"] = sweep
res["final_config_on_folds"] = run(X_final, hl=C.RECENCY_HALF_LIFE_DAYS)

# 3. Cold-start: hold out 8 whole cities (validation contains 8 unseen cities)
rng = np.random.RandomState(C.SEED)
cities = np.array(sorted(set(tr.pickup) | set(tr.delivery)))
cold = []
for _ in range(3):
    held = set(rng.choice(cities, 8, replace=False))
    touch = (tr.pickup.isin(held) | tr.delivery.isin(held)).values
    cold.append(metrics(y[touch], predict_rate(fit_model(X_final[~touch], y[~touch]), X_final[touch])))
res["cold_start_8_unseen_cities"] = cold

# 4. Feature importance (gain), final features, trained on the holdout train window
tri = win(*C.HOLDOUT[:2])
m = fit_model(X_final[tri], y[tri])
imp = pd.Series(m.booster_.feature_importance("gain"), index=X_final.columns)
res["feature_importance_gain_%"] = (imp / imp.sum() * 100).round(2).sort_values(ascending=False).to_dict()

C.REPORTS.mkdir(exist_ok=True)
(C.REPORTS / "experiment_results.json").write_text(json.dumps(res, indent=2))
for k in ("folds_final_features(no_market,no_qs)", "folds_plus_market_index", "folds_plus_market_and_quote_signal"):
    print(k, [(x["test"][:7], x["MAE"], x["MedAPE_%"]) for x in res[k]])
print("sweep", json.dumps(sweep))
print("final", res["final_config_on_folds"])
print("regime-matched Aug", res["regime_matched_Aug"])
print("cold-start", res["cold_start_8_unseen_cities"])
