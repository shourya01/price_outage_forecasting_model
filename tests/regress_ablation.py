"""Regression script: PRiCE ablation variants on chosen experiments.

    python tests/regress_ablation.py <data-dir> <forecast-dir> <state>:<lookahead> [<state>:<lookahead> ...]

Prints the test RMSE of the full model, its two experts and the alternative aggregators, then of the AR(1) backbone
(here with its residual feature, as in the original script), the ARIMA(1,0,1) backbone and the weather variant.
Run from the repository root. It trains models and needs the datasets and the weather forecasts."""

import sys, time

sys.path.insert(0, ".")
from price.data import build_experiment, find_forecast, score
from price.methods import price as M

data_dir, fc_dir = sys.argv[1], sys.argv[2]
for spec in sys.argv[3:]:
    ds, la = spec.split(":")
    la = int(la)
    exp = build_experiment(f"{data_dir}/{ds}.nc", ds, la, find_forecast(fc_dir, ds, la), fix_missing_hours=False)
    t = time.time()
    r = M.run(exp, return_components=True)
    print(
        f"{ds} {la}h  persist {score(r['persistence'], exp.truth)['rmse']:.3f}  cluster {score(r['cluster_expert'], exp.truth)['rmse']:.3f}  arith {score(r['arithmetic_mean'], exp.truth)['rmse']:.3f}  convex(a={r['convex_alpha']:.2f}) {score(r['convex'], exp.truth)['rmse']:.3f}  full {score(r['price'], exp.truth)['rmse']:.3f}  [{time.time() - t:.0f}s]",
        flush=True,
    )
    for name, kw in [
        ("ar1 (with residual feature)", dict(ar_order=1, residual_feature=True)),
        ("arima11", dict(ar_order="arima11")),
        ("weather persistence", dict(persist_weather=True)),
    ]:
        t = time.time()
        s = score(M.run(exp, **kw), exp.truth)
        print(f"   {name:28s} RMSE {s['rmse']:.3f} MAE {s['mae']:.3f}  [{time.time() - t:.0f}s]", flush=True)
