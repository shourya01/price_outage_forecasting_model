"""Regression script: run one method of the main table and print its test RMSE and MAE.

    python tests/regress_price.py <method> <data-dir> <forecast-dir> [key=value ...]

Options: exps=michigan:24,california:48,... (default: all four), jobs=N (default 4), clip=0|1 (clip forecast
weather to physical bounds, default 1), fix=0|1 (fill the all-zero weather hours; the default here is 1, whereas
the paper's tables use the weather as stored, i.e. fix=0), cafeats=a+b+... (weather variables for California).
Run from the repository root. It trains models and needs the datasets and the weather forecasts."""

import sys, time

sys.path.insert(0, ".")
from price.data import build_experiment, find_forecast, score
from price.methods import METHODS

method, data_dir, fc_dir = sys.argv[1], sys.argv[2], sys.argv[3]
opts = dict(kv.split("=") for kv in sys.argv[4:])
fix = opts.get("fix", "1") == "1"
clip = opts.get("clip", "1") == "1"
exps = [tuple(e.split(":")) for e in opts.get("exps", "michigan:24,michigan:48,california:24,california:48").split(",")]
mod = METHODS[method]
feats = {"california": opts["cafeats"].split("+")} if "cafeats" in opts else {}
for ds, la in exps:
    t = time.time()
    exp = build_experiment(
        f"{data_dir}/{ds}.nc",
        ds,
        int(la),
        find_forecast(fc_dir, ds, int(la)),
        fix_missing_hours=fix,
        clip_forecast=clip,
        features=feats.get(ds),
    )
    pred = mod.run(exp, n_jobs=int(opts.get("jobs", 4)))
    s = score(pred, exp.truth)
    print(
        f"{method:10s} {ds:10s} {la}h  RMSE {s['rmse']:.3f}  MAE {s['mae']:.3f}   [{time.time() - t:.0f}s; held={exp.notes['held_at_last_value']}; filled={len(exp.notes.get('missing_hours_filled', []))}]",
        flush=True,
    )
