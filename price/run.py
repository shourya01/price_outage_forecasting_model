"""Run the methods of the main results table.

    python -m price.run --data-dir DATA --forecast-dir FORECASTS --out results

DATA holds michigan.nc and california.nc; FORECASTS holds one weather-forecast CSV per state and lookahead
(see price.data.find_forecast for the accepted layouts). Every (state, lookahead, method) is one task; its test
predictions go to <out>/<state>_<lookahead>/<method>/outages_forecast.csv (timestamp, location, pred) with a
metrics.json next to it. `python -m price.table` then builds the table."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

DATASETS = ["michigan", "california"]
LOOKAHEADS = [24, 48]


def run_task(dataset, lookahead, method, data_dir, forecast_dir, out_dir, seed, threads, device, force):
    """Runs in its own process. Returns a one-line summary."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    from price.data import build_experiment, find_forecast, predictions_frame, score
    from price.methods import METHODS

    target = Path(out_dir) / f"{dataset}_{lookahead}" / method
    tag = f"{dataset:10s} {lookahead}h {method:14s}"
    if (target / "metrics.json").is_file() and not force:
        m = json.load(open(target / "metrics.json"))
        if m.get("seed") == seed:
            return f"{tag} RMSE {m['rmse']:9.3f}  MAE {m['mae']:8.3f}   (cached)"
    t0 = time.time()
    exp = build_experiment(
        Path(data_dir) / f"{dataset}.nc", dataset, lookahead, find_forecast(forecast_dir, dataset, lookahead)
    )
    pred = METHODS[method].run(exp, seed=seed, n_jobs=threads, device=device)
    metrics = score(pred, exp.truth)
    metrics.update(
        dataset=dataset,
        lookahead=lookahead,
        method=method,
        label=METHODS[method].label,
        seed=seed,
        seconds=round(time.time() - t0, 1),
        n_counties=len(exp.locs),
        n_target_hours=int(pred.shape[1]),
        weather_variables=exp.features if METHODS[method].uses_weather else [],
        **exp.notes,
    )
    target.mkdir(parents=True, exist_ok=True)
    predictions_frame(exp, pred).to_csv(target / "outages_forecast.csv", index=False)
    tmp = target / "metrics.json.tmp"  # written last and atomically: its presence marks a finished task
    json.dump(metrics, open(tmp, "w"), indent=1)
    os.replace(tmp, target / "metrics.json")
    return f"{tag} RMSE {metrics['rmse']:9.3f}  MAE {metrics['mae']:8.3f}   ({metrics['seconds']:.0f}s)"


def _safe(args):
    """Call `run_task`; an exception becomes a 'FAILED' summary line so that the pool keeps going."""
    try:
        return run_task(*args)
    except Exception:
        return f"{args[0]} {args[1]}h {args[2]}: FAILED\n{traceback.format_exc()}"


def main(argv=None):
    """Command-line entry point: check the inputs, then run CPU tasks in parallel and GPU tasks one at a time."""
    from price.methods import METHODS, TABLE_ORDER

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="directory with michigan.nc and california.nc")
    ap.add_argument("--forecast-dir", required=True, help="directory with the weather-forecast CSVs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--lookaheads", nargs="+", type=int, default=LOOKAHEADS, choices=LOOKAHEADS)
    ap.add_argument("--methods", nargs="+", default=TABLE_ORDER, choices=TABLE_ORDER)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4), help="CPU tasks run at once")
    ap.add_argument("--threads", type=int, default=4, help="threads per task (LightGBM is pinned to 4 regardless)")
    ap.add_argument("--device", default=None, help="cuda or cpu for LSTM and TimesFM (default: cuda if available)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="rerun tasks that already have results")
    a = ap.parse_args(argv)

    from price.data import find_forecast

    for ds in a.datasets:  # fail early if an input is missing
        if not (Path(a.data_dir) / f"{ds}.nc").is_file():
            sys.exit(f"missing {Path(a.data_dir) / (ds + '.nc')}")
        for la in a.lookaheads:
            find_forecast(a.forecast_dir, ds, la)

    tasks = [
        (ds, la, m, a.data_dir, a.forecast_dir, a.out, a.seed, a.threads, a.device, a.force)
        for ds in a.datasets
        for la in a.lookaheads
        for m in a.methods
    ]
    cpu = [t for t in tasks if not METHODS[t[2]].gpu]
    gpu = [t for t in tasks if METHODS[t[2]].gpu]
    cpu.sort(key=lambda t: (t[0] != "california", t[2] != "tshm"))  # longest first
    print(f"{len(tasks)} tasks: {len(cpu)} on CPU ({a.jobs} at a time), {len(gpu)} on the GPU queue", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as cpu_pool, ProcessPoolExecutor(max_workers=1) as gpu_pool:
        futs = [cpu_pool.submit(_safe, t) for t in cpu] + [gpu_pool.submit(_safe, t) for t in gpu]
        for f in as_completed(futs):
            line = f.result()
            failed += "FAILED" in line
            print(line, flush=True)
    if failed:
        sys.exit(f"{failed} task(s) failed")


if __name__ == "__main__":
    main()
