"""Ablation table of PRiCE (test RMSE per variant).

    python -m price.ablation --data-dir DATA --forecast-dir FORECASTS --out results [--seeds 42 1 2 ...]

Variants
  component   persistence expert alone; cluster expert alone
  backbone    ARIMA(1,0,1) or AR(1) in place of AR(2)
  aggregator  arithmetic mean; convex combination with the weight tuned on validation
  weather     target-hour weather replaced by the last observed weather (in training and at test time)
  full        PRiCE

Four models are trained per state, lookahead and seed (full, each backbone, weather); the component and
aggregator variants are read off the full model. Metrics go to
<out>/<state>_<lookahead>/ablation/seed_<seed>/<variant>/metrics.json, predictions are saved for the first seed, and
<out>/ablation_table.{md,tex,csv} reports the mean over seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DATASETS = ["michigan", "california"]
LOOKAHEADS = [24, 48]
COLUMNS = [("michigan", 24), ("michigan", 48), ("california", 24), ("california", 48)]
# training run -> variants it yields (component key or None for the run's forecast)
RUNS = {
    "full": {
        "full": "price",
        "persistence_only": "persistence",
        "cluster_only": "cluster_expert",
        "arithmetic_mean": "arithmetic_mean",
        "convex": "convex",
    },
    "backbone_arima": {"backbone_arima": None},
    "backbone_ar1": {"backbone_ar1": None},
    "weather_persistence": {"weather_persistence": None},
}
RUN_KWARGS = {
    "full": dict(return_components=True),
    "backbone_arima": dict(ar_order="arima11"),
    "backbone_ar1": dict(ar_order=1),
    "weather_persistence": dict(persist_weather=True),
}
ROWS = [  # (group, variant, markdown label, LaTeX label)
    ("Component (drops one expert)", "persistence_only", "Persistence expert alone", r"$\ep$ alone"),
    ("Component (drops one expert)", "cluster_only", "Cluster expert alone", r"$\ec$ alone"),
    ("Backbone (replaces AR(2))", "backbone_arima", "ARIMA(1,0,1)", "ARIMA(1,0,1)"),
    ("Backbone (replaces AR(2))", "backbone_ar1", "AR(1)", "AR(1)"),
    ("Aggregator (replaces the geometric mean)", "arithmetic_mean", "Arithmetic mean", "Arithmetic mean"),
    (
        "Aggregator (replaces the geometric mean)",
        "convex",
        "Convex (weight tuned on validation)",
        r"Convex (val-tuned $\alpha$)",
    ),
    (
        "Weather (replaces the forecast)",
        "weather_persistence",
        "Last observed weather",
        r"Persistence ($\hat{\mb{w}}_{i,t+\tau}=\mb{w}_{i,t}$)",
    ),
    ("", "full", "PRiCE (full)", r"\textbf{\price\ (full)}"),
]
TEX_GROUP = {
    "Component (drops one expert)": r"\emph{Component (drops one expert)}",
    "Backbone (replaces AR(2))": r"\emph{Backbone (replaces $\eb$)}",
    "Aggregator (replaces the geometric mean)": r"\emph{Aggregator (replaces $\sqrt{\cdot\,\cdot}$)}",
    "Weather (replaces the forecast)": r"\emph{Weather (replaces $\hat{\mb{w}}_{i,t+\tau}$)}",
}


def _arima_backbone(exp, cache: Path, force: bool):
    """The ARIMA(1,0,1) backbone does not depend on the seed; fit it once per experiment and reuse it.
    The cache is tied to the training outages by a checksum, so a different data file refits."""
    import hashlib
    from price.methods.price import fit_backbone

    key = hashlib.sha1(np.ascontiguousarray(exp.out[:, : exp.train_end]).tobytes()).hexdigest()
    if cache.is_file() and not force:
        z = np.load(cache)
        if "key" in z.files and str(z["key"]) == key:
            return {
                "order": "arima11",
                "phi1": z["phi1"],
                "phi2": z["phi2"],
                "theta": z["theta"],
                "mu": z["mu"],
                "resid": z["resid"],
            }
    bb = fit_backbone(exp.out, exp.train_end, "arima11")
    tmp = cache.with_name(cache.stem + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, key=key, **{k: v for k, v in bb.items() if k != "order"})
    os.replace(tmp, cache)
    return bb


def run_task(dataset, lookahead, run_name, seed, data_dir, forecast_dir, out_dir, threads, save_pred, force):
    """Train one ablation run for a state, lookahead and seed and write metrics.json for every variant it yields.

    Runs in its own process and returns a one-line summary. Predictions are saved only when `save_pred` is set.
    """
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(threads)
    from price.data import build_experiment, find_forecast, predictions_frame, score
    from price.methods import price as model

    base = Path(out_dir) / f"{dataset}_{lookahead}" / "ablation"
    sdir = base / f"seed_{seed}"
    variants = RUNS[run_name]
    tag = f"{dataset:10s} {lookahead}h seed {seed:<3d} {run_name:20s}"
    if not force and all((sdir / v / "metrics.json").is_file() for v in variants):
        return f"{tag} (cached)"
    t0 = time.time()
    exp = build_experiment(
        Path(data_dir) / f"{dataset}.nc", dataset, lookahead, find_forecast(forecast_dir, dataset, lookahead)
    )
    kw = dict(RUN_KWARGS[run_name])
    if run_name == "backbone_arima":
        base.mkdir(parents=True, exist_ok=True)
        kw["backbone"] = _arima_backbone(exp, base / "_arima_backbone.npz", force)
    res = model.run(exp, seed=seed, **kw)
    parts = []
    for variant, key in variants.items():
        pred = res if key is None else res[key]
        m = score(pred, exp.truth)
        m.update(dataset=dataset, lookahead=lookahead, variant=variant, seed=seed, seconds=round(time.time() - t0, 1))
        if variant == "convex":
            m["alpha"] = res["convex_alpha"]
        d = sdir / variant
        d.mkdir(parents=True, exist_ok=True)
        if save_pred:
            predictions_frame(exp, pred).to_csv(d / "outages_forecast.csv", index=False)
        json.dump(m, open(d / "metrics.json.tmp", "w"), indent=1)
        os.replace(d / "metrics.json.tmp", d / "metrics.json")
        parts.append(f"{variant} {m['rmse']:.3f}")
    return f"{tag} {'  '.join(parts)}   ({time.time() - t0:.0f}s)"


def _safe(args):
    """Call `run_task`; an exception becomes a 'FAILED' summary line so that the pool keeps going."""
    try:
        return run_task(*args)
    except Exception:
        return f"{args[0]} {args[1]}h {args[2]} seed {args[3]}: FAILED\n{traceback.format_exc()}"


def build_table(out_dir, seeds=None):
    """Collect the per-variant test RMSE under `out_dir` (only the given `seeds`, if any) and write
    ablation_table.{md,tex,csv}.

    With one seed the cells are that seed's RMSE. With several, the Markdown table shows mean and standard deviation
    over seeds, the LaTeX table the mean, and the CSV both.
    """
    out = Path(out_dir)
    vals: dict = {}
    for f in out.glob("*/ablation/seed_*/*/metrics.json"):
        m = json.load(open(f))
        if seeds is not None and m["seed"] not in seeds:
            continue
        vals.setdefault((m["variant"], m["dataset"], m["lookahead"]), {})[m["seed"]] = m["rmse"]
    if not vals:
        raise SystemExit(f"no ablation metrics under {out}")
    n_seeds = max(len(v) for v in vals.values())
    mean = lambda v, c: float(np.mean(list(vals[(v, *c)].values()))) if (v, *c) in vals else None
    std = lambda v, c: (
        float(np.std(list(vals[(v, *c)].values()), ddof=1)) if (v, *c) in vals and len(vals[(v, *c)]) > 1 else None
    )
    head = ["Variant"] + [f"{d[:2].upper()} {la}h" for d, la in COLUMNS]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = [r"\begin{tabular}{lcccc}", r"\toprule", "Variant & " + " & ".join(head[1:]) + r" \\"]
    csv = ["variant," + ",".join(f"{d}_{la}_rmse_mean,{d}_{la}_rmse_std" for d, la in COLUMNS)]
    group = None
    for g, v, label, tex_label in ROWS:
        if g != group:
            tex.append(r"\midrule")
            if g:
                md.append(f"| *{g}* |" + " |" * len(COLUMNS))
                tex.append(rf"\multicolumn{{5}}{{l}}{{{TEX_GROUP[g]}}} \\")
            group = g
        cells_md, cells_tex, cells_csv = [], [], []
        for c in COLUMNS:
            mu, sd = mean(v, c), std(v, c)
            if mu is None:
                cells_md.append("--")
                cells_tex.append("--")
                cells_csv += ["", ""]
                continue
            cells_md.append(
                f"{mu:.1f} ± {sd:.1f}" if sd is not None and sd > 0 else f"{mu:.3f}" if sd is None else f"{mu:.1f}"
            )
            cells_tex.append(rf"\textbf{{{mu:.3f}}}" if v == "full" else f"{mu:.3f}")
            cells_csv += [f"{mu:.3f}", "" if sd is None else f"{sd:.3f}"]
        md.append(f"| {'**' + label + '**' if v == 'full' else label} | " + " | ".join(cells_md) + " |")
        tex.append(f"{tex_label} & " + " & ".join(cells_tex) + r" \\")
        csv.append(",".join([v] + cells_csv))
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "ablation_table.md").write_text("\n".join(md) + "\n")
    (out / "ablation_table.tex").write_text("\n".join(tex) + "\n")
    (out / "ablation_table.csv").write_text("\n".join(csv) + "\n")
    print("\n".join(md))
    print(
        f"\nTest RMSE per county, {'mean ± std over ' + str(n_seeds) + ' seeds' if n_seeds > 1 else 'one seed'}. "
        f"Written to {out}/ablation_table.{{md,tex,csv}}"
    )


def main(argv=None):
    """Command-line entry point: run the trainings (unless --table-only), then build the table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="directory with michigan.nc and california.nc")
    ap.add_argument("--forecast-dir", help="directory with the weather-forecast CSVs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--lookaheads", nargs="+", type=int, default=LOOKAHEADS, choices=LOOKAHEADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4), help="trainings run at once")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rerun trainings that already have results")
    ap.add_argument("--table-only", action="store_true", help="only rebuild the table from existing metrics")
    a = ap.parse_args(argv)
    if not a.table_only:
        if not a.data_dir or not a.forecast_dir:
            ap.error("--data-dir and --forecast-dir are required")
        from price.data import find_forecast

        for ds in a.datasets:
            if not (Path(a.data_dir) / f"{ds}.nc").is_file():
                sys.exit(f"missing {Path(a.data_dir) / (ds + '.nc')}")
            for la in a.lookaheads:
                find_forecast(a.forecast_dir, ds, la)
        tasks = [
            (ds, la, r, s, a.data_dir, a.forecast_dir, a.out, a.threads, s == a.seeds[0], a.force)
            for ds in sorted(a.datasets)
            for la in a.lookaheads
            for s in a.seeds
            for r in RUNS
        ]
        print(f"{len(tasks)} trainings, {a.jobs} at a time", flush=True)
        failed = 0
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            for f in as_completed([pool.submit(_safe, t) for t in tasks]):
                line = f.result()
                failed += "FAILED" in line
                print(line, flush=True)
        if failed:
            sys.exit(f"{failed} training(s) failed")
    build_table(a.out, a.seeds)


if __name__ == "__main__":
    main()
