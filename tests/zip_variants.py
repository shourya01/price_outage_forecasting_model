"""Regression script: two non-default variants of the ZIP baseline on one experiment.

    python tests/zip_variants.py <data-dir> <forecast-dir> <state> <lookahead>

The variants are the earlier behaviour (count stage fit on all samples, reweighted classifier) and the count stage
fit on positive samples with a reweighted classifier. The default of price.methods.zip (positives only, unweighted
classifier) is the one in the main table.
Run from the repository root. It trains models and needs the datasets and the weather forecasts."""

import sys, time

sys.path.insert(0, ".")
from price.data import build_experiment, find_forecast, score
from price.methods import zip as Z

data_dir, fc_dir, ds, la = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
exp = build_experiment(f"{data_dir}/{ds}.nc", ds, la, find_forecast(fc_dir, ds, la))
for name, kw in [
    ("old (all samples, reweighted classifier)", dict(positive_only=False, balance_classes=True)),
    ("positives only, reweighted classifier", dict(positive_only=True, balance_classes=True)),
]:
    t = time.time()
    p = Z.run(exp, **kw)
    s = score(p, exp.truth)
    print(
        f"{ds} {la}h  {name:42s} RMSE {s['rmse']:.3f} MAE {s['mae']:.3f}  mean pred {p.mean():.1f} vs truth {exp.truth.mean():.1f}  [{time.time() - t:.0f}s]",
        flush=True,
    )
