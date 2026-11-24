"""Zero-inflated Poisson baseline with gradient-boosted stages.

The forecast is the mean of a two-part count model,

    E[y | x] = (1 - pi(x)) * lambda(x),    pi(x) = P(y = 0 | x),    lambda(x) = E[y | x, y > 0].

pi is a LightGBM binary classifier of y == 0 and lambda a LightGBM Poisson regressor fit on the samples with y > 0.
Both are early-stopped on validation and then refit on training + validation with the selected number of rounds.
Features are those of PRiCE without the autoregressive backbone.

lambda must be fit on positive samples only: fitting it on all samples makes it an estimate of E[y | x] already,
and multiplying by (1 - pi) then discounts the zeros a second time. For the same reason the classifier is not
class-reweighted, so that pi is a probability. `positive_only=False, balance_classes=True` restores the earlier
behaviour."""

from __future__ import annotations

import numpy as np
import lightgbm as lgb

from price.data import lookback_mean

NAME = "ZIP"
NEEDS_GPU = False
LGB_THREADS = 4  # fixed so results do not depend on the machine
HPARAMS = {"michigan": dict(train_stride=4, num_leaves=31), "california": dict(train_stride=8, num_leaves=63)}


def build_features(out, weather, ts, windows, lookback, lookahead, weather_future=None):
    """One row per (window, horizon step, county): the PRiCE features without the backbone columns.

    Target-hour weather is read from `weather_future` when given (test time) and from `weather` otherwise.
    Returns X, y, and (county, window start, step) per row.
    """
    n_loc, n_hours = out.shape
    hours = ts.hour.values
    dows = ts.dayofweek.values
    months = ts.month.values
    calendar = np.concatenate(
        [
            np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], axis=1),
            np.stack([np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7)], axis=1),
            np.stack([np.sin(2 * np.pi * months / 12), np.cos(2 * np.pi * months / 12)], axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    w_future = weather if weather_future is None else weather_future
    loc_col = np.arange(n_loc, dtype=np.float32)
    rows, ys, meta = [], [], []
    for s in windows:
        end_lb = s + lookback
        if end_lb + lookahead > n_hours:
            continue
        y_lb = out[:, s:end_lb]
        y_last = y_lb[:, -1]
        y_mean = y_lb.mean(axis=1)
        y_max = y_lb.max(axis=1)
        y_std = y_lb.std(axis=1)
        run = np.zeros(n_loc, dtype=np.float32)
        for c in range(n_loc):
            ct = 0
            for v in y_lb[c, ::-1]:
                if v == 0:
                    ct += 1
                else:
                    break
            run[c] = ct
        w_lb_mean = lookback_mean(weather, s, end_lb)
        w_lb_max = weather[:, s:end_lb, :].max(axis=1)
        for tau in range(1, lookahead + 1):
            t = end_lb + tau - 1
            tau_col = np.full(n_loc, tau, dtype=np.float32)
            X = np.concatenate(
                [
                    loc_col[:, None],
                    tau_col[:, None],
                    y_last[:, None].astype(np.float32),
                    y_mean[:, None].astype(np.float32),
                    y_std[:, None].astype(np.float32),
                    y_max[:, None].astype(np.float32),
                    run[:, None],
                    w_lb_mean.astype(np.float32),
                    w_lb_max.astype(np.float32),
                    w_future[:, t, :].astype(np.float32),
                    np.broadcast_to(calendar[t], (n_loc, 6)).astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32)
            rows.append(X)
            ys.append(out[:, t].astype(np.float32))
            meta.append(np.stack([loc_col, np.full(n_loc, s, dtype=np.float32), tau_col], axis=1))
    return np.concatenate(rows), np.concatenate(ys), np.concatenate(meta).astype(np.int32)


def run(
    exp, *, seed=42, n_rounds=400, learning_rate=0.03, early_stopping=30, positive_only=True, balance_classes=False, **_
):
    """ZIP forecasts for the test targets, (counties, m * lookahead).

    `positive_only` fits the count stage on the samples with y > 0 and `balance_classes` reweights the classifier
    (see the module docstring).
    """
    np.random.seed(seed)
    hp = HPARAMS[exp.dataset]
    lb, la = exp.lookback, exp.lookahead
    train_end = exp.train_end
    val0, val1 = exp.val_range
    test0, _ = exp.test_range
    train_w = list(range(0, train_end - lb - la + 1, hp["train_stride"]))
    val_w = list(range(val0, val1 - lb - la + 1, 1))
    test_w = [test0 + b * la for b in range(exp.m)]
    Xtr, ytr, _ = build_features(exp.out, exp.weather, exp.ts, train_w, lb, la)
    Xva, yva, _ = build_features(exp.out, exp.weather, exp.ts, val_w, lb, la)
    Xte, _, mte = build_features(
        exp.out, exp.weather, exp.ts, test_w, lb, la, weather_future=exp.weather_with_forecast()
    )

    common = {
        "learning_rate": learning_rate,
        "num_leaves": hp["num_leaves"],
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
        "num_threads": LGB_THREADS,
    }
    cls_params = {"objective": "binary", "metric": "binary_logloss", **common}
    if balance_classes:
        cls_params["is_unbalance"] = True
    poi_params = {"objective": "poisson", "metric": "poisson", **common}
    stop = [lgb.early_stopping(early_stopping, verbose=False)]
    ztr = (ytr == 0).astype(np.int32)
    zva = (yva == 0).astype(np.int32)
    g_cls = lgb.train(
        cls_params, lgb.Dataset(Xtr, ztr), num_boost_round=n_rounds, valid_sets=[lgb.Dataset(Xva, zva)], callbacks=stop
    )
    ptr = ytr > 0 if positive_only else np.ones(len(ytr), bool)  # samples the count stage is fit on
    pva = yva > 0 if positive_only else np.ones(len(yva), bool)
    g_poi = lgb.train(
        poi_params,
        lgb.Dataset(Xtr[ptr], ytr[ptr]),
        num_boost_round=n_rounds,
        valid_sets=[lgb.Dataset(Xva[pva], yva[pva])],
        callbacks=stop,
    )

    Xtv = np.concatenate([Xtr, Xva])
    ytv = np.concatenate([ytr, yva])
    ztv = (ytv == 0).astype(np.int32)
    ptv = np.concatenate([ptr, pva])
    g_cls = lgb.train(cls_params, lgb.Dataset(Xtv, ztv), num_boost_round=g_cls.best_iteration)
    g_poi = lgb.train(poi_params, lgb.Dataset(Xtv[ptv], ytv[ptv]), num_boost_round=g_poi.best_iteration)
    pred = np.maximum((1 - g_cls.predict(Xte)) * g_poi.predict(Xte), 0.0)

    o = np.zeros((len(exp.locs), exp.m * la))
    o[mte[:, 0], (mte[:, 1] - test0) // la * la + mte[:, 2] - 1] = pred
    return o
