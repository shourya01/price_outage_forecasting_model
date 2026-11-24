"""PRiCE: persistence expert x cluster expert, combined by a shifted geometric mean.

* Persistence expert E_p: the last observed outage count of each county, held over the horizon.
* Cluster expert E_c: a per-county autoregressive backbone (AR(2) by Yule-Walker, fit on the training split) plus a
  gradient-boosted residual regressor (absolute-error loss) shared by the counties of a behavioural cluster.
* Counties are clustered by k-means on seven training-split statistics; K is chosen by silhouette.
* Aggregator: y_hat = max( sqrt((E_p + kappa) (E_c + kappa)) - kappa, 0 ) with kappa = 0.5. The shift kappa keeps
  the geometric mean from collapsing to zero whenever one expert predicts zero outages (a plain geometric mean of
  counts would return 0 in that case, whatever the other expert says); it is fixed, not tuned.
"""

from __future__ import annotations

import warnings

import numpy as np
import lightgbm as lgb

from price.data import lookback_mean

NAME = "PRiCE"
NEEDS_GPU = False
HPARAMS = {  # per dataset
    "michigan": dict(train_stride=4, num_leaves=31),
    "california": dict(train_stride=8, num_leaves=63),
}
K_GRID = (3, 4, 5, 6, 8)
KAPPA = 0.5  # shift of the geometric mean (see the module docstring)
LGB_THREADS = 4  # fixed so results do not depend on the machine


# ------------------------------------------------------------------ clustering


def cluster_features(out, weather_all, train_end):
    """Seven per-county statistics on the training split: log-mean, log-p99, zero fraction, lag-1 autocorrelation,
    mean of the five largest |correlations| with weather, log-std, and the share of hours above the p99."""
    n_loc = out.shape[0]
    feats = np.zeros((n_loc, 7), dtype=np.float64)
    for c in range(n_loc):
        y = out[c, :train_end].astype(np.float64)
        feats[c, 0] = np.log1p(y.mean())
        feats[c, 1] = np.log1p(np.percentile(y, 99))
        feats[c, 2] = (y == 0).mean()
        if y.std() > 0:
            a = y[:-1] - y.mean()
            b = y[1:] - y.mean()
            denom = (a * a).sum()
            feats[c, 3] = float((a * b).sum() / denom) if denom > 0 else 0.0
        corrs = []
        for f in range(weather_all.shape[2]):
            x = weather_all[c, :train_end, f].astype(np.float64)
            if x.std() <= 0 or y.std() <= 0:
                continue
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r):
                corrs.append(abs(r))
        corrs = sorted(corrs, reverse=True)
        feats[c, 4] = np.mean(corrs[:5]) if len(corrs) >= 1 else 0.0
        feats[c, 5] = np.log1p(y.std())
        feats[c, 6] = float((y > np.percentile(y, 99)).mean())
    return feats


def kmeans(X, K, n_iter=100, seed=42):
    """k-means on the rows of X with k-means++ initialisation. Returns (assignment, centers)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    centers_idx = [int(rng.randint(n))]  # k-means++ initialisation
    for _ in range(K - 1):
        dists = np.array([min(np.linalg.norm(X[i] - X[j]) ** 2 for j in centers_idx) for i in range(n)])
        if dists.sum() == 0:
            break
        centers_idx.append(int(rng.choice(n, p=dists / dists.sum())))
    centers = X[centers_idx].copy()
    for _ in range(n_iter):
        assn = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = assn == k
            new_centers[k] = X[mask].mean(axis=0) if mask.sum() > 0 else centers[k]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return assn, centers


def silhouette(X, assn):
    """Mean silhouette coefficient of the assignment `assn` (Euclidean distance); 0 with fewer than two clusters."""
    n = len(X)
    K = len(np.unique(assn))
    if K < 2 or n < 2:
        return 0.0
    s = []
    for i in range(n):
        own = [np.linalg.norm(X[i] - X[j]) for j in range(n) if assn[j] == assn[i] and j != i]
        a = np.mean(own) if (assn == assn[i]).sum() > 1 else 0
        bs = [
            np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n) if assn[j] == k])
            for k in range(K)
            if k != assn[i] and (assn == k).sum() > 0
        ]
        b = min(bs) if bs else a
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def fit_clusters(out, weather_all, train_end, k_grid=K_GRID):
    """Cluster the counties on their standardized training-split statistics. K is the value in `k_grid` with the
    largest silhouette. Returns (cluster index per county, K).
    """
    feats = cluster_features(out, weather_all, train_end)
    Z = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    best = None
    for K in k_grid:
        if K >= len(Z):
            continue
        assn, _ = kmeans(Z, K)
        sil = silhouette(Z, assn)
        if best is None or sil > best[0]:
            best = (sil, K, assn)
    return best[2].astype(np.int32), int(best[1])


# ------------------------------------------------------------------ per-county backbone


def fit_ar1(y):
    """AR(1) around the series mean: the lag-1 least-squares ratio, clipped to [-0.99, 0.99]. Returns (phi, mu)."""
    if y.std() == 0:
        return 0.0, float(y.mean())
    mu = float(y.mean())
    a = y[:-1] - mu
    b = y[1:] - mu
    denom = (a * a).sum()
    phi = float((a * b).sum() / denom) if denom > 0 else 0.0
    return max(-0.99, min(0.99, phi)), mu


def fit_ar2(y):
    """AR(2) by Yule-Walker; falls back to AR(1) when the estimate is not stationary. Returns (phi1, phi2, mu)."""
    if y.std() == 0:
        return 0.0, 0.0, float(y.mean())
    mu = float(y.mean())
    ya = y - mu
    if len(ya) < 4:
        return fit_ar1(y)[0], 0.0, mu
    r0 = (ya * ya).mean()
    r1 = (ya[:-1] * ya[1:]).mean()
    r2 = (ya[:-2] * ya[2:]).mean()
    if r0 <= 0:
        return 0.0, 0.0, mu
    rho1 = r1 / r0
    rho2 = r2 / r0
    denom = 1 - rho1 * rho1
    if abs(denom) < 1e-9:
        return fit_ar1(y)[0], 0.0, mu
    phi1 = (rho1 * (1 - rho2)) / denom
    phi2 = (rho2 - rho1 * rho1) / denom
    if not (abs(phi2) < 0.99 and phi1 + phi2 < 0.99 and phi2 - phi1 < 0.99):
        return fit_ar1(y)[0], 0.0, mu
    return max(-0.99, min(0.99, phi1)), max(-0.99, min(0.99, phi2)), mu


def fit_arima11(y):
    """ARIMA(1,0,1) by maximum likelihood (statsmodels); phi and theta are clipped to +-0.99 and the series mean
    is used as the level. Falls back to a non-negative AR(1) coefficient if the fit fails."""
    from statsmodels.tsa.arima.model import ARIMA

    if y.std() == 0:
        return 0.0, 0.0, float(y.mean())
    try:
        m = ARIMA(y, order=(1, 0, 1)).fit(method_kwargs={"warn_convergence": False})
        phi = float(m.arparams[0]) if len(m.arparams) > 0 else 0.0
        theta = float(m.maparams[0]) if len(m.maparams) > 0 else 0.0
        return max(-0.99, min(0.99, phi)), max(-0.99, min(0.99, theta)), float(y.mean())
    except Exception:
        mu = float(y.mean())
        a = y[:-1] - mu
        b = y[1:] - mu
        denom = (a * a).sum()
        phi = float((a * b).sum() / denom) if denom > 0 else 0.0
        return max(0.0, min(0.99, phi)), 0.0, mu


def fit_backbone(out, train_end, order):
    """Per-county backbone fit on the training split. `order` is 2 (default), 1, or "arima11".
    Returns a dict with phi1, phi2, theta, mu (one value per county) and, for the one-lag models, the one-step
    residuals over the whole series (needed for the moving-average term)."""
    n_loc, n_hours = out.shape
    phi1 = np.zeros(n_loc)
    phi2 = np.zeros(n_loc)
    theta = np.zeros(n_loc)
    mu = np.zeros(n_loc)
    for c in range(n_loc):
        y = out[c, :train_end]
        if order == 2:
            phi1[c], phi2[c], mu[c] = fit_ar2(y)
        elif order == 1:
            phi1[c], mu[c] = fit_ar1(y)
        elif order == "arima11":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi1[c], theta[c], mu[c] = fit_arima11(y)
        else:
            raise ValueError("ar_order must be 2, 1 or 'arima11'")
    resid = None
    if order != 2:
        resid = np.zeros((n_loc, n_hours))
        for c in range(n_loc):
            y = out[c]
            e = np.zeros(n_hours)
            for t in range(1, n_hours):
                e[t] = y[t] - (mu[c] + phi1[c] * (y[t - 1] - mu[c]) + theta[c] * e[t - 1])
            resid[c] = e
    return {"order": order, "phi1": phi1, "phi2": phi2, "theta": theta, "mu": mu, "resid": resid}


def backbone_forecast(bb, y_last, y_last2, resid_last, tau):
    """tau-step-ahead forecast of the backbone, clipped at zero."""
    phi1, phi2, mu = bb["phi1"], bb["phi2"], bb["mu"]
    if bb["order"] != 2:
        p = (phi1**tau) * (y_last - mu) + mu + (phi1 ** (tau - 1)) * bb["theta"] * resid_last
    elif tau == 1:
        p = mu + phi1 * (y_last - mu) + phi2 * (y_last2 - mu)
    elif tau == 2:
        p1 = mu + phi1 * (y_last - mu) + phi2 * (y_last2 - mu)
        p = mu + phi1 * (p1 - mu) + phi2 * (y_last - mu)
    else:
        ym2 = y_last2.copy()
        ym1 = y_last.copy()
        for _ in range(tau):
            yi = mu + phi1 * (ym1 - mu) + phi2 * (ym2 - mu)
            ym2 = ym1
            ym1 = yi
        p = ym1
    return np.maximum(p, 0.0).astype(np.float32)


# ------------------------------------------------------------------ features


def build_features(
    out,
    weather,
    ts,
    bb,
    windows,
    lookback,
    lookahead,
    weather_future=None,
    residual_feature=False,
    persist_weather=False,
):
    """One row per (window, horizon step, county).

    Columns: county index, horizon step, last outage, lookback mean/std/max of outages, hours since the last
    non-zero outage, backbone forecast, last one-step backbone residual (zero unless `residual_feature`), lookback
    mean and max of each weather variable, weather at the target hour, and six calendar terms.
    Target-hour weather is read from `weather_future` when given (test time) and from `weather` otherwise; with
    `persist_weather` it is the last observed weather of the lookback instead (weather ablation).
    Returns X, y, backbone forecast, persistence forecast, and (county, window start, step) per row."""
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
    rows, ys, back, pers, meta = [], [], [], [], []
    for s in windows:
        end_lb = s + lookback
        if end_lb + lookahead > n_hours:
            continue
        y_lb = out[:, s:end_lb]
        y_last = y_lb[:, -1]
        y_last2 = y_lb[:, -2]
        resid_last = bb["resid"][:, end_lb - 1] if bb["resid"] is not None else np.zeros(n_loc)
        resid_col = (resid_last if residual_feature else np.zeros(n_loc)).astype(np.float32)
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
            p_back = backbone_forecast(bb, y_last, y_last2, resid_last, tau)
            tau_col = np.full(n_loc, tau, dtype=np.float32)
            w_t = weather[:, end_lb - 1, :] if persist_weather else w_future[:, t, :]
            X = np.concatenate(
                [
                    loc_col[:, None],
                    tau_col[:, None],
                    y_last[:, None].astype(np.float32),
                    y_mean[:, None].astype(np.float32),
                    y_std[:, None].astype(np.float32),
                    y_max[:, None].astype(np.float32),
                    run[:, None],
                    p_back[:, None],
                    resid_col[:, None],
                    w_lb_mean.astype(np.float32),
                    w_lb_max.astype(np.float32),
                    w_t.astype(np.float32),
                    np.broadcast_to(calendar[t], (n_loc, 6)).astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32)
            rows.append(X)
            ys.append(out[:, t].astype(np.float32))
            back.append(p_back)
            pers.append(y_last.astype(np.float32))
            meta.append(np.stack([loc_col, np.full(n_loc, s, dtype=np.float32), tau_col], axis=1))
    return (
        np.concatenate(rows),
        np.concatenate(ys),
        np.concatenate(back),
        np.concatenate(pers),
        np.concatenate(meta).astype(np.int32),
    )


# ------------------------------------------------------------------ model


def shifted_geometric_mean(a, b, kappa=KAPPA):
    """max( sqrt((a + kappa)(b + kappa)) - kappa, 0 ), computed in log space."""
    return np.maximum(np.exp((np.log(a + kappa) + np.log(b + kappa)) / 2.0) - kappa, 0.0)


def run(
    exp,
    *,
    seed=42,
    ar_order=2,
    residual_feature=None,
    persist_weather=False,
    n_rounds=400,
    learning_rate=0.03,
    early_stopping=30,
    backbone=None,
    return_components=False,
    **_,
):
    """PRiCE forecasts for the test targets, (L, m*lookahead).

    Ablation switches: `ar_order` (2, 1 or "arima11") replaces the backbone; `persist_weather` replaces target-hour
    weather by the last observed weather in training and at test time. `residual_feature` adds the backbone's last
    one-step residual as a feature (default: only for "arima11", whose forecast depends on it).
    With `return_components` a dict is returned with the full model, each expert alone, and the two alternative
    aggregators (arithmetic mean; convex combination with the weight tuned on validation)."""
    np.random.seed(seed)
    hp = HPARAMS[exp.dataset]
    lb, la = exp.lookback, exp.lookahead
    train_end = exp.train_end
    val0, val1 = exp.val_range
    test0, _ = exp.test_range
    if residual_feature is None:
        residual_feature = ar_order == "arima11"

    cluster, n_clusters = fit_clusters(exp.out, exp.weather_all, train_end)
    bb = backbone if backbone is not None else fit_backbone(exp.out, train_end, ar_order)

    train_w = list(range(0, train_end - lb - la + 1, hp["train_stride"]))
    val_w = list(range(val0, val1 - lb - la + 1, 1))
    test_w = [test0 + b * la for b in range(exp.m)]
    kw = dict(residual_feature=residual_feature, persist_weather=persist_weather)
    args = (exp.out, exp.weather, exp.ts, bb)
    Xtr, ytr, btr, _, mtr = build_features(*args, train_w, lb, la, **kw)
    Xva, yva, bva, pva, mva = build_features(*args, val_w, lb, la, **kw)
    Xte, yte, bte, pte, mte = build_features(*args, test_w, lb, la, weather_future=exp.weather_with_forecast(), **kw)

    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": learning_rate,
        "num_leaves": hp["num_leaves"],
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
        "num_threads": LGB_THREADS,
    }
    ctr, cva, cte = cluster[mtr[:, 0]], cluster[mva[:, 0]], cluster[mte[:, 0]]
    rtr = ytr - btr
    rva = yva - bva
    cluster_te = bte.copy()
    cluster_va = bva.copy()
    for k in range(n_clusters):
        m_tr = ctr == k
        m_va = cva == k
        m_te = cte == k
        if m_tr.sum() < 100 or m_va.sum() < 10:
            continue
        dtr = lgb.Dataset(Xtr[m_tr], rtr[m_tr])
        dva = lgb.Dataset(Xva[m_va], rva[m_va], reference=dtr)
        g = lgb.train(
            params,
            dtr,
            num_boost_round=n_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        cluster_te[m_te] = bte[m_te] + g.predict(Xte[m_te], num_iteration=g.best_iteration)
        if return_components:
            cluster_va[m_va] = bva[m_va] + g.predict(Xva[m_va], num_iteration=g.best_iteration)
    cluster_te = np.maximum(cluster_te, 0.0).astype(np.float32)
    combined = shifted_geometric_mean(pte, cluster_te)

    def grid(v):
        """Scatter per-row values into a (county, m * lookahead) array."""
        o = np.zeros((len(exp.locs), exp.m * la))
        o[mte[:, 0], (mte[:, 1] - test0) // la * la + mte[:, 2] - 1] = v
        return o

    if not return_components:
        return grid(combined)

    # alternative aggregators
    cluster_va = np.maximum(cluster_va, 0.0).astype(np.float32)
    alphas = np.linspace(0.0, 1.0, 21)  # weight on the persistence expert, tuned on pooled validation RMSE
    val_rmse = [float(np.sqrt(((yva - (a * pva + (1.0 - a) * cluster_va)) ** 2).mean())) for a in alphas]
    alpha = float(alphas[int(np.argmin(val_rmse))])
    return {
        "price": grid(combined),
        "persistence": grid(pte),
        "cluster_expert": grid(cluster_te),
        "backbone": grid(bte),
        "arithmetic_mean": grid(np.maximum(0.5 * (pte + cluster_te), 0.0)),
        "convex": grid(np.maximum(alpha * pte + (1.0 - alpha) * cluster_te, 0.0)),
        "convex_alpha": alpha,
        "n_clusters": n_clusters,
        "cluster_assignment": cluster,
    }
