#!/usr/bin/env python3
"""
PRiCE-Minimal — 2-expert Occam-direction architecture.

Experts:
  E_persist : y_{t+τ-1}                                       (zero-parameter storm-floor anchor)
  E_cluster : per-county AR + per-cluster GBDT residual       (temporal × spatial structure)

Aggregator:
  ŷ = sqrt(E_persist · E_cluster)                              (geometric mean in log-space)

Hyperparameter `--ar-order` selects the per-county backbone:
  --ar-order 'arima11'  : ARIMA(1,0,1) MLE (statsmodels)       (current default)
  --ar-order 'ar1'      : AR(1) closed-form Yule-Walker         (simpler, no MA)
  --ar-order 'ar2'      : AR(2) closed-form Yule-Walker
"""
import argparse, json, time, sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xarray as xr
import lightgbm as lgb
# statsmodels is only needed for --ar-order arima11 and is imported lazily.


def load_nc(path):
    ds = xr.load_dataset(path)
    return (ds['out'].values.astype(np.float64),
            ds['weather'].values.astype(np.float32),
            pd.to_datetime(ds['timestamp'].values),
            [str(v) for v in ds['location'].values],
            [str(v) for v in ds['feature'].values])


def fit_arima11(y):
    from statsmodels.tsa.arima.model import ARIMA
    if y.std() == 0: return (0.0, 0.0, float(y.mean()))
    try:
        m = ARIMA(y, order=(1, 0, 1)).fit(method_kwargs={'warn_convergence': False})
        phi = float(m.arparams[0]) if len(m.arparams) > 0 else 0.0
        theta = float(m.maparams[0]) if len(m.maparams) > 0 else 0.0
        phi = max(-0.99, min(0.99, phi)); theta = max(-0.99, min(0.99, theta))
        return (phi, theta, float(y.mean()))
    except Exception:
        mu = float(y.mean()); a = y[:-1] - mu; b = y[1:] - mu
        denom = (a*a).sum()
        phi = float((a*b).sum()/denom) if denom > 0 else 0.0
        return (max(0.0, min(0.99, phi)), 0.0, mu)


def fit_ar1(y):
    """Closed-form AR(1) via Yule-Walker. No MA term."""
    if y.std() == 0: return (0.0, 0.0, float(y.mean()))
    mu = float(y.mean()); a = y[:-1] - mu; b = y[1:] - mu
    denom = (a*a).sum()
    phi = float((a*b).sum()/denom) if denom > 0 else 0.0
    return (max(-0.99, min(0.99, phi)), 0.0, mu)


def fit_ar2(y):
    """Closed-form AR(2) via Yule-Walker. Falls back to AR(1) if non-stationary."""
    if y.std() == 0: return (0.0, 0.0, float(y.mean()), 0.0)
    mu = float(y.mean()); ya = y - mu
    n = len(ya)
    if n < 4: return fit_ar1(y) + (0.0,)
    r0 = (ya*ya).mean(); r1 = (ya[:-1]*ya[1:]).mean(); r2 = (ya[:-2]*ya[2:]).mean()
    if r0 <= 0: return (0.0, 0.0, mu, 0.0)
    rho1 = r1/r0; rho2 = r2/r0
    denom = 1 - rho1*rho1
    if abs(denom) < 1e-9: return fit_ar1(y) + (0.0,)
    phi1 = (rho1*(1 - rho2)) / denom
    phi2 = (rho2 - rho1*rho1) / denom
    # Stationarity check: |phi2| < 1, phi1 + phi2 < 1, phi2 - phi1 < 1
    if not (abs(phi2) < 0.99 and phi1 + phi2 < 0.99 and phi2 - phi1 < 0.99):
        return fit_ar1(y) + (0.0,)
    return (max(-0.99, min(0.99, phi1)), max(-0.99, min(0.99, phi2)), mu, 0.0)


def fit_per_county_backbone(out, t0, t1, ar_order='arima11'):
    L, T = out.shape
    if ar_order == 'ar2':
        phi1s = np.zeros(L); phi2s = np.zeros(L); mus = np.zeros(L)
        eps_full = np.zeros((L, T), dtype=np.float64)
        for c in range(L):
            y = out[c, t0:t1]
            phi1, phi2, mu, _ = fit_ar2(y)
            phi1s[c] = phi1; phi2s[c] = phi2; mus[c] = mu
            full_y = out[c, :]; eps = np.zeros(T)
            for t in range(2, T):
                yhat = mus[c] + phi1s[c]*(full_y[t-1] - mus[c]) + phi2s[c]*(full_y[t-2] - mus[c])
                eps[t] = full_y[t] - yhat
            eps_full[c] = eps
        return ('ar2', phi1s, phi2s, mus, eps_full)
    else:
        phis = np.zeros(L); thetas = np.zeros(L); mus = np.zeros(L)
        eps_full = np.zeros((L, T), dtype=np.float64)
        n_ok = 0
        for c in range(L):
            y = out[c, t0:t1]
            if ar_order == 'arima11':
                phi, theta, mu = fit_arima11(y)
            else:  # ar1
                phi, theta, mu = fit_ar1(y)
            phis[c] = phi; thetas[c] = theta; mus[c] = mu
            if phi != 0 or mu != 0: n_ok += 1
            full_y = out[c, :]; eps = np.zeros(T)
            for t in range(1, T):
                yhat = mus[c] + phis[c]*(full_y[t-1] - mus[c]) + thetas[c]*eps[t-1]
                eps[t] = full_y[t] - yhat
            eps_full[c] = eps
        print(f'  {ar_order} fits: {n_ok}/{L} OK')
        return (ar_order, phis, thetas, mus, eps_full)


def select_top_weather_features(weather, train_t0, train_t1, out, top_k=24):
    F = weather.shape[2]; L = out.shape[0]
    scores = np.zeros(F)
    for f in range(F):
        cs = []
        for c in range(L):
            x = weather[c, train_t0:train_t1, f].astype(np.float64)
            y = out[c, train_t0:train_t1].astype(np.float64)
            if x.std() <= 0 or y.std() <= 0: continue
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r): cs.append(abs(r))
        scores[f] = float(np.mean(cs)) if cs else 0.0
    return np.argsort(-scores)[:top_k].tolist()


def per_loc(y, yh, fips):
    rs, ms = [], []
    for f in np.unique(fips):
        m = fips == f
        rs.append(np.sqrt(((y[m]-yh[m])**2).mean())); ms.append(np.abs(y[m]-yh[m]).mean())
    return float(np.mean(rs)), float(np.mean(ms))


def build_features(out, weather, ts_pd, sel_feats, backbone, cluster_assn, windows, LB, LA, weather_mode='oracle', weather_future='single', drop=()):
    L, Ttot = out.shape
    hours = ts_pd.hour.values; dows = ts_pd.dayofweek.values; months = ts_pd.month.values
    H = np.stack([np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)], axis=1)
    D = np.stack([np.sin(2*np.pi*dows/7),  np.cos(2*np.pi*dows/7)],  axis=1)
    M = np.stack([np.sin(2*np.pi*months/12), np.cos(2*np.pi*months/12)], axis=1)
    TE = np.concatenate([H, D, M], axis=1).astype(np.float32)
    weather_sel = weather[:, :, sel_feats]

    kind = backbone[0]
    rows = []; ys = []; pa = []; pp = []; meta = []
    for s in windows:
        end_lb = s + LB
        if end_lb + LA > Ttot: continue
        y_lb = out[:, s:end_lb]
        y_last = y_lb[:, -1]
        y_last2 = y_lb[:, -2] if y_lb.shape[1] >= 2 else y_lb[:, -1]
        if kind == 'ar2':
            _, phi1s, phi2s, mus, eps_full = backbone
            eps_last = np.zeros(L)
        else:
            _, phis, thetas, mus, eps_full = backbone
            eps_last = eps_full[:, end_lb - 1]
        y_mean = y_lb.mean(axis=1); y_max = y_lb.max(axis=1); y_std = y_lb.std(axis=1)
        rl = np.zeros(L, dtype=np.float32)
        for c in range(L):
            ct = 0
            for v in y_lb[c, ::-1]:
                if v == 0: ct += 1
                else: break
            rl[c] = ct
        w_lb_mean = weather_sel[:, s:end_lb, :].mean(axis=1)
        w_lb_max  = weather_sel[:, s:end_lb, :].max(axis=1)

        for tau in range(1, LA + 1):
            t_idx = end_lb + tau - 1
            if kind == 'ar2':
                # Recursive AR(2) unrolling — closed-form
                if tau == 1:
                    p_back = mus + phi1s*(y_last - mus) + phi2s*(y_last2 - mus)
                elif tau == 2:
                    p1 = mus + phi1s*(y_last - mus) + phi2s*(y_last2 - mus)
                    p_back = mus + phi1s*(p1 - mus) + phi2s*(y_last - mus)
                else:
                    # General recursion
                    yim2 = y_last2.copy(); yim1 = y_last.copy()
                    for _ in range(tau):
                        yi = mus + phi1s*(yim1 - mus) + phi2s*(yim2 - mus)
                        yim2 = yim1; yim1 = yi
                    p_back = yim1
                p_back = np.maximum(p_back, 0.0).astype(np.float32)
            else:
                ap = phis ** tau
                p_back = ap * (y_last - mus) + mus + (phis ** (tau-1)) * thetas * eps_last
                p_back = np.maximum(p_back, 0.0).astype(np.float32)
            p_persist = y_last.astype(np.float32)

            if weather_mode == 'persistence':
                w_fut = weather_sel[:, end_lb - 1, :].astype(np.float32)
            else:
                w_fut = weather_sel[:, t_idx, :].astype(np.float32)
            te = TE[t_idx]; te_b = np.broadcast_to(te, (L, 6))
            tau_col = np.full(L, tau, dtype=np.float32)
            loc_col = np.arange(L, dtype=np.float32)

            if weather_future == 'window' and weather_mode != 'persistence':
                # Aggregates over the full lookahead forecast window [end_lb, t_idx]
                w_la = weather_sel[:, end_lb:t_idx + 1, :]
                if w_la.shape[1] > 0:
                    w_fut_mean = w_la.mean(axis=1).astype(np.float32)
                    w_fut_max  = w_la.max(axis=1).astype(np.float32)
                else:
                    w_fut_mean = w_fut.copy()
                    w_fut_max = w_fut.copy()
                fut_block = [w_fut.astype(np.float32),
                             w_fut_mean,
                             w_fut_max]
            else:
                fut_block = [w_fut.astype(np.float32)]

            blocks = []
            if 'locid'    not in drop: blocks.append(loc_col[:, None])
            if 'tau'      not in drop: blocks.append(tau_col[:, None])
            if 'ylast'    not in drop: blocks.append(y_last[:, None].astype(np.float32))
            if 'ymean'    not in drop: blocks.append(y_mean[:, None].astype(np.float32))
            if 'ystd'     not in drop: blocks.append(y_std[:, None].astype(np.float32))
            if 'ymax'     not in drop: blocks.append(y_max[:, None].astype(np.float32))
            if 'runlen'   not in drop: blocks.append(rl[:, None])
            if 'pback'    not in drop: blocks.append(p_back[:, None])
            if 'eps'      not in drop: blocks.append(eps_last[:, None].astype(np.float32))
            if 'wlbmean'  not in drop: blocks.append(w_lb_mean.astype(np.float32))
            if 'wlbmax'   not in drop: blocks.append(w_lb_max.astype(np.float32))
            if 'wfut'     not in drop: blocks.extend([b.astype(np.float32) for b in fut_block])
            if 'temporal' not in drop: blocks.append(te_b.astype(np.float32))
            X_step = np.concatenate(blocks, axis=1).astype(np.float32)

            y_step = out[:, t_idx].astype(np.float32)
            rows.append(X_step); ys.append(y_step); pa.append(p_back); pp.append(p_persist)
            meta.append(np.stack([loc_col, np.full(L, s, dtype=np.float32), tau_col,
                                  cluster_assn.astype(np.float32)], axis=1))
    X = np.concatenate(rows, axis=0)
    y = np.concatenate(ys, axis=0)
    p_back_arr = np.concatenate(pa, axis=0)
    p_persist_arr = np.concatenate(pp, axis=0)
    m = np.concatenate(meta, axis=0).astype(np.int32)
    return X, y, p_back_arr, p_persist_arr, m


def lgb_params_mae(args):
    """LightGBM hyperparameters for the cluster GBDT, with absolute-error loss
    (regression_l1) following Friedman 2001."""
    return {'objective': 'regression_l1', 'metric': 'mae',
            'learning_rate': args.lr, 'num_leaves': args.num_leaves,
            'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 5,
            'verbose': -1, 'seed': args.seed}


def train_reg(X, y, val_X, val_y, params, n_rounds, early_stop=30):
    dtr = lgb.Dataset(X, y); dva = lgb.Dataset(val_X, val_y, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(early_stop, verbose=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nc', type=str, required=True)
    ap.add_argument('--clusters', type=str, required=True)
    ap.add_argument('--lookback', type=int, default=48)
    ap.add_argument('--lookahead', type=int, default=24)
    ap.add_argument('--m-val', type=int, default=5)
    ap.add_argument('--m-test', type=int, default=5)
    ap.add_argument('--top-weather', type=int, default=24)
    ap.add_argument('--train-stride', type=int, default=8)
    ap.add_argument('--n-rounds', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.03)
    ap.add_argument('--num-leaves', type=int, default=31)
    ap.add_argument('--ar-order', type=str, default='arima11', choices=['arima11', 'ar1', 'ar2'])
    ap.add_argument('--weather-mode', type=str, default='oracle', choices=['oracle', 'persistence'])
    ap.add_argument('--weather-future', type=str, default='single', choices=['single', 'window'],
                    help='Whether to use only the forecast weather at t+tau (single) or also include mean and max of the lookahead forecast over [t+1, t+tau] (window).')
    ap.add_argument('--drop-features', type=str, default='',
                    help='Comma-separated feature groups to drop from the cluster GBDT input. Groups: locid, tau, ylast, ymean, ystd, ymax, runlen, pback, eps, wlbmean, wlbmax, wfut, temporal.')
    ap.add_argument('--out-csv', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    np.random.seed(args.seed)
    LB, LA = args.lookback, args.lookahead

    print(f'Loading {args.nc}... (ar_order={args.ar_order})')
    out, weather, ts_pd, locs, feat_names = load_nc(args.nc)
    L, Ttot = out.shape
    cluster_data = json.load(open(args.clusters))
    assert cluster_data['locs'] == locs
    cluster_assn = np.array(cluster_data['assn'], dtype=np.int32)
    K_clusters = cluster_data['K']

    val_len = LB + args.m_val * LA
    test_len = LB + args.m_test * LA
    train_len = Ttot - val_len - test_len
    train_idx = (0, train_len)
    val_idx = (train_len, train_len + val_len)
    test_idx = (train_len + val_len, Ttot)
    print(f'  L={L} T={Ttot} K_clusters={K_clusters}  num_leaves={args.num_leaves}')
    print(f'  splits: train{train_idx} val{val_idx} test{test_idx}')

    print(f'Backbone fit (train-only, {args.ar_order})...')
    t0 = time.time()
    bb = fit_per_county_backbone(out, *train_idx, ar_order=args.ar_order)
    print(f'  {time.time()-t0:.1f}s')

    sel = select_top_weather_features(weather, *train_idx, out, args.top_weather)

    train_starts = list(range(train_idx[0], train_idx[1]-LB-LA+1, args.train_stride))
    val_starts = list(range(val_idx[0], val_idx[1]-LB-LA+1, 1))
    test_starts = [test_idx[0] + b*LA for b in range(args.m_test)]
    print(f'Windows: train={len(train_starts)} val={len(val_starts)} test={len(test_starts)}')

    print('Building features (train-only backbone)...')
    t0 = time.time()
    drop_feats = tuple(s.strip() for s in args.drop_features.split(",") if s.strip())
    Xtr, ytr, ptr_back, ptr_pers, mtr = build_features(out, weather, ts_pd, sel, bb, cluster_assn, train_starts, LB, LA, weather_mode=args.weather_mode, weather_future=args.weather_future, drop=drop_feats)
    Xva, yva, pva_back, pva_pers, mva = build_features(out, weather, ts_pd, sel, bb, cluster_assn, val_starts, LB, LA, weather_mode=args.weather_mode, weather_future=args.weather_future, drop=drop_feats)
    Xte, yte, pte_back, pte_pers, mte = build_features(out, weather, ts_pd, sel, bb, cluster_assn, test_starts, LB, LA, weather_mode=args.weather_mode, weather_future=args.weather_future, drop=drop_feats)
    print(f'  features: {time.time()-t0:.1f}s')

    # E_cluster: backbone + per-cluster GBDT residual
    print(f'Training {K_clusters} cluster GBDT residual regressors (train-only)...')
    reg_params = lgb_params_mae(args)
    cluster_idx_tr = mtr[:, 3]; cluster_idx_va = mva[:, 3]; cluster_idx_te = mte[:, 3]
    pte_cluster = pte_back.copy()
    rtr = ytr - ptr_back; rva = yva - pva_back
    cluster_models_tr = {}
    for kk in range(K_clusters):
        m_tr = cluster_idx_tr == kk; m_va = cluster_idx_va == kk
        if m_tr.sum() < 100 or m_va.sum() < 10:
            cluster_models_tr[kk] = None; continue
        t0 = time.time()
        g = train_reg(Xtr[m_tr], rtr[m_tr], Xva[m_va], rva[m_va], reg_params, args.n_rounds)
        cluster_models_tr[kk] = g
        print(f'  cluster {kk} (n_train={m_tr.sum()}): {time.time()-t0:.1f}s, best_iter={g.best_iteration}')

    for kk in range(K_clusters):
        m_te = cluster_idx_te == kk
        g = cluster_models_tr[kk]
        if g is None: continue
        pte_cluster[m_te] = pte_back[m_te] + g.predict(Xte[m_te], num_iteration=g.best_iteration)
    pte_cluster = np.maximum(pte_cluster, 0.0).astype(np.float32)

    # Aggregate (default): geomean of (E_persist, E_cluster)
    eps = 0.5
    yhat_unified = np.maximum(np.exp((np.log(pte_pers + eps) + np.log(pte_cluster + eps)) / 2.0) - eps, 0.0).astype(np.float32)

    # Aggregator ablation: arithmetic mean and validation-tuned convex combination.
    pva_cluster = pva_back.copy()
    for kk in range(K_clusters):
        g = cluster_models_tr[kk]
        if g is None: continue
        m_va_kk = cluster_idx_va == kk
        if not m_va_kk.any(): continue
        pva_cluster[m_va_kk] = pva_back[m_va_kk] + g.predict(Xva[m_va_kk], num_iteration=g.best_iteration)
    pva_cluster = np.maximum(pva_cluster, 0.0).astype(np.float32)

    yhat_arith = np.maximum(0.5 * (pte_pers + pte_cluster), 0.0).astype(np.float32)

    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha, best_rmse = 0.5, np.inf
    for a in alphas:
        pred_va = a * pva_pers + (1.0 - a) * pva_cluster
        r = float(np.sqrt(((yva - pred_va)**2).mean()))
        if r < best_rmse:
            best_rmse, best_alpha = r, float(a)
    yhat_convex = np.maximum(best_alpha * pte_pers + (1.0 - best_alpha) * pte_cluster, 0.0).astype(np.float32)
    print(f'Aggregator ablation: val-tuned convex selected alpha={best_alpha:.2f}')

    loc_t = mte[:, 0]
    print()
    print(f'Test results — PRiCE-Minimal (ar_order={args.ar_order}, K={K_clusters}, num_leaves={args.num_leaves}):')
    for name, yh in [
        ('All-Zeros', np.zeros_like(yte)),
        ('E_persist', pte_pers),
        ('E_backbone alone', pte_back),
        ('E_cluster (backbone + cluster GBDT)', pte_cluster),
        ('Aggregator ablation: arithmetic mean', yhat_arith),
        (f'Aggregator ablation: val-tuned convex (a={best_alpha:.2f})', yhat_convex),
        ('PRiCE-Minimal: geomean(persist, cluster)', yhat_unified),
    ]:
        r, m = per_loc(yte, yh, loc_t)
        print(f'  {name:55s}  RMSE={r:9.3f}  MAE={m:8.3f}')

    if args.out_csv:
        pd.DataFrame({
            'loc_idx': loc_t, 'win_start': mte[:, 1], 'tau': mte[:, 2], 'cluster': cluster_idx_te,
            'y_true': yte, 'y_persist': pte_pers, 'y_backbone': pte_back, 'y_cluster': pte_cluster,
            'y_unified': yhat_unified,
        }).to_csv(args.out_csv, index=False)
        print(f'Saved {args.out_csv}')


if __name__ == '__main__':
    main()
