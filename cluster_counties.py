#!/usr/bin/env python3
"""Cluster counties using training-set features only.

Features per county (all standardized):
  - log1p(mean outage)
  - lag-1 AC
  - zero-fraction
  - log1p(p99 outage)
  - mean weather correlation magnitude (top-3 features)

Output: cluster assignment per county for both Michigan and California.
Saves to /tmp/clusters_<state>_<LA>.json.
"""
import json, sys, argparse
import numpy as np
import pandas as pd
import xarray as xr


def cluster_features(out, weather, train_t0, train_t1):
    L = out.shape[0]
    feats = np.zeros((L, 7), dtype=np.float64)
    for c in range(L):
        y = out[c, train_t0:train_t1].astype(np.float64)
        feats[c, 0] = np.log1p(y.mean())
        feats[c, 1] = np.log1p(np.percentile(y, 99))
        feats[c, 2] = (y == 0).mean()
        # lag-1 AC
        if y.std() > 0:
            a = y[:-1] - y.mean(); b = y[1:] - y.mean()
            denom = (a*a).sum()
            feats[c, 3] = float((a*b).sum() / denom) if denom > 0 else 0.0
        # weather correlations: pick top-5 abs corrs across features, take mean
        corrs = []
        F = weather.shape[2]
        for f in range(F):
            x = weather[c, train_t0:train_t1, f].astype(np.float64)
            if x.std() <= 0 or y.std() <= 0: continue
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r): corrs.append(abs(r))
        corrs = sorted(corrs, reverse=True)
        feats[c, 4] = np.mean(corrs[:5]) if len(corrs) >= 1 else 0.0
        feats[c, 5] = np.log1p(y.std())
        feats[c, 6] = float((y > np.percentile(y, 99)).mean())
    return feats


def kmeans(X, K, n_iter=100, seed=42):
    rng = np.random.RandomState(seed)
    n = len(X)
    # k-means++ init
    centers_idx = [int(rng.randint(n))]
    for _ in range(K-1):
        dists = np.array([min(np.linalg.norm(X[i] - X[j])**2 for j in centers_idx) for i in range(n)])
        if dists.sum() == 0: break
        probs = dists / dists.sum()
        centers_idx.append(int(rng.choice(n, p=probs)))
    centers = X[centers_idx].copy()
    for _ in range(n_iter):
        # assign
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # [n, K]
        assn = d.argmin(axis=1)
        # update
        new_centers = np.zeros_like(centers)
        for k in range(K):
            mask = assn == k
            if mask.sum() > 0:
                new_centers[k] = X[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return assn, centers


def silhouette(X, assn):
    """Simple silhouette-ish score: mean intra-cluster dist / mean nearest-other-cluster dist."""
    n = len(X)
    K = len(np.unique(assn))
    if K < 2 or n < 2: return 0.0
    s = []
    for i in range(n):
        same = X[assn == assn[i]]
        a = np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n) if assn[j] == assn[i] and j != i]) if (assn == assn[i]).sum() > 1 else 0
        bs = []
        for k in range(K):
            if k == assn[i]: continue
            mask = assn == k
            if mask.sum() > 0:
                bs.append(np.mean([np.linalg.norm(X[i] - X[j]) for j in range(n) if assn[j] == k]))
        b = min(bs) if bs else a
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def fit_clusters(nc_path, LA, MV, MT, K_grid=None):
    if K_grid is None: K_grid = [3, 4, 5, 6, 8]
    ds = xr.load_dataset(nc_path)
    out = ds['out'].values.astype(np.float64)
    weather = ds['weather'].values.astype(np.float32)
    locs = [str(v) for v in ds['location'].values]
    L, T = out.shape
    LB = 48
    val_len = LB + MV * LA
    test_len = LB + MT * LA
    train_t1 = T - val_len - test_len
    feats = cluster_features(out, weather, 0, train_t1)
    # standardize
    mu = feats.mean(axis=0); sd = feats.std(axis=0) + 1e-9
    Z = (feats - mu) / sd
    print(f'  Feature mu: {mu.round(3)}')
    print(f'  Feature sd: {sd.round(3)}')
    best = None
    for K in K_grid:
        if K >= L: continue
        assn, centers = kmeans(Z, K, n_iter=100)
        sil = silhouette(Z, assn)
        n_per_cluster = np.bincount(assn, minlength=K)
        print(f'  K={K}: silhouette={sil:.3f}  cluster_sizes={n_per_cluster.tolist()}')
        if best is None or sil > best[0]: best = (sil, K, assn, centers)
    sil, K, assn, centers = best
    print(f'  PICKED K={K} (silhouette={sil:.3f})')
    return locs, assn.tolist(), Z, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nc', type=str, required=True)
    ap.add_argument('--lookahead', type=int, default=24)
    ap.add_argument('--m-val', type=int, default=5)
    ap.add_argument('--m-test', type=int, default=5)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()
    print(f'Clustering {args.nc} (lookahead={args.lookahead})...')
    locs, assn, Z, feats = fit_clusters(args.nc, args.lookahead, args.m_val, args.m_test)
    K = max(assn) + 1
    print(f'  Per-cluster summary (raw features):')
    for k in range(K):
        idx = [i for i, a in enumerate(assn) if a == k]
        print(f'    cluster {k} (n={len(idx)}): mean_outage={np.expm1(np.mean([feats[i, 0] for i in idx])):.1f}, '
              f'lag1AC={np.mean([feats[i, 3] for i in idx]):.3f}, '
              f'zero_frac={np.mean([feats[i, 2] for i in idx]):.3f}, '
              f'top5corr={np.mean([feats[i, 4] for i in idx]):.3f}')
    json.dump({'locs': locs, 'assn': assn, 'K': K}, open(args.out, 'w'))
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
