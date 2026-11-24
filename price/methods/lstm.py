"""LSTM baseline, with or without sparsity clustering.

Input sequence: `lookback` past steps [outages, weather, 0] followed by `lookahead` future steps [0, weather, 1];
the last hidden state is mapped to the `lookahead` outage values. Weather over future steps is actual in training
and validation and forecast at test time. Inputs and targets are standardized on the training split.
With sparsity clustering, counties are split into three groups by their share of zero-outage hours in training
(< 0.4, 0.4-0.8, >= 0.8) and one model is trained per group."""

from __future__ import annotations

import copy

import numpy as np

NAME = "LSTM"
NEEDS_GPU = True
HPARAMS = {"michigan": dict(hidden=128), "california": dict(hidden=256)}
SPARSITY_EDGES = (0.4, 0.8)


def _windows(w, y, lookback, lookahead):
    """w (T, L, F), y (T, L) standardized -> X (N*L, lookback+lookahead, F+2), Y (N*L, lookahead); county-major."""
    n_hours, n_loc, n_feat = w.shape
    n = n_hours - lookback - lookahead + 1
    X = np.zeros((n_loc, n, lookback + lookahead, n_feat + 2), dtype=np.float32)
    Y = np.zeros((n_loc, n, lookahead), dtype=np.float32)
    for t in range(n):
        X[:, t, :lookback, 0] = y[t : t + lookback].T
        X[:, t, :, 1:-1] = w[t : t + lookback + lookahead].transpose(1, 0, 2)
        X[:, t, lookback:, -1] = 1.0
        Y[:, t] = y[t + lookback : t + lookback + lookahead].T
    return X.reshape(n_loc * n, lookback + lookahead, n_feat + 2), Y.reshape(n_loc * n, lookahead)


def _train_predict(exp, counties, *, hidden, layers, batch_size, lr, epochs, patience, device, seed):
    """Train one LSTM on the given counties (MSE loss, Adam, early stopping on the validation loss) and return its
    forecasts for the test targets, (len(counties), m * lookahead), clipped at zero.
    """
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    np.random.seed(seed)
    lb, la = exp.lookback, exp.lookahead
    tr = slice(0, exp.train_end)
    va = slice(*exp.val_range)
    out = exp.out[counties].T  # (T, C)
    w = exp.weather64[counties].transpose(1, 0, 2)  # (T, C, F)
    n_feat = w.shape[2]
    w_scaler = StandardScaler().fit(w[tr].reshape(-1, n_feat))
    y_scaler = StandardScaler().fit(out[tr].reshape(-1, 1))
    sw = lambda a: w_scaler.transform(a.reshape(-1, n_feat)).reshape(a.shape)
    sy = lambda a: y_scaler.transform(a.reshape(-1, 1)).reshape(a.shape)
    Xtr, Ytr = _windows(sw(w[tr]), sy(out[tr]), lb, la)
    Xva, Yva = _windows(sw(w[va]), sy(out[va]), lb, la)

    class Net(nn.Module):
        """LSTM whose final hidden state of the last layer is mapped linearly to the `lookahead` outputs."""

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_feat + 2, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, la)

        def forward(self, x):
            """(batch, lookback + lookahead, n_feat + 2) -> (batch, lookahead)."""
            _, (h, _) = self.lstm(x)
            return self.head(h[-1])

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xva_t, Yva_t = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for k in range(0, len(Xtr_t), batch_size):  # fixed order, as in the reference script
            xb, yb = Xtr_t[k : k + batch_size].to(device), Ytr_t[k : k + batch_size].to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for k in range(0, len(Xva_t), 4096):
                xb, yb = Xva_t[k : k + 4096].to(device), Yva_t[k : k + 4096].to(device)
                tot += loss_fn(model(xb), yb).item() * len(xb)
        val = tot / len(Xva_t)
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    # test: lookback from actual outages and weather, horizon weather from the forecast
    wa = sw(w)
    ya = sy(out)
    wf = sw(exp.future[counties].transpose(1, 0, 2).astype(np.float64))  # (m*la, C, F)
    preds = np.zeros((len(counties), exp.m * la))
    with torch.no_grad():
        for i, a in enumerate(exp.window_starts()):
            X = np.zeros((len(counties), lb + la, n_feat + 2), dtype=np.float32)
            X[:, :lb, 0] = ya[a - lb : a].T
            X[:, :lb, 1:-1] = wa[a - lb : a].transpose(1, 0, 2)
            X[:, lb:, 1:-1] = wf[i * la : (i + 1) * la].transpose(1, 0, 2)
            X[:, lb:, -1] = 1.0
            p = model(torch.from_numpy(X).to(device)).cpu().numpy()
            preds[:, i * la : (i + 1) * la] = y_scaler.inverse_transform(p.reshape(-1, 1)).reshape(p.shape)
    return np.clip(np.nan_to_num(preds, nan=0.0, posinf=None, neginf=0.0), 0.0, None)


def sparsity_groups(exp):
    """Split the counties into up to three groups by their share of zero-outage hours in training (edges in
    SPARSITY_EDGES). Empty groups are dropped.
    """
    zero_share = (exp.out[:, : exp.train_end] == 0).mean(axis=1)
    lo, hi = SPARSITY_EDGES
    groups = [
        np.where(zero_share < lo)[0],
        np.where((zero_share >= lo) & (zero_share < hi))[0],
        np.where(zero_share >= hi)[0],
    ]
    return [g for g in groups if len(g)]


def run(
    exp,
    *,
    sparsity_clusters=False,
    device=None,
    seed=42,
    layers=1,
    batch_size=64,
    lr=1e-4,
    epochs=200,
    patience=20,
    **_,
):
    """LSTM forecasts for the test targets, (counties, m * lookahead).

    With `sparsity_clusters` one model is trained per sparsity group of counties.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(
        hidden=HPARAMS[exp.dataset]["hidden"],
        layers=layers,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        patience=patience,
        device=device,
        seed=seed,
    )
    if not sparsity_clusters:
        return _train_predict(exp, np.arange(len(exp.locs)), **kw)
    pred = np.zeros((len(exp.locs), exp.m * exp.lookahead))
    for g in sparsity_groups(exp):
        pred[g] = _train_predict(exp, g, **kw)
    return pred
