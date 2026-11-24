"""Zero-shot TimesFM 3.0 on outages with weather covariates (the model is not trained or fine-tuned).

Each window is forecast from the last `lookback` hours of outages, with the weather variables as covariates:
actual weather over the lookback and forecast weather over the horizon. The point forecast is the median."""

from __future__ import annotations

import numpy as np

NAME = "TimesFM 3.0 (zero-shot)"
NEEDS_GPU = True
CHECKPOINT = "google/timesfm-3.0-pytorch"


def run(exp, *, device=None, batch_size=32, lookback=None, **_):
    """Zero-shot TimesFM 3.0 forecasts for the test targets, (counties, m * lookahead), clipped at zero.

    `lookback` defaults to the experiment's lookback.
    """
    import torch
    from timesfm3 import TimesFM3Forecaster

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lb = exp.lookback if lookback is None else lookback
    la = exp.lookahead
    forecaster = TimesFM3Forecaster.from_pretrained(CHECKPOINT, device=device, per_core_batch_size=batch_size)
    contexts, covariates = [], []
    for c in range(len(exp.locs)):
        for i, a in enumerate(exp.window_starts()):
            cov = exp.weather[c, a - lb : a + la].copy()  # actual weather ...
            cov[lb:] = exp.future[c, i * la : (i + 1) * la]  # ... forecast over the horizon
            contexts.append(exp.out[c, a - lb : a].astype(np.float32))
            covariates.append(cov.T.astype(np.float32))
    outputs = forecaster.predict_batch(
        contexts=contexts, horizon=la, past_future_covariates=covariates, make_positive=True
    )
    pred = np.stack([o.forecast for o in outputs]).reshape(len(exp.locs), exp.m * la)
    return np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)
