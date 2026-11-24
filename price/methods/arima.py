"""ARIMA(1,0,1) and ARIMAX(1,0,1) per county.

Each county's model is fit once on all hours before the test targets. It then forecasts one window at a time;
after each window the filter is updated with the observed outages (and, for ARIMAX, the observed weather) without
refitting. ARIMAX uses the weather variables as exogenous regressors: actual weather in fitting and updating,
forecast weather over each window being predicted."""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

NAME = "ARIMA(1,0,1)"
NEEDS_GPU = False
ORDER = (1, 0, 1)


def _init_worker():
    """Pin the numerical libraries to one thread in each worker process."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def _county(job):
    """Fit one county's model and forecast its test targets window by window.

    `job` is (outages before the test targets, outages over the test targets, exogenous weather before the test
    targets, actual weather over the test targets, forecast weather over the test targets, lookahead, order). The
    three weather entries are None for ARIMA.
    """
    y_train, y_test, x_train, x_test, x_future, lookahead, order = job
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        state = SARIMAX(
            y_train, exog=x_train, order=order, enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False)
        chunks = []
        for a in range(0, len(y_test), lookahead):
            b = min(a + lookahead, len(y_test))
            kw = {} if x_future is None else {"exog": x_future[a:b]}
            chunks.append(np.asarray(state.forecast(steps=b - a, **kw), dtype=float))
            kw = {} if x_test is None else {"exog": x_test[a:b]}
            state = state.append(endog=y_test[a:b], refit=False, **kw)
    pred = np.concatenate(chunks)
    return np.clip(np.nan_to_num(pred, nan=0.0, posinf=None, neginf=0.0), 0.0, None)


def run(exp, *, exog=False, n_jobs=1, order=ORDER, **_):
    """ARIMA (`exog=False`) or ARIMAX (`exog=True`) forecasts for the test targets, (counties, m * lookahead).

    Counties are fit in `n_jobs` processes.
    """
    t0 = exp.target_start
    jobs = []
    for c in range(len(exp.locs)):
        if exog:
            xw = exp.weather64[c]
            x = (xw[:t0], xw[t0:], exp.future[c].astype(np.float64))
        else:
            x = (None, None, None)
        jobs.append((exp.out[c, :t0], exp.out[c, t0:], x[0], x[1], x[2], exp.lookahead, order))
    if n_jobs <= 1:
        preds = [_county(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as pool:
            preds = list(pool.map(_county, jobs))
    return np.stack(preds)
