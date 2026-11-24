"""Shared data layer: every method sees the same splits, the same weather variables and the same test-time weather.

Inputs
------
* An outage NetCDF per state with variables ``out`` [location, timestamp] and ``weather`` [location, timestamp, feature].
* A weather-forecast CSV per experiment with columns ``timestamp, county, <variables>``, one row per county and
  test target hour (rows outside the test target hours are ignored).

Conventions (as in the paper)
-----------------------------
* Lookback L = 48 h; lookahead T in {24, 48} h; m = 5 (Michigan) or 30 (California) windows.
* Test split = last L + m*T hours; its last m*T hours are the test targets, in m consecutive windows of T hours.
  Validation split = the L + m*T hours before the test split. Training = everything before that.
* Weather: models are trained and validated on actual weather. At test time a window's lookback uses actual
  weather and its T target hours use the forecast file.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

LOOKBACK = 48
N_WINDOWS = {"michigan": 5, "california": 30}
TOP_WEATHER = 24

# Physical bounds used to clip forecast weather. A lower bound of 0 is applied only when the actual data of that
# variable is itself non-negative (some sources store fluxes with a negative sign).
NONNEGATIVE = {
    "blh",
    "cape",
    "gust",
    "max_10si",
    "pwat",
    "tcolw",
    "tp",
    "sp",
    "pres",
    "sdswrf",
    "suswrf",
    "sdlwrf",
    "sulwrf",
}
FRACTIONS = {"tcc", "lcc", "mcc", "hcc", "lsm"}


@dataclasses.dataclass
class Experiment:
    """One experiment (state x lookahead): outages, weather, forecast weather over the test target hours, and the
    split indices. Arrays are indexed [county, hour] or [county, hour, variable].
    """

    dataset: str
    lookahead: int
    locs: list[str]
    ts: pd.DatetimeIndex
    out: np.ndarray  # (L, T) outage counts, float64
    weather_all: np.ndarray  # (L, T, F_all) every variable in the file, float32
    all_features: list[str]
    features: list[str]  # the variables every method uses, in selection order
    weather: np.ndarray  # (L, T, F) actual weather of `features`, float32
    weather64: np.ndarray  # the same in float64 (as stored), for the methods that standardize or regress on it
    future: np.ndarray  # (L, m*lookahead, F) weather over the test target hours (forecast)
    lookback: int = LOOKBACK
    notes: dict = dataclasses.field(default_factory=dict)

    @property
    def m(self) -> int:
        """Number of test windows."""
        return N_WINDOWS[self.dataset]

    @property
    def T(self) -> int:
        """Number of hourly steps in the file."""
        return self.out.shape[1]

    @property
    def split_len(self) -> int:
        """Length of the validation split and of the test split: lookback + m * lookahead."""
        return self.lookback + self.m * self.lookahead

    @property
    def train_end(self) -> int:
        """End (exclusive) of the training split; training is hours [0, train_end)."""
        return self.T - 2 * self.split_len

    @property
    def val_range(self) -> tuple[int, int]:
        """Half-open hour range of the validation split."""
        return self.train_end, self.train_end + self.split_len

    @property
    def test_range(self) -> tuple[int, int]:
        """Half-open hour range of the test split."""
        return self.T - self.split_len, self.T

    @property
    def target_start(self) -> int:
        """Index of the first test target hour."""
        return self.T - self.m * self.lookahead

    @property
    def target_ts(self) -> pd.DatetimeIndex:
        """Timestamps of the test target hours."""
        return self.ts[self.target_start :]

    @property
    def truth(self) -> np.ndarray:  # (L, m*lookahead)
        """Observed outages over the test target hours, (counties, m * lookahead)."""
        return self.out[:, self.target_start :]

    def weather_with_forecast(self) -> np.ndarray:
        """Actual weather with the test target hours replaced by the forecast. Slicing a window's target hours
        from this array gives forecast weather; lookbacks must be sliced from `weather` (actual)."""
        w = self.weather.copy()
        w[:, self.target_start :, :] = self.future
        return w

    def window_starts(self) -> list[int]:
        """Index of the first target hour of each test window."""
        return [self.target_start + i * self.lookahead for i in range(self.m)]


def load_netcdf(path):
    """Read an outage NetCDF file.

    Returns outages (county, hour), weather (county, hour, variable), timestamps, county identifiers and weather
    variable names.
    """
    ds = xr.load_dataset(path)
    out = ds["out"].values.astype(np.float64)
    weather = ds["weather"].values.astype(np.float64)
    ts = pd.to_datetime(ds["timestamp"].values)
    locs = [str(v) for v in ds["location"].values]
    feats = [str(v) for v in ds["feature"].values]
    ds.close()
    return out, weather, ts, locs, feats


def fill_missing_hours(weather: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Hours where every weather variable is exactly 0 in every county are missing records stored as zeros.
    Replace them by the average of the neighbouring hours. (This uses the following hour; the two such hours in
    Michigan lie in the training period, so no test-time information is involved.)"""
    missing = np.where((weather == 0).all(axis=(0, 2)))[0].tolist()
    w = weather.copy()
    for t in missing:
        if 0 < t < w.shape[1] - 1:
            w[:, t, :] = (weather[:, t - 1, :] + weather[:, t + 1, :]) / 2
    return w, missing


def select_weather(weather: np.ndarray, out: np.ndarray, train_end: int, top_k: int = TOP_WEATHER) -> list[int]:
    """Rank variables by the mean over counties of |Pearson correlation| with outages on the training split and
    keep the top `top_k`. With a file of exactly `top_k` variables this keeps them all (in ranked order)."""
    n_loc, _, n_feat = weather.shape
    scores = np.zeros(n_feat)
    for f in range(n_feat):
        cs = []
        for c in range(n_loc):
            x = weather[c, :train_end, f].astype(np.float64)
            y = out[c, :train_end].astype(np.float64)
            if x.std() <= 0 or y.std() <= 0:
                continue
            r = np.corrcoef(x, y)[0, 1]
            if not np.isnan(r):
                cs.append(abs(r))
        scores[f] = float(np.mean(cs)) if cs else 0.0
    return np.argsort(-scores)[:top_k].tolist()


def prepare_future_weather(csv_path, weather, ts, locs, features, starts, lookahead, clip=True):
    """Weather over the test target hours, (L, m*lookahead, F), from the forecast file.

    * A variable absent from the file is held at its last observed value (the hour before the window starts).
    * With `clip`, forecasts are clamped to physical bounds.
    Returns the array and a dict describing what was filled and clipped."""
    fc = pd.read_csv(csv_path, dtype={"county": str}, parse_dates=["timestamp"])
    n_loc, _, n_feat = weather.shape
    t0 = starts[0]
    span = len(starts) * lookahead
    target_ts = ts[t0 : t0 + span]
    fc = fc[fc["timestamp"].isin(target_ts)]
    width = max(len(l) for l in locs)
    fc = fc.assign(county=fc["county"].str.zfill(width))
    have = [f for f in features if f in fc.columns]
    held = [f for f in features if f not in fc.columns]
    future = np.zeros((n_loc, span, n_feat), dtype=np.float32)
    clipped: dict[str, int] = {}
    for j, f in enumerate(features):
        if f in have:
            piv = fc.pivot(index="county", columns="timestamp", values=f)
            missing_rows = [l for l in locs if l not in piv.index]
            if missing_rows or len(piv.columns) != span:
                raise ValueError(
                    f"{csv_path}: forecast of '{f}' does not cover every county and test target hour "
                    f"(missing counties: {missing_rows[:3]}, hours found: {len(piv.columns)} of {span})"
                )
            vals = piv.loc[locs, target_ts].values.astype(np.float32)
            if np.isnan(vals).any():
                raise ValueError(f"{csv_path}: forecast of '{f}' has missing values")
            if clip:
                act = weather[:, :, j]
                nonneg_in_data = np.nanmin(act) >= -1e-3 * max(float(np.nanmax(np.abs(act))), 1e-12)
                lo = 0.0 if ((f in NONNEGATIVE or f in FRACTIONS) and nonneg_in_data) else -np.inf
                hi = (100.0 if np.nanmax(act) > 1.5 else 1.0) if f in FRACTIONS else np.inf
                n = int(((vals < lo) | (vals > hi)).sum())
                if n:
                    clipped[f] = n
                vals = np.clip(vals, lo, hi)
            future[:, :, j] = vals
        else:
            for i, a in enumerate(starts):
                future[:, i * lookahead : (i + 1) * lookahead, j] = weather[:, a - 1, j][:, None]
    return future, {"held_at_last_value": held, "clipped": clipped, "forecast_file": str(csv_path)}


def build_experiment(
    nc_path,
    dataset: str,
    lookahead: int,
    forecast_csv,
    *,
    fix_missing_hours: bool = False,
    clip_forecast: bool = True,
    features: list[str] | None = None,
) -> Experiment:
    """Load one experiment (state x lookahead). `features` overrides the automatic variable selection.
    The weather is used as stored. `fix_missing_hours=True` replaces hours whose weather is zero in every variable and
    county (two in Michigan, both in the training period) by the average of their neighbours; it is off by default
    so that results match the paper."""
    if dataset not in N_WINDOWS:
        raise ValueError(f"unknown dataset '{dataset}', expected one of {sorted(N_WINDOWS)}")
    for p, what in ((nc_path, "outage NetCDF"), (forecast_csv, "weather forecast CSV")):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{what} not found: {p}")
    out, weather_raw, ts, locs, all_features = load_netcdf(nc_path)
    notes: dict = {}
    m = N_WINDOWS[dataset]
    n_hours = out.shape[1]
    train_end = n_hours - 2 * (LOOKBACK + m * lookahead)
    if features is None:
        # The ranking is always computed on the file as stored, also when missing hours are filled afterwards:
        # filling them makes some near-static variables constant in most counties, which distorts the ranking.
        sel = select_weather(weather_raw.astype(np.float32), out, train_end)
    else:
        sel = [all_features.index(f) for f in features]
    if fix_missing_hours:
        weather_raw, missing = fill_missing_hours(weather_raw)
        notes["missing_hours_filled"] = [str(ts[t]) for t in missing]
    weather_all = weather_raw.astype(np.float32)
    names = [all_features[i] for i in sel]
    weather = np.ascontiguousarray(weather_all[:, :, sel])
    weather64 = np.ascontiguousarray(weather_raw[:, :, sel])
    t0 = n_hours - m * lookahead
    starts = [t0 + i * lookahead for i in range(m)]
    future, info = prepare_future_weather(forecast_csv, weather, ts, locs, names, starts, lookahead, clip=clip_forecast)
    notes.update(info)
    return Experiment(
        dataset=dataset,
        lookahead=lookahead,
        locs=locs,
        ts=ts,
        out=out,
        weather_all=weather_all,
        all_features=all_features,
        features=names,
        weather=weather,
        weather64=weather64,
        future=future,
        notes=notes,
    )


def find_forecast(forecast_dir, dataset: str, lookahead: int) -> Path:
    """Locate the forecast CSV of an experiment. Accepted layouts inside `forecast_dir`:
    ``<dataset>_<lookahead>.csv`` or ``<dataset>[_hourly]_<context>_<lookahead>_<tag>/weather_forecast.csv``
    (e.g. ``california_672_24_zero_shot/``), where <tag> does not start with a digit."""
    d = Path(forecast_dir)
    flat = d / f"{dataset}_{lookahead}.csv"
    if flat.is_file():
        return flat
    pattern = re.compile(rf"^{re.escape(dataset)}(?:_hourly)?_\d+_{lookahead}_(?!\d)")
    hits = (
        sorted(
            p / "weather_forecast.csv"
            for p in d.iterdir()
            if p.is_dir() and pattern.match(p.name) and (p / "weather_forecast.csv").is_file()
        )
        if d.is_dir()
        else []
    )
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"no weather forecast for {dataset} {lookahead}h in {d}: expected {flat.name} or "
            f"{dataset}_<context>_{lookahead}_<tag>/weather_forecast.csv"
        )
    raise FileNotFoundError(f"several weather forecasts match {dataset} {lookahead}h in {d}: {[str(h) for h in hits]}")


def lookback_mean(weather: np.ndarray, start: int, end: int) -> np.ndarray:
    """Mean of each weather variable over hours [start, end), per county -> (L, F), float32.

    The sum runs along time in float32 with numpy's pairwise summation (time is made the contiguous axis), which
    is how the paper's code computed it. Gradient-boosted trees are sensitive to the last bits of their inputs,
    so the summation order is fixed here instead of being left to the array's memory layout."""
    block = np.ascontiguousarray(weather[:, start:end, :].transpose(0, 2, 1))
    return block.mean(axis=2)


# ---------------------------------------------------------------- scoring and output


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Per-county RMSE and MAE over the test target hours, averaged over counties."""
    err = np.asarray(pred, dtype=np.float64) - truth
    return {"rmse": float(np.sqrt((err**2).mean(axis=1)).mean()), "mae": float(np.abs(err).mean(axis=1).mean())}


def predictions_frame(exp: Experiment, pred: np.ndarray) -> pd.DataFrame:
    """(L, m*lookahead) predictions -> long frame with columns timestamp, location, pred."""
    n = pred.shape[1]
    return pd.DataFrame(
        {
            "timestamp": np.tile(exp.target_ts, len(exp.locs)),
            "location": np.repeat(exp.locs, n),
            "pred": np.asarray(pred, dtype=np.float64).ravel(),
        }
    )


def read_predictions(path, exp: Experiment) -> np.ndarray:
    """Read an outages_forecast.csv written from `predictions_frame` back into a (county, test target hour) array
    ordered like `exp`.
    """
    df = pd.read_csv(path, dtype={"location": str}, parse_dates=["timestamp"])
    piv = df.pivot(index="location", columns="timestamp", values="pred")
    return piv.loc[exp.locs, exp.target_ts].values
