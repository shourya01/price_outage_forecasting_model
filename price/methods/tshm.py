"""TSHM: two-stage hurdle model with gradient-boosted stages.

Stage 1 is a LightGBM classifier of outage occurrence (y > 0), calibrated on the validation split with Dirichlet
calibration. Stage 2 is a LightGBM Tweedie regressor of the count, fit on positive samples. The forecast is
P(y > 0) x E[y | y > 0]. Features are hourly lags of outages over the lookback, the previous-hour value and change
of each weather variable, calendar terms and a county target encoding. Windows are forecast recursively: each
step feeds its own prediction back as the latest outage lag, with forecast weather over the horizon.

The feature construction and recursive prediction are kept as in the reference implementation."""

from __future__ import annotations

import math
import random
import re
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator

NAME = "TSHM"
NEEDS_GPU = False

POST_24_LAG_SPACING = 1
LAG_LIST: List[int] = []
ROLL_SUM_WINDOWS: List[int] = []
ROLL_MAX_WINDOWS: List[int] = []
WEATHER_ROLL_MAX_WINDOWS: List[int] = []
WEATHER_ROLL_SUM_WINDOWS: List[int] = []
REQUIRED_HISTORY = 24
CLIP_QUANTILES = (0.005, 0.995)
MIN_STD = 1e-6


def sanitize_feature_name(name: str) -> str:
    """Lower-case a variable name and replace runs of non-alphanumeric characters by underscores."""
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    safe = safe.strip("_")
    return safe or "feature"


def build_feature_matrix(df: pd.DataFrame, hazard_cols: List[str]) -> pd.DataFrame:
    """Features for fitting, one row per (county, hour), built causally from each county's history.

    Outage lags and rolling statistics, hours since the last non-zero outage, the previous-hour value and one-hour
    change of each weather column, and hour and day-of-week terms. `_history_len` records how many past hours were
    available for the row.
    """
    feature_frames: List[pd.DataFrame] = []
    for county, group in df.sort_values(["county_id", "ts"]).groupby("county_id", sort=False):
        out_history: deque = deque(maxlen=REQUIRED_HISTORY)
        run_length: int | None = None
        weather_histories: Dict[str, deque] = {col: deque(maxlen=REQUIRED_HISTORY) for col in hazard_cols}
        records: Dict[int, Dict[str, float]] = {}

        for idx, row in group.iterrows():
            features: Dict[str, float] = {}
            hist = list(out_history)
            hist_len = len(hist)
            features["_history_len"] = float(hist_len)

            for lag in LAG_LIST:
                features[f"out_lag_{lag}"] = hist[-lag] if hist_len >= lag else np.nan
            for window in ROLL_SUM_WINDOWS:
                values = hist[-window:]
                features[f"out_roll_sum_{window}"] = float(np.sum(values)) if values else np.nan
            for window in ROLL_MAX_WINDOWS:
                values = hist[-window:]
                features[f"out_roll_max_{window}"] = float(np.max(values)) if values else np.nan

            features["out_run_length_since_nonzero"] = float(run_length) if run_length is not None else np.nan

            for col in hazard_cols:
                w_hist = weather_histories[col]
                w_list = list(w_hist)
                w_len = len(w_list)
                features[f"{col}_lag1"] = w_list[-1] if w_len >= 1 else np.nan
                if w_len >= 2:
                    features[f"{col}_delta_1"] = w_list[-1] - w_list[-2]
                elif w_len == 1:
                    features[f"{col}_delta_1"] = 0.0
                else:
                    features[f"{col}_delta_1"] = np.nan
                for window in WEATHER_ROLL_MAX_WINDOWS:
                    values = w_list[-window:]
                    features[f"{col}_roll_max_{window}"] = float(np.nanmax(values)) if values else np.nan
                for window in WEATHER_ROLL_SUM_WINDOWS:
                    values = w_list[-window:]
                    features[f"{col}_roll_sum_{window}"] = float(np.nansum(values)) if values else np.nan

            ts = row["ts"]
            hour = ts.hour
            dow = ts.dayofweek
            features["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
            features["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
            features["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
            features["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)

            records[idx] = features

            out_val = row["outages"]
            out_val = 0.0 if pd.isna(out_val) else float(out_val)
            out_history.append(out_val)
            if out_val > 0:
                run_length = 0
            else:
                run_length = 1 if run_length is None else run_length + 1

            for col in hazard_cols:
                w_val = row[col]
                w_val = float(w_val) if not pd.isna(w_val) else np.nan
                weather_histories[col].append(w_val)

        county_frame = pd.DataFrame.from_dict(records, orient="index")
        feature_frames.append(county_frame)

    features = pd.concat(feature_frames, axis=0).sort_index()
    return features


def build_dynamic_lag_list(lookback: int) -> List[int]:
    """Dense hourly lags 1..lookback."""
    if lookback < 1:
        raise ValueError("Lookback must be at least 1 hour for dense lag generation.")
    return list(range(1, lookback + 1))


def update_required_history(lags: List[int]) -> None:
    """Set REQUIRED_HISTORY to the longest outage lag or rolling window in use."""
    global REQUIRED_HISTORY
    candidates: List[int] = []
    if lags:
        candidates.append(max(lags))
    if ROLL_SUM_WINDOWS:
        candidates.append(max(ROLL_SUM_WINDOWS))
    if ROLL_MAX_WINDOWS:
        candidates.append(max(ROLL_MAX_WINDOWS))
    if not candidates:
        candidates.append(REQUIRED_HISTORY)
    REQUIRED_HISTORY = max(candidates)


def get_outage_history_length() -> int:
    """Number of past outage values the recursive forecast has to keep."""
    lengths: List[int] = [REQUIRED_HISTORY]
    lengths.extend(LAG_LIST)
    lengths.extend(ROLL_SUM_WINDOWS)
    lengths.extend(ROLL_MAX_WINDOWS)
    if not lengths:
        return 1
    return max(max(lengths), 1)


def get_weather_history_length() -> int:
    """Number of past weather values the recursive forecast has to keep (at least 2, for the one-hour change)."""
    lengths: List[int] = [2]
    lengths.extend(WEATHER_ROLL_MAX_WINDOWS)
    lengths.extend(WEATHER_ROLL_SUM_WINDOWS)
    if not lengths:
        return 1
    return max(max(lengths), 1)


def predict_counts(
    classifier: lgb.LGBMClassifier,
    calibrator: Any | None,
    regressor: lgb.LGBMRegressor,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (occurrence probability, count given occurrence, expected count = their product) for the rows of X.

    The probability is calibrated when a calibrator is given.
    """
    probs_raw = classifier.predict_proba(X)
    if calibrator is not None:
        if FullDirichletCalibrator is not None and isinstance(calibrator, FullDirichletCalibrator):
            probs = calibrator.predict_proba(probs_raw)[:, 1]
        else:
            probs = np.asarray(calibrator.transform(probs_raw[:, 1]))
    else:
        probs = probs_raw[:, 1]
    intensity = regressor.predict(X)
    intensity = np.maximum(intensity, 0.0)
    expected = np.maximum(probs * intensity, 0.0)
    return probs, intensity, expected


def build_histories_for_county(
    county_df: pd.DataFrame,
    hazard_cols: List[str],
    cutoff_ts: pd.Timestamp,
    include_current: bool,
    outage_history_len: int,
    weather_history_len: int,
    hazard_medians: Dict[str, float],
) -> Tuple[deque, float, Dict[str, deque]]:
    """Outage history, length of the current zero-outage run and weather histories of one county, from observations
    before `cutoff_ts` (up to and including it with `include_current`), padded to the required lengths.
    """
    mask = county_df["ts"] <= cutoff_ts if include_current else county_df["ts"] < cutoff_ts
    subset = county_df.loc[mask].sort_values("ts")
    outages = subset["outages"].astype(float).tolist()
    outage_vals = [0.0 if pd.isna(val) else float(val) for val in outages]
    history_vals = outage_vals[-outage_history_len:]
    while len(history_vals) < outage_history_len:
        history_vals.insert(0, 0.0)
    outage_history = deque(history_vals, maxlen=outage_history_len)
    run_length = 0.0
    for val in reversed(outage_vals):
        if val > 0:
            run_length = 0.0
            break
        run_length += 1.0
    hazard_histories: Dict[str, deque] = {}
    for hazard in hazard_cols:
        if hazard in subset.columns:
            hazard_values = subset[hazard].astype(float).tolist()
        else:
            hazard_values = []
        cleaned = [
            hazard_medians.get(hazard, 0.0) if pd.isna(val) else float(val)
            for val in hazard_values[-weather_history_len:]
        ]
        while len(cleaned) < weather_history_len:
            cleaned.insert(0, hazard_medians.get(hazard, 0.0))
        hazard_histories[hazard] = deque(cleaned, maxlen=weather_history_len)
    return outage_history, run_length, hazard_histories


def get_weather_inputs(
    county: str,
    ts: pd.Timestamp,
    hazard_cols: List[str],
    forecast_lookup: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] | None,
    county_weather_lookup: Dict[str, pd.DataFrame],
    hazard_medians: Dict[str, float],
) -> Dict[str, float]:
    """Weather of a county at hour `ts`: the forecast when `forecast_lookup` holds that (county, hour), otherwise the
    observed value. Missing values fall back to the training medians.
    """
    key = (county, ts)
    if forecast_lookup and key in forecast_lookup:
        entry = {}
        for hazard in hazard_cols:
            val = forecast_lookup[key].get(hazard, float("nan"))
            entry[hazard] = hazard_medians.get(hazard, 0.0) if pd.isna(val) else float(val)
        return entry
    weather_df = county_weather_lookup.get(county)
    if weather_df is not None and ts in weather_df.index:
        series = weather_df.loc[ts]
        return {
            hazard: (
                hazard_medians.get(hazard, 0.0)
                if pd.isna(series.get(hazard, float("nan")))
                else float(series.get(hazard))
            )
            for hazard in hazard_cols
        }
    return {hazard: hazard_medians.get(hazard, 0.0) for hazard in hazard_cols}


def sequential_test_predictions(
    df: pd.DataFrame,
    classifier: lgb.LGBMClassifier,
    calibrator: IsotonicRegression | None,
    regressor: lgb.LGBMRegressor,
    model_features: List[str],
    cat_features: List[str],
    clip_bounds: Dict[str, Tuple[float | None, float | None]],
    medians: Dict[str, float],
    county_to_idx: Dict[str, int],
    category_dtype: pd.api.types.CategoricalDtype,
    hazard_cols: List[str],
    lookahead: int,
    val_test: int,
    weather_forecast: Dict[Tuple[str, pd.Timestamp], Dict[str, float]] | None,
    hazard_medians: Dict[str, float],
) -> pd.DataFrame:
    """Recursive forecasts over the test targets.

    For each county and each of the `val_test` windows of `lookahead` hours, the features are rebuilt hour by hour from
    a history into which the model's own predictions are fed back; after a window the history is reset from the
    observations. Returns a frame with columns county_id, ts, outages, prediction, p_occ, intensity.
    """
    if val_test <= 0 or lookahead <= 0:
        return pd.DataFrame(columns=["county_id", "ts", "outages", "prediction", "p_occ", "intensity"])
    feature_set = model_features + cat_features
    interaction_features = [col for col in model_features if col.startswith("int_out1_")]
    county_groups = {county: group.sort_values("ts").copy() for county, group in df.groupby("county_id", sort=False)}
    county_weather_lookup = {
        county: group.set_index("ts")[hazard_cols] if hazard_cols else pd.DataFrame(index=group["ts"])
        for county, group in county_groups.items()
    }
    county_te_values = df.groupby("county_id")["county_te"].first().to_dict()
    outage_history_len = get_outage_history_length()
    weather_history_len = get_weather_history_length()
    results: List[Dict[str, Any]] = []

    for county, county_df in county_groups.items():
        county_test = county_df[county_df["split"] == "test"]
        if county_test.empty:
            continue
        required = lookahead * val_test
        if len(county_test) < required:
            raise ValueError(f"County {county} test window is too short for {val_test} lookahead blocks.")
        target_rows = county_test.tail(required).sort_values("ts")
        eval_rows = list(target_rows.itertuples(index=False))
        start_ts = eval_rows[0].ts
        outage_history, run_length, hazard_histories = build_histories_for_county(
            county_df,
            hazard_cols,
            start_ts,
            include_current=False,
            outage_history_len=outage_history_len,
            weather_history_len=weather_history_len,
            hazard_medians=hazard_medians,
        )

        for block_idx in range(val_test):
            block_rows = eval_rows[block_idx * lookahead : (block_idx + 1) * lookahead]
            if not block_rows:
                break
            for row in block_rows:
                ts = row.ts
                hour = ts.hour
                dow = ts.dayofweek
                feature_values: Dict[str, float] = {}
                history_list = list(outage_history)
                for lag in LAG_LIST:
                    col_name = f"out_lag_{lag}"
                    if len(history_list) >= lag:
                        feature_values[col_name] = float(history_list[-lag])
                    else:
                        feature_values[col_name] = float(medians.get(col_name, 0.0))
                for window in ROLL_SUM_WINDOWS:
                    col_name = f"out_roll_sum_{window}"
                    if len(history_list) >= window:
                        feature_values[col_name] = float(np.sum(history_list[-window:]))
                    else:
                        feature_values[col_name] = float(medians.get(col_name, 0.0))
                for window in ROLL_MAX_WINDOWS:
                    col_name = f"out_roll_max_{window}"
                    if len(history_list) >= window:
                        feature_values[col_name] = float(np.max(history_list[-window:]))
                    else:
                        feature_values[col_name] = float(medians.get(col_name, 0.0))
                feature_values["out_run_length_since_nonzero"] = float(run_length)

                hazard_inputs = get_weather_inputs(
                    county,
                    ts,
                    hazard_cols,
                    weather_forecast,
                    county_weather_lookup,
                    hazard_medians,
                )
                hazard_updates: Dict[str, float] = {}
                for hazard in hazard_cols:
                    history = hazard_histories[hazard]
                    hazard_updates[hazard] = hazard_inputs[hazard]
                    lag_key = f"{hazard}_lag1"
                    if len(history):
                        feature_values[lag_key] = float(history[-1])
                    else:
                        feature_values[lag_key] = float(hazard_medians.get(hazard, 0.0))
                    delta_key = f"{hazard}_delta_1"
                    if len(history) >= 2:
                        feature_values[delta_key] = float(history[-1] - history[-2])
                    elif len(history) == 1:
                        feature_values[delta_key] = 0.0
                    else:
                        feature_values[delta_key] = float(medians.get(delta_key, 0.0))
                    for window in WEATHER_ROLL_MAX_WINDOWS:
                        roll_key = f"{hazard}_roll_max_{window}"
                        if len(history) >= window:
                            feature_values[roll_key] = float(np.nanmax(list(history)[-window:]))
                        else:
                            feature_values[roll_key] = float(hazard_medians.get(hazard, 0.0))
                for window in WEATHER_ROLL_SUM_WINDOWS:
                    roll_key = f"{hazard}_roll_sum_{window}"
                    if len(history) >= window:
                        feature_values[roll_key] = float(np.nansum(list(history)[-window:]))
                    else:
                        feature_values[roll_key] = float(medians.get(roll_key, 0.0))

                feature_values["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
                feature_values["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
                feature_values["dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
                feature_values["dow_cos"] = math.cos(2 * math.pi * dow / 7.0)
                feature_values["county_te"] = float(county_te_values.get(county, medians.get("county_te", 0.0)))
                feature_values["county_idx"] = county_to_idx[county]

                out_lag1_val = feature_values.get("out_lag_1", float(medians.get("out_lag_1", 0.0)))
                for interaction_key in interaction_features:
                    hazard_key = interaction_key.replace("int_out1_", "")
                    hazard_val = feature_values.get(hazard_key, float(medians.get(hazard_key, 0.0)))
                    feature_values[interaction_key] = out_lag1_val * hazard_val

                row_values = {col: feature_values.get(col, medians.get(col, 0.0)) for col in model_features}
                for col, bounds in clip_bounds.items():
                    if col not in row_values:
                        continue
                    lower, upper = bounds
                    val = row_values[col]
                    if lower is not None:
                        val = max(val, lower)
                    if upper is not None:
                        val = min(val, upper)
                    row_values[col] = val

                row_df = pd.DataFrame([row_values])
                for cat_feature in cat_features:
                    if cat_feature == "county_idx":
                        row_df[cat_feature] = pd.Series([county_to_idx[county]], dtype=category_dtype)
                    else:
                        row_df[cat_feature] = medians.get(cat_feature, 0.0)
                row_df = row_df[feature_set]

                probs, intensity, expected = predict_counts(
                    classifier,
                    calibrator,
                    regressor,
                    row_df,
                )

                prediction = float(expected[0])
                p_occ = float(probs[0])
                results.append(
                    {
                        "county_id": county,
                        "ts": ts,
                        "outages": float(row.outages),
                        "prediction": prediction,
                        "p_occ": p_occ,
                        "intensity": float(intensity[0]),
                    }
                )

                outage_history.append(prediction)
                if p_occ >= 0.5:
                    run_length = 0.0
                else:
                    run_length = run_length + 1.0

                for hazard, value in hazard_updates.items():
                    hazard_histories[hazard].append(value)

            block_end_ts = block_rows[-1].ts
            outage_history, run_length, hazard_histories = build_histories_for_county(
                county_df,
                hazard_cols,
                block_end_ts,
                include_current=True,
                outage_history_len=outage_history_len,
                weather_history_len=weather_history_len,
                hazard_medians=hazard_medians,
            )

    return pd.DataFrame(results)


def run(exp, *, seed=42, tweedie_power=1.55, calibrate=True, **_):
    """TSHM forecasts for the test targets, (counties, m * lookahead).

    `tweedie_power` is the variance power of the count stage; `calibrate` switches the Dirichlet calibration of the
    occurrence probability.
    """
    lookback, lookahead, m = exp.lookback, exp.lookahead, exp.m
    lag_list = build_dynamic_lag_list(lookback)
    globals()["LAG_LIST"] = lag_list
    update_required_history(lag_list)
    random.seed(seed)
    np.random.seed(seed)

    # long frame: one row per (hour, county), weather columns prefixed wx_
    hazard_cols = [f"wx_{sanitize_feature_name(f)}" for f in exp.features]
    if len(set(hazard_cols)) != len(hazard_cols):
        raise ValueError("weather variable names collide after sanitizing")
    index = pd.MultiIndex.from_product([exp.ts, exp.locs], names=["ts", "county_id"])
    df = pd.DataFrame(index=index).reset_index()
    df["outages"] = exp.out.T.reshape(-1)
    for j, col in enumerate(hazard_cols):
        df[col] = exp.weather64[:, :, j].T.reshape(-1)
    split = np.array(["train"] * exp.T, dtype=object)
    split[exp.val_range[0] : exp.val_range[1]] = "val"
    split[exp.test_range[0] : exp.test_range[1]] = "test"
    df["split"] = df["ts"].map(dict(zip(exp.ts, split)))

    feature_df = build_feature_matrix(df, hazard_cols)
    df = df.join(feature_df, how="left")
    df = df[df["_history_len"] >= REQUIRED_HISTORY].copy()
    df.drop(columns=["_history_len"], inplace=True)
    df.sort_values(["county_id", "ts"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"
    if not train_mask.any():
        raise RuntimeError("Training split has no samples after feature filtering.")

    county_to_idx = {cid: idx for idx, cid in enumerate(sorted(set(df["county_id"].tolist())))}
    df["county_idx"] = df["county_id"].map(county_to_idx)
    category_dtype = pd.api.types.CategoricalDtype(categories=list(county_to_idx.values()), ordered=False)
    df["county_idx"] = df["county_idx"].astype(category_dtype)

    county_te_map = df.loc[train_mask].groupby("county_id")["outages"].mean()
    global_mean = float(df.loc[train_mask, "outages"].mean())
    df["county_te"] = df["county_id"].map(county_te_map).fillna(global_mean)

    feature_cols = [
        *(f"out_lag_{lag}" for lag in LAG_LIST),
        *(f"out_roll_sum_{window}" for window in ROLL_SUM_WINDOWS),
        *(f"out_roll_max_{window}" for window in ROLL_MAX_WINDOWS),
        "out_run_length_since_nonzero",
    ]
    for col in hazard_cols:
        feature_cols.extend(
            [
                f"{col}_lag1",
                f"{col}_delta_1",
                *(f"{col}_roll_max_{window}" for window in WEATHER_ROLL_MAX_WINDOWS),
                *(f"{col}_roll_sum_{window}" for window in WEATHER_ROLL_SUM_WINDOWS),
            ]
        )
    feature_cols.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos", "county_te"])
    feature_cols = [col for col in feature_cols if col in df.columns]

    interaction_cols: List[str] = []
    if "out_lag_1" in df.columns:
        top_hazards = hazard_cols[: min(5, len(hazard_cols))]
        for hz in top_hazards:
            lag_col = f"{hz}_lag1"
            if lag_col not in df.columns:
                continue
            new_col = f"int_out1_{lag_col}"
            df[new_col] = df["out_lag_1"] * df[lag_col]
            interaction_cols.append(new_col)
    if interaction_cols:
        feature_cols.extend(interaction_cols)

    df[feature_cols] = df.groupby("county_id")[feature_cols].ffill()
    medians_series = df.loc[train_mask, feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians_series)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    train_stats = df.loc[train_mask, feature_cols]
    clip_bounds: Dict[str, Tuple[float | None, float | None]] = {}
    for col in feature_cols:
        col_data = train_stats[col].dropna()
        if col_data.empty:
            clip_bounds[col] = (None, None)
            continue
        lower = float(col_data.quantile(CLIP_QUANTILES[0]))
        upper = float(col_data.quantile(CLIP_QUANTILES[1]))
        df[col] = df[col].clip(lower=lower, upper=upper)
        clip_bounds[col] = (lower, upper)

    stds = train_stats.std().fillna(0.0)
    keep_cols = stds[stds > MIN_STD].index.tolist()
    dropped_cols = sorted(set(feature_cols) - set(keep_cols))
    if dropped_cols:
        df.drop(columns=dropped_cols, inplace=True)
    feature_cols = [col for col in feature_cols if col in keep_cols]

    medians = {col: float(medians_series.get(col, 0.0)) for col in feature_cols}
    hazard_medians = {
        hazard: float(df.loc[train_mask, hazard].median(skipna=True)) if hazard in df.columns else 0.0
        for hazard in hazard_cols
    }

    cat_features = ["county_idx"]
    eval_features = feature_cols + cat_features

    df["y_occ"] = (df["outages"] > 0).astype(int)
    df["y_cnt"] = df["outages"].astype(float)

    calibrator: IsotonicRegression | None = None
    X_train = df.loc[train_mask, eval_features]
    y_train_occ = df.loc[train_mask, "y_occ"].to_numpy()
    pos_count = y_train_occ.sum()
    neg_count = len(y_train_occ) - pos_count
    if pos_count == 0:
        raise RuntimeError("Training data contains no outage events; cannot fit occurrence model.")
    scale_pos_weight = float(neg_count / pos_count) if pos_count else 1.0

    classifier_params: Dict[str, Any] = {
        "objective": "binary",
        "learning_rate": 0.005,
        "num_leaves": 2000,
        "max_depth": 10,
        "min_child_samples": 300,
        "min_sum_hessian_in_leaf": 25.0,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "feature_fraction_bynode": 0.8,
        "reg_alpha": 0.03,
        "reg_lambda": 0.03,
        "n_estimators": 12000,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "num_threads": 1,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
    }
    classifier = lgb.LGBMClassifier(**classifier_params)
    classifier.fit(
        X_train,
        y_train_occ,
        eval_set=[(df.loc[val_mask, eval_features], df.loc[val_mask, "y_occ"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(200, verbose=False)],
        categorical_feature=cat_features,
    )

    if calibrate:
        val_occ = df.loc[val_mask, "y_occ"].to_numpy()
        val_probs_raw = classifier.predict_proba(df.loc[val_mask, eval_features])
        if len(np.unique(val_occ)) >= 2:
            calibrator = FullDirichletCalibrator(reg_lambda=0.0)
            calibrator.fit(val_probs_raw, val_occ)

    train_positive_mask = train_mask & (df["y_occ"] == 1)
    val_positive_mask = val_mask & (df["y_occ"] == 1)
    X_train_pos = df.loc[train_positive_mask, eval_features]
    y_train_pos = df.loc[train_positive_mask, "y_cnt"].to_numpy()
    X_val_pos = df.loc[val_positive_mask, eval_features]
    y_val_pos = df.loc[val_positive_mask, "y_cnt"].to_numpy()

    regressor_params: Dict[str, Any] = {
        "objective": "tweedie",
        "tweedie_variance_power": float(tweedie_power),
        "learning_rate": 0.005,
        "num_leaves": 1000,
        "max_depth": 10,
        "min_child_samples": 300,
        "min_sum_hessian_in_leaf": 25.0,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "feature_fraction_bynode": 0.8,
        "reg_alpha": 0.03,
        "reg_lambda": 0.03,
        "n_estimators": 12000,
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "num_threads": 1,
        "verbosity": -1,
    }
    regressor = lgb.LGBMRegressor(**regressor_params)
    eval_set = [(X_val_pos, y_val_pos)] if len(X_val_pos) else [(X_train_pos, y_train_pos)]
    regressor.fit(
        X_train_pos,
        y_train_pos,
        eval_set=eval_set,
        eval_metric="mae",
        callbacks=[lgb.early_stopping(200, verbose=False)],
        categorical_feature=cat_features,
    )

    # forecast weather over the test target hours, keyed by (county, hour)
    weather_forecast = {}
    for c, loc in enumerate(exp.locs):
        for h, ts in enumerate(exp.target_ts):
            weather_forecast[(loc, ts)] = {col: float(exp.future[c, h, j]) for j, col in enumerate(hazard_cols)}
    test_predictions = sequential_test_predictions(
        df,
        classifier,
        calibrator,
        regressor,
        feature_cols,
        cat_features,
        clip_bounds,
        medians,
        county_to_idx,
        category_dtype,
        hazard_cols,
        lookahead,
        m,
        weather_forecast,
        hazard_medians,
    )
    piv = test_predictions.pivot(index="county_id", columns="ts", values="prediction")
    return piv.loc[exp.locs, exp.target_ts].values
