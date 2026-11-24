# PRiCE: outage forecasting over sensor networks

Code for the paper

> Shourya Bose, Tomas Kaljevic, Yu Zhang.
> *PRiCE: A Two-Expert Hybrid Model for Outage Forecasting over Sensor Networks.*

## What it is

PRiCE forecasts hourly county-level power outage counts 24 or 48 hours ahead from the last 48 hours of outages and
weather. It combines two experts:

- **Persistence expert.** The last observed outage count of each county, held over the whole horizon.
- **Cluster expert.** A per-county AR(2) backbone plus a gradient-boosted residual regressor (LightGBM,
  absolute-error loss) shared by the counties of a cluster. The backbone is fit by Yule-Walker on the training
  split; when the estimate is not stationary it falls back to AR(1). Counties are clustered by k-means on seven
  training-split statistics, and the number of clusters K is chosen by silhouette from {3, 4, 5, 6, 8}.

The two experts are combined by a shifted geometric mean,

$$\hat y = \max\{\sqrt{(E_p+\kappa)(E_c+\kappa)}-\kappa,\ 0\}, \qquad \kappa = 0.5,$$

so that a zero from one expert does not force the forecast to zero. The shift is fixed, not tuned (`KAPPA` in
`price/methods/price.py`).

This repository builds the paper's main results table (PRiCE and eight baselines on two datasets and two
lookaheads) and its ablation table.

| Key | Method | Weather | Device |
|---|---|---|---|
| `zeros` | All-Zeros | no | CPU |
| `arima` | ARIMA(1,0,1), per county | no | CPU |
| `arimax` | ARIMAX(1,0,1), per county, weather as exogenous regressors | yes | CPU |
| `lstm` | LSTM | yes | GPU |
| `lstm_sparsity` | LSTM, one model per sparsity group of counties | yes | GPU |
| `zip` | Zero-inflated Poisson with gradient-boosted stages | yes | CPU |
| `tshm` | Two-stage hurdle model (occurrence classifier, Tweedie count regressor) | yes | CPU |
| `timesfm` | TimesFM 3.0 on outages with weather covariates, zero-shot | yes | GPU |
| `price` | PRiCE | yes | CPU |

Each method is one module in `price/methods/` exposing `run(experiment, **options) -> predictions`.

## Install

Python 3.12.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build; skip for CPU only
pip install -r requirements.txt
```

The TimesFM 3.0 baseline downloads its pretrained checkpoint (`google/timesfm-3.0-pytorch`) from Hugging Face on
first use.

## Data layout expected

No data is shipped with this repository. Two inputs are needed: a directory with the outage files and a directory
with the weather forecasts.

### Outage data

`<data-dir>/michigan.nc` and `<data-dir>/california.nc`, NetCDF files with

| Variable | Dimensions | Content |
|---|---|---|
| `out` | location, timestamp | hourly outage counts per county |
| `weather` | location, timestamp, feature | hourly weather covariates |
| `timestamp` | timestamp | hourly timestamps |
| `location` | location | county identifiers |
| `feature` | feature | names of the weather variables |

- **California** is public: <https://huggingface.co/datasets/shourya01/ca_power_outages>. Download
  `california_hourly.nc` and save it as `california.nc` (commands below).
- **Michigan** comes from the INFORMS DMDA 2025 data challenge
  (<https://sites.google.com/view/dmdaworkshop2025/data-challenge>) and is not redistributable. After obtaining it
  from the organizers, build `michigan.nc` with the layout above.

### Weather forecasts

One CSV per state and lookahead with columns `timestamp, county, <weather variables>` and one row per county and
test target hour. `county` holds the identifiers of `location`. Accepted locations inside `<forecast-dir>`:

```
<state>_<lookahead>.csv                                   e.g. michigan_24.csv
<state>*_<lookahead>_*/weather_forecast.csv               e.g. california_672_24_zero_shot/weather_forecast.csv
```

The paper uses zero-shot TimesFM 3.0 forecasts of each weather variable (one univariate forecast per county and
variable). A variable that the forecast file does not contain is held at its last observed value over the horizon
(California's precipitation, `tp`).

- **California**: the forecasts used in the paper are in the `forecasts/` folder of the same Hugging Face dataset
  (24 h lookahead from a 672 h context, 48 h lookahead from a 336 h context).
- **Michigan**: the forecasts are derived from the challenge data and are not redistributed. Produce them with
  TimesFM 3.0 in the format above (the paper uses a 672 h context for both lookaheads).

To fetch everything public for California:

```bash
pip install huggingface_hub
hf download shourya01/ca_power_outages california_hourly.nc --repo-type dataset --local-dir <data-dir>
mv <data-dir>/california_hourly.nc <data-dir>/california.nc
hf download shourya01/ca_power_outages --repo-type dataset --include "forecasts/*" --local-dir <download-dir>
# then pass <download-dir>/forecasts as <forecast-dir>
```

With only the California inputs, add `--datasets california` to `price.run` and `price.ablation`.

## How to reproduce the two tables

```bash
./run_all.sh <data-dir> <forecast-dir> [results-dir]     # runs every method, then builds both tables
```

`run_all.sh` runs three commands, which can also be run separately from the repository root:

```bash
python -m price.run      --data-dir <data-dir> --forecast-dir <forecast-dir> --out results   # main table: every method
python -m price.table    --results results
python -m price.ablation --data-dir <data-dir> --forecast-dir <forecast-dir> --out results   # ablation table of PRiCE
```

A task that already has results is skipped, so an interrupted run can be restarted; `--force` reruns everything.
Useful options of `price.run`: `--methods`, `--datasets`, `--lookaheads`, `--jobs` (CPU tasks at once), `--device`.
`python -m price.ablation --table-only --out results` rebuilds the ablation table from existing metrics.

### Outputs

```
results/<state>_<lookahead>/<method>/outages_forecast.csv   timestamp, location, pred
results/<state>_<lookahead>/<method>/metrics.json           rmse, mae, runtime, variables, forecast handling
results/main_table.{md,tex,csv}
results/<state>_<lookahead>/ablation/seed_<seed>/<variant>/metrics.json
results/ablation_table.{md,tex,csv}
```

### Reference tables

`results/main_table.{md,tex,csv}` and `results/ablation_table.{md,tex,csv}` in this repository are the two tables
produced by this code with the inputs described above (seed 42), for comparison with a fresh run. A run that writes
to `results/` overwrites them. The `.tex` files use the paper's macros (`\price`, `\ep`, `\ec`, `\eb`, `\mb`).

### Protocol shared by all methods

Splits, weather variables, test-time weather and scoring live in `price/data.py` and are the same for every method.
What each method fits on follows its original design: ARIMA and ARIMAX are fit on all hours before the test targets,
ZIP refits on training plus validation after early stopping, and the others fit on training and use validation for
early stopping or calibration.

- **Horizon.** Lookback 48 h; lookahead 24 h or 48 h; `m` = 5 (Michigan) or 30 (California) test windows.
- **Splits.** The test split is the last 48 + m x lookahead hours, of which the last m x lookahead hours are the
  test targets, forecast in m consecutive windows. The validation split is the same length, immediately before.
  Training is everything earlier.
- **Weather variables.** The 24 variables with the largest mean absolute correlation with outages on the training
  split. Michigan's file has 109 variables, so this is a selection (it differs between the two lookaheads because
  the training split does). California's file has exactly 24.
- **Weather at test time.** Models are trained and validated on actual weather. For a test window, the lookback
  uses actual weather and the target hours use the forecast file. Forecasts are clipped to physical bounds.
- **Missing hours.** The weather is used as stored. Michigan has two hours whose weather is zero in every variable
  and county (both in the training period); they are left as they are.
- **Metrics.** RMSE and MAE per county over the test targets, averaged over counties.

### Ablation table

`python -m price.ablation` reports the test RMSE of PRiCE with one part changed:

| Group | Variant |
|---|---|
| Component | persistence expert alone; cluster expert alone |
| Backbone | ARIMA(1,0,1) or AR(1) in place of AR(2) |
| Aggregator | arithmetic mean or a convex combination (weight tuned on validation) in place of the shifted geometric mean |
| Weather | target-hour weather replaced by the last observed weather, in training and at test time |

Four models are trained per state and lookahead (full, the two backbones, the weather variant); the component and
aggregator rows are read off the full model. The default is one seed (42). `--seeds 42 1 2 ...` repeats the trainings
per seed and the table then reports the mean over the seeds given.

### Runtime

The full table takes a few hours on 8 CPU cores and one GPU. The California runs of `tshm`, `zip` and the two LSTMs
dominate. The California LSTMs hold about 7 GB of training windows in memory each; they run one at a time.

### Reproducibility

One seed (default 42) is used for everything, and LightGBM runs with a fixed number of threads (4 for PRiCE and ZIP,
1 for TSHM), so a rerun on the same machine gives the same numbers. The county clustering of PRiCE uses its own
fixed seed. Across machines, expect the LightGBM-based methods to match, ARIMA/ARIMAX to agree to about 0.1%, and
the GPU-trained LSTMs to vary slightly.

## Repository layout

```
price/data.py            experiment loading, splits, variable selection, forecast handling, scoring
price/methods/           one module per method, plus the registry
price/run.py             task runner for the main table
price/table.py           main table builder
price/ablation.py        ablation runner and table builder
run_all.sh               all steps
results/                 the two reference tables (md, tex, csv)
tests/                   regression scripts; they train models and need the datasets (usage in each file)
requirements.txt
```

## Citation

If you use this code, please cite the paper. The entry below is a placeholder until the publication details are
final.

```bibtex
@article{bose_price,
  title   = {PRiCE: A Two-Expert Hybrid Model for Outage Forecasting over Sensor Networks},
  author  = {Bose, Shourya and Kaljevic, Tomas and Zhang, Yu},
  journal = {TODO},
  year    = {TODO}
}
```
