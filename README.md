# PRiCE: A Two-Expert Hybrid Model for Outage Forecasting over Sensor Networks

Reference implementation for the paper

> Shourya Bose, Tomas Kaljevic, Yu Zhang.
> *PRiCE: A Two-Expert Hybrid Model for Outage Forecasting over Sensor Networks.*

PRiCE combines a parameter-free persistence forecast with a per-sensor AR(2)
backbone augmented by a per-cluster gradient-boosted residual regressor, and
aggregates the two predictors by a fixed geometric mean. The architecture
targets outage forecasting over a two-layer sensor network in which one layer
(e.g., county-level outage sensors) feeds the response and the other (a
meteorological sensor network) feeds the covariates.

```
code/
├── reproduce.py         # one-click runner (auto-downloads California)
├── price_minimal.py     # forecasting script (trains and evaluates PRiCE)
├── cluster_counties.py  # silhouette-based sensor clustering
├── requirements.txt
└── README.md
```

The scripts read NetCDF data from a sibling `data/` directory by default
(`../data/` relative to this folder).

## Quick start

```bash
# 1. Set up the environment (Python 3.10+).
pip install -r requirements.txt

# 2. Reproduce the California results (auto-downloads the dataset).
python reproduce.py --dataset california

# 3. Reproduce the Michigan results (after obtaining the dataset, see below).
python reproduce.py --dataset michigan --michigan-nc /path/to/michigan.nc
```

Each call runs both 24-hour and 48-hour lookaheads. The reported metrics
match Tables I and II of the paper. Per-row predictions are saved to
`../data/{ca,mi}{24,48}_price.csv`.

## Environment

Tested with Python 3.10. `pip install -r requirements.txt` installs `numpy`,
`pandas`, `xarray`, `netCDF4`, `lightgbm`, `scipy`, `scikit-learn`, and
`huggingface_hub`. The `--ar-order arima11` flag in `price_minimal.py`
additionally requires `pip install statsmodels`, but this is optional and not
needed for the default `ar2` backbone.

## Datasets

Both datasets are stored as NetCDF with the layout

- `out` — outage counts, shape `(n_sensors, n_timesteps)`, integer-valued.
- `weather` — weather covariates, shape `(n_sensors, n_timesteps, n_features)`, float-valued.
- `timestamp` — UTC hourly timestamps, shape `(n_timesteps,)`.
- `location` — sensor identifiers, shape `(n_sensors,)`.
- `feature` — weather feature names, shape `(n_features,)`.

### California (auto-downloaded)

`reproduce.py --dataset california` downloads the preprocessed hourly NetCDF
from Hugging Face the first time it is run

- <https://huggingface.co/datasets/shourya01/ca_power_outages>

The file is saved to `../data/california.nc`. The dataset covers 58 California
counties from January 2019 through December 2020, with 24 ERA5 weather
features collocated on the outage grid.

### Michigan (user-supplied)

The Michigan dataset is the property of the INFORMS Data Mining Society and is
**not redistributable**. To obtain it, contact the organizers of the INFORMS
DMDA Data Challenge

- Challenge website <https://sites.google.com/view/dmdaworkshop2025/data-challenge>

After receiving access, build a `michigan.nc` file with the variable layout
above and pass its path to `reproduce.py` via `--michigan-nc`.

## What `reproduce.py` does

For each requested lookahead $\tau\in\{24,48\}$ the script

1. ensures the dataset is on disk (downloading California if necessary),
2. runs `cluster_counties.py` to compute the silhouette-selected $k$-means
   partition of the sensors and write `clusters_{ca,mi}_{24,48}.json`,
3. runs `price_minimal.py` with the paper's hyperparameters (lookback 48,
   400 boosting rounds with patience-30 early stopping, MAE loss, AR(2)
   backbone, 31 leaves on Michigan and 63 leaves on California, training
   stride 4 on Michigan and 8 on California) and writes the per-row test
   predictions to `../data/{ca,mi}{24,48}_price.csv`.

Each `price_minimal.py` invocation prints the per-sensor RMSE and MAE for the
all-zeros baseline, the persistence expert, the AR backbone alone, the cluster
expert, two aggregator ablations (arithmetic mean and validation-tuned convex
combination), and PRiCE itself. The component ablation in the paper
corresponds to the printed `E_persist` and `E_cluster` rows; the aggregator
ablation corresponds to the `arithmetic mean` and `val-tuned convex` rows;
the backbone and weather ablations are reproduced by re-running
`price_minimal.py` directly with the appropriate flags listed below.

All parametric components (the AR backbone and the per-cluster GBDTs) are fit
on the training set; the validation set is used only for early stopping of the
GBDTs.

## Direct command-line use

If you want to bypass `reproduce.py` and call the underlying scripts
directly, the commands used in the paper are

```bash
# Cluster the sensors (once per dataset and lookahead).
python cluster_counties.py --nc ../data/michigan.nc   --lookahead 24 --m-val 5  --m-test 5  --out ../data/clusters_mi_24.json
python cluster_counties.py --nc ../data/michigan.nc   --lookahead 48 --m-val 5  --m-test 5  --out ../data/clusters_mi_48.json
python cluster_counties.py --nc ../data/california.nc --lookahead 24 --m-val 30 --m-test 30 --out ../data/clusters_ca_24.json
python cluster_counties.py --nc ../data/california.nc --lookahead 48 --m-val 30 --m-test 30 --out ../data/clusters_ca_48.json

# Fit and evaluate PRiCE.
python price_minimal.py --nc ../data/michigan.nc   --clusters ../data/clusters_mi_24.json \
    --lookahead 24 --m-val 5  --m-test 5  --train-stride 4 --top-weather 24 \
    --n-rounds 400 --num-leaves 31 --ar-order ar2 --out-csv ../data/mi24_price.csv

python price_minimal.py --nc ../data/michigan.nc   --clusters ../data/clusters_mi_48.json \
    --lookahead 48 --m-val 5  --m-test 5  --train-stride 4 --top-weather 24 \
    --n-rounds 400 --num-leaves 31 --ar-order ar2 --out-csv ../data/mi48_price.csv

python price_minimal.py --nc ../data/california.nc --clusters ../data/clusters_ca_24.json \
    --lookahead 24 --m-val 30 --m-test 30 --train-stride 8 --top-weather 24 \
    --n-rounds 400 --num-leaves 63 --ar-order ar2 --out-csv ../data/ca24_price.csv

python price_minimal.py --nc ../data/california.nc --clusters ../data/clusters_ca_48.json \
    --lookahead 48 --m-val 30 --m-test 30 --train-stride 8 --top-weather 24 \
    --n-rounds 400 --num-leaves 63 --ar-order ar2 --out-csv ../data/ca48_price.csv
```

## Flags

- `--ar-order {arima11,ar1,ar2}` selects the per-sensor temporal backbone.
  The default is `ar2`. The `arima11` option additionally requires `pip
  install statsmodels`.
- `--weather-mode {oracle,persistence}` controls whether the future weather
  covariate is the supplied forecast (`oracle`, default) or the last observed
  reading (`persistence`).
- `--num-leaves`, `--n-rounds`, `--lr`, `--train-stride`, `--top-weather`,
  `--lookback` follow LightGBM and feature-construction conventions; the
  values used in the paper are the recommended defaults.

## Citation

If you use this code, please cite the paper.
