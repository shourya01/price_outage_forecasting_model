#!/usr/bin/env bash
# One-click reproduction of the paper's tables.
#   ./run_all.sh <data-dir> <forecast-dir> [results-dir]
# <data-dir>     michigan.nc and california.nc
# <forecast-dir> one weather-forecast CSV per state and lookahead (see README)
# Set SEEDS to average the ablation table over several seeds, e.g. SEEDS="42 1 2 3 4 5 6 7 8 9 10".
set -euo pipefail
DATA=${1:?usage: ./run_all.sh <data-dir> <forecast-dir> [results-dir]}
FORECASTS=${2:?usage: ./run_all.sh <data-dir> <forecast-dir> [results-dir]}
OUT=${3:-results}
DATA=$(realpath "$DATA"); FORECASTS=$(realpath "$FORECASTS"); OUT=$(realpath -m "$OUT")   # the paths are relative to where you call from
cd "$(dirname "$0")"
python -m price.run      --data-dir "$DATA" --forecast-dir "$FORECASTS" --out "$OUT"     # main table: every method
python -m price.table    --results "$OUT"
python -m price.ablation --data-dir "$DATA" --forecast-dir "$FORECASTS" --out "$OUT" --seeds ${SEEDS:-42}   # ablation table
