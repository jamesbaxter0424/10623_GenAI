#!/usr/bin/env bash
# Full-city build. Logs go to logs/. Stages 1-3 run per borough (NYC_SET=city).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p logs
export PYTHONUNBUFFERED=1
[ "${SKIP_DOWNLOAD:-0}" = 1 ] || $PY 00b_download_city.py      2>&1 | tee logs/00b_download.log
NYC_SET=city $PY 01_lots.py         2>&1 | tee logs/city_01_lots.log
NYC_SET=city $PY 02_buildings.py    2>&1 | tee logs/city_02_buildings.log
NYC_SET=city $PY 03_blocks.py       2>&1 | tee logs/city_03_blocks.log
$PY 04b_mosaic_city.py              2>&1 | tee logs/city_04_mosaic.log
$PY 05_samples.py                   2>&1 | tee logs/city_05_samples.log
$PY qa_city.py                      2>&1 | tee logs/city_qa.log
