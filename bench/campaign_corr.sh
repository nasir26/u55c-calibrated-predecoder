#!/usr/bin/env bash
# Correlated matching pass over every point of bench/campaign_cpu.sh (same shots). 20 jobs at a time, 4 threads each.
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
mkdir -p results/ler/logs_corr
jobs() {
  for csv in data/calibration/*.csv; do
    for mode in median best; do for s in 1 0.5 0.25; do for d in 3 5; do for b in X Z; do
      echo "--csv $csv --mode $mode --scale $s --d $d --basis $b"
    done; done; done; done
  done
  for csv in data/calibration/ibm_kingston_*.csv data/calibration/ibm_aachen_*.csv; do
    for seed in 0 1 2; do for d in 3 5; do for b in X Z; do
      echo "--csv $csv --mode screened --seed $seed --scale 0.5 --d $d --basis $b"
    done; done; done
  done
}
jobs | xargs -P 20 -I{} sh -c "OMP_NUM_THREADS=4 $PY bench/eval_corr.py {} --threads 4 > results/ler/logs_corr/\$(echo '{}' | tr ' /' '__' | tr -d '-').log 2>&1 || echo FAIL {}"
echo corr campaign done
