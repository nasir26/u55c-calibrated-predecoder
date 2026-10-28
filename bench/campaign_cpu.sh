#!/usr/bin/env bash
# CPU campaign: every calibration CSV, placements median and best, error scales 1, 0.5, 0.25, d = 3 and 5, both bases,
# all four decoders, 200,000 shots per point; plus placement variance (screened, seeds 0 to 2) on the two newest
# snapshots at scale 0.5. Jobs run 30 at a time with 8 threads each. Results: results/ler/*.json and .npz.
# Author: Nasir Ali, C-DAC Noida.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../envs/neural/bin/python}
mkdir -p results/ler/logs
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
jobs | xargs -P 30 -I{} sh -c "OMP_NUM_THREADS=8 $PY bench/eval_ler.py {} --threads 8 --shots 200000 > results/ler/logs/\$(echo '{}' | tr ' /' '__' | tr -d '-').log 2>&1 || echo FAIL {}"
echo campaign done
