# Calibration aware neural pre-decoding on one Alveo U55C

Author: Nasir Ali, C-DAC Noida.

Surface code memory experiments whose noise comes from real IBM device calibrations, read from a CSV, decoded by
PyMatching with and without NVIDIA's Fast neural pre-decoder, with the pre-decoder run on one AMD Alveo U55C card.
The repository regenerates every number, table and figure of the accompanying paper and the team deck from the
scripts below.

```
calibration CSV ──> per qubit noise model ──> Stim syndromes ──> pre-decoder (CPU or U55C) ──> PyMatching
data/calibration     calib/noise.py                              model/, hls/, host/          harness/pipeline.py
```

## Results in brief

Eight IBM devices (Heron r1 to r3, Eagle r3), three placements, error scales 1, 0.5 and 0.25, d = 3 and 5 with d
rounds, X and Z bases: 108 pooled points, 43,200,000 shots (`results/summary.json`, `results/tables/decoders.csv`).

| INT8 pre-decoder + PyMatching against | d | significantly better | significantly worse | failure ratio |
|---|---|---|---|---|
| PyMatching | 3 | 35 of 54 | 4 | 0.85 to 1.17 |
| PyMatching | 5 | 44 of 54 | 4 | 0.64 to 1.06 |
| PyMatching with correlations | 3 | 21 of 54 | 12 | 0.73 to 1.20 |
| PyMatching with correlations | 5 | 0 of 54 | 51 | 1.07 to 1.61 |

- PyMatching with correlations (`enable_correlations=True`), which needs no network, is the strongest decoder at
  d = 5 on every device and setting. Putting the pre-decoder in front of it does not help.
- The pre-decoder still pays against plain PyMatching, and at d = 3 with low noise, where correlated matching is
  worse than plain matching.
- NVIDIA's network was trained on uniform noise. On strongly uneven qubits (Kingston, screened placements) it makes
  plain PyMatching worse, by up to 17%.
- Matching weights from each qubit's own rates beat device medians by up to 16% (ratio 0.97 to 1.16).
- Card: 18 of 18 runs bit exact against the integer model, and identical logical failures to the CPU integer model,
  shot by shot, in all 14 runs that have a CPU counterpart. One card decodes 4,201 d = 3 windows per second and
  2,330 at d = 5 (1.05 and 1.50 ms for a single window, through PCIe).

These are simulations with calibrated error rates, not runs on IBM hardware; see
[docs/LIMITATIONS.md](docs/LIMITATIONS.md). Design choices and their reasons are in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Quick start: d = 1, 2, 3, 5 on a laptop

```bash
git clone https://github.com/nasir26/u55c-calibrated-predecoder
cd u55c-calibrated-predecoder
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/small_codes.py data/calibration/ibm_kingston_2026-04-15.csv --d 1 2 3 5 --rounds 5 --scale 1
```

- d = 1 is one bare qubit: the error rate every code has to beat.
- d = 2 detects any single error but cannot correct it, so flagged shots are discarded and the discarded fraction
  is reported.
- d = 3 and 5 are decoded by PyMatching.
- `--scale 0.5` halves every error rate, which shows larger codes winning below threshold.

## Calibration CSVs

`data/calibration/` holds eight device snapshots exported from the fake backends of qiskit-ibm-runtime 0.50.0
(Apache License 2.0); no IBM account is needed. Any CSV in IBM's calibration layout loads, including an export from
the IBM Quantum dashboard. Headers are matched by keywords and extra columns are ignored.

| Column | Used for |
|---|---|
| Qubit | row key |
| T1 (us), T2 (us) | idle decay over each layer |
| Readout assignment error | measurement flip |
| Prob meas1 prep0 | preparation flip (half of it) |
| Readout length (ns), Reset length (ns) | how long data qubits idle |
| √x (sx) error, Gate length (ns) | one qubit gates and layer durations |
| CZ error or ECR error | two qubit gates, as `a_b:value;` per coupler |

```bash
python calib/ibm_csv.py summary data/calibration/ibm_aachen_2026-04-17.csv
pip install -r requirements-extra.txt && python calib/ibm_csv.py export --device fake_kingston --out data/calibration
```

## Full pipeline (CPU)

NVIDIA's code (pinned commit) and the Fast model weights are needed. The weights are gated: accept NVIDIA's Open
Model License on Hugging Face, then pass a read token.

```bash
scripts/setup_reference.sh
HF_TOKEN=... scripts/fetch_weights.sh fast
pytest -q
python bench/eval_ler.py  --csv data/calibration/ibm_aachen_2026-04-17.csv --mode median --scale 1 --d 5 --basis X
python bench/eval_corr.py --csv data/calibration/ibm_aachen_2026-04-17.csv --mode median --scale 1 --d 5 --basis X
```

`--mode` is `median` (every location gets the device median), `best` (lowest error qubits) or `screened` (a random
draw after removing the worst 10% of every metric; `--seed` picks the draw). Each run writes
`results/ler/<name>.json` and the per shot failure bits of every decoder (`.npz`).

| Decoder | What it is |
|---|---|
| pm_cal | PyMatching with every qubit's own rates |
| pm_median | PyMatching with device medians only |
| pm_corr | PyMatching with correlations |
| fast_fp32, fast_int8 | NVIDIA Fast pre-decoder (float or the card's integer model), then pm_cal |
| fast_fp32_corr | float pre-decoder, then pm_corr |

The whole campaign is `bench/campaign_cpu.sh` and `bench/campaign_corr.sh` (216 runs each at 200,000 shots; 104 job
hours at 8 threads per job and 18 job hours at 4 threads, as recorded in the `seconds` field of every result).

## On a U55C card

```bash
source /opt/xilinx/xrt/setup.sh
scripts/fetch_bitstreams.sh          # or build/build_hw.sh with Vitis 2023.2
scripts/build_host.sh
python bench/run_card.py --xclbin build/hw/predecoder.xclbin --device 0 \
    --config data/calibration/ibm_aachen_2026-04-17.csv:median:1:5:X
```

The bitstream is the one of the four card study
([u55c-neural-predecoder-nodes](https://github.com/nasir26/u55c-neural-predecoder-nodes)); window size (D, T up to
13) and weights are launch arguments, so no new build is needed. `run_card.py` checks both kernel instances bit
exact against `model/qmodel.py`, decodes every test window on the card, and measures throughput and latency.

## Tables, figures, deck

```bash
python analysis/summarize.py                 # results/summary.json, results/tables, results/figures, paper/generated
pip install -r requirements-extra.txt
python deck/make_deck.py                     # deck/calibrated_predecoder_team.pptx
```

Every number on a slide or in the paper is read from `results/`; none is typed in.

## Layout

| Path | Contents |
|---|---|
| `calib/` | CSV reader and writer, calibrated noise model |
| `harness/` | NVIDIA pipeline wrapper, PyMatching helpers |
| `model/` | integer model of the Fast network |
| `hls/`, `build/`, `host/` | kernel source, bitstream build, XRT host |
| `bench/` | CPU and card experiments, campaign scripts |
| `analysis/` | summary, tables, figures, paper macros |
| `examples/` | the d = 1, 2, 3, 5 teaching example |
| `deck/` | team slide deck generator |
| `results/` | all measured data |
| `tests/` | `pytest -q` |

## License and credits

Code: see [LICENSE](LICENSE). Calibration data: IBM, from qiskit-ibm-runtime (Apache License 2.0). Pre-decoder model
and evaluation code: NVIDIA Ising-Decoding, under NVIDIA's licenses; not redistributed here. Details in
[NOTICE](NOTICE).

Funding: Ministry of Electronics and Information Technology (MeitY), Government of India.
