# Decisions

Author: Nasir Ali, C-DAC Noida. Dated entries; each records what was decided and why.

## 2026-10-22: scope

- Contribution: decoding under calibration derived noise. NVIDIA's Fast pre-decoder (trained on uniform circuit
  noise) and PyMatching are evaluated on circuits whose every fault location carries the error rate of a real IBM
  qubit or coupler, read from a CSV, and the pre-decoder runs on one Alveo U55C. The four card throughput study of
  the same kernel is earlier work (u55c-neural-predecoder-nodes) and is cited, not repeated.
- Calibration data: the snapshots shipped with qiskit-ibm-runtime 0.50.0 (`fake_provider/backends/*/props_*.json`,
  Apache License 2.0), exported to CSV once (`calib/ibm_csv.py export`); no IBM account is used. Eight devices:
  Heron r3 (Kingston 2026-04-15, Aachen 2026-04-17), Heron r2 (Fez, Marrakesh, 2025-02-26), Heron r1 (Torino),
  Eagle (Sherbrooke, Brisbane, Kyiv, 2025-02-26).
- d = 1 and d = 2 cannot correct errors (one unprotected qubit; detection only). They appear in the teaching example
  (`examples/small_codes.py`) only; decoder and FPGA results are at d = 3, 5 and 7.

## 2026-10-22: noise model

- NVIDIA's `MemoryCircuit` with boundary detectors is generated with distinct placeholder probabilities; every
  placeholder location gets a calibrated channel (table in `calib/noise.py`). This keeps NVIDIA's fault locations
  (noise in every round but the last, which is perfect; noiseless final data readout), so the network sees the
  structure it was trained on and only the magnitudes change.
- Gate errors are IBM's reported errors per gate (randomized benchmarking), converted to depolarizing probabilities
  with 3r/2 (one qubit) and 5r/4 (two qubits). A CX costs the coupler's CZ or ECR error plus two sx gates on the
  target.
- Idle noise is the Pauli twirl of T1 and T2 decay over the layer's duration; T2 is clipped to 2 T1. Data qubits
  idle through the readout and the reset of the ancillas (1.3 to 2.6 us on these devices), which makes idle noise as
  large as readout error. No dynamical decoupling is modelled beyond what the reported (echo) T2 contains.
- Preparation error: half of "Prob meas1 prep0" (the other half is attributed to the readout). An assumption; IBM
  does not report a reset error.
- Placement (the heavy hexagonal devices cannot host the square grid without SWAPs): median, best and screened modes
  (`calib/noise.py:place`). Screening removes the worst 10% of every metric. A first version drew qubits uniformly
  at random and picked qubits with 49% readout error and couplers with 9% error, which no layout tool would choose;
  it was replaced before any result was recorded. "best" first ranked couplers among all usable ones and at d = 7
  reached the worst couplers of a 176 coupler chip; it now ranks the screened pool.
- Error scale s multiplies every probability and the decay rates 1/T1, 1/T2 (s = 0.5: a device with half the
  error rates and the same profile). Reported at s = 1, 0.5, 0.25.

## 2026-10-22: residual convention of NVIDIA's evaluation module

- With no corrections, the residual that NVIDIA's `PreDecoderMemoryEvalModule` returns zeroes the last round's
  basis detector layer. On NVIDIA's own circuit (d = 3, 5, both bases, 200,000 shots each) and on calibrated
  circuits, PyMatching fails on exactly the same shots with that residual as with the raw detectors
  (`tests/test_calib.py::test_pipeline_without_corrections_is_pymatching`), so the module is used unchanged.

## 2026-10-22: metrics

- Logical error per shot of a d round memory experiment, and the error per round
  eps = (1 - (1 - 2 P)^(1/r)) / 2 for comparisons across d. A first threshold probe compared errors per shot at
  r = d and wrongly suggested every device was above threshold; per round, the Heron r3 medians are below threshold
  at s = 1.
- Integer model calibrated on 1,024 windows of separately drawn shots (seed + 7), never on test shots. Test shots:
  seed 1,000,003 + 1000 d (+1 for basis Z), the same for every decoder and for the card.

## 2026-10-28: the strongest baseline

- Plain PyMatching is not the strongest decoder that needs no network. PyMatching 2.4 with `enable_correlations`
  (pm_corr) was run on the same shots at every point (`bench/eval_corr.py`), alone and after the float pre-decoder.
  The paper's headline compares against pm_corr; the comparison with plain PyMatching is kept because it is the
  setting the pre-decoder was designed for.
- Result: at d = 5 pm_corr fails less often than pre-decoder + PyMatching at all 54 points (51 significant); putting
  the pre-decoder in front of pm_corr does not close the gap. At d = 3 and low noise pm_corr is worse than plain
  PyMatching and the pre-decoder wins. Reported as found.
- The exact sign test summed big integer binomials and stalled at tens of thousands of discordant shots; it is now a
  log space sum with the same values to 12 digits.
