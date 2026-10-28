# Limitations

Author: Nasir Ali, C-DAC Noida.

- **Not a device experiment.** No circuit was run on IBM hardware. The calibration CSVs set the error rate of every
  fault location of a simulated circuit; correlated errors, leakage, crosstalk, drift between calibrations and
  measurement induced dephasing beyond T2 are not modelled.
- **No embedding.** IBM devices are heavy hexagonal; the rotated surface code needs a square grid. Code qubits and
  couplers are assigned device qubits and couplers by a placement rule, without SWAPs or flag qubits, so the noise is
  drawn from a device's measured distribution, not from running the code on that chip.
- **Model assumptions.** Gate errors from randomized benchmarking converted to depolarizing channels; idle noise as
  Pauli twirled T1 and T2 decay with no dynamical decoupling beyond the echo T2; preparation error taken as half of
  Prob meas1 prep0; readout as a symmetric flip with the average assignment error.
- **Snapshots, not live data.** The CSVs come from the snapshots shipped with qiskit-ibm-runtime 0.50.0 (2025-02-26
  for six devices, 2026-04 for Kingston and Aachen). Any CSV in the same layout can be used instead.
- **Pre-decoder trained on other noise.** NVIDIA's Fast model was trained on uniform circuit noise at d = 9. It is
  used as published; no model was trained on calibrated noise.
- **A stronger baseline exists.** PyMatching with correlations, which needs no network, beats pre-decoder +
  PyMatching at every d = 5 point. Other strong decoders (belief matching, Tesseract, correlated union find) were not
  run.
- **Decoding latency.** Card latency is measured per window through PCIe from the host; it is not a real time
  decoding loop and includes XRT overheads.
- **Teaching example.** `examples/small_codes.py` uses Stim's generated circuits with a simpler, homogeneous noise
  model; its numbers are not comparable with the study's.
