"""Teaching example: memory experiments at d = 1, 2, 3, 5 with noise from a calibration CSV, on a laptop CPU.

Author: Nasir Ali, C-DAC Noida.

  d = 1  one unencoded qubit prepared, left idle for as long as `rounds` error correction rounds take, measured;
         nothing can be corrected, so its error rate is the reference every code must beat.
  d = 2  Stim's rotated surface code of distance 2: it detects any single error but cannot tell which one, so the
         shots with a detection event are discarded (post-selection); reported: the fraction discarded and the logical
         error rate of the shots kept, plus the error rate without any decoding.
  d >= 3 Stim's rotated surface code memory decoded by PyMatching.

Noise: every location gets the device median from the CSV (calib/ibm_csv.py), layer by layer:
  after a reset: flip with half of Prob meas1 prep0; before a measurement: flip with the readout error;
  after a two qubit gate: DEPOLARIZE2(5/4 x CZ or ECR error); after a one qubit gate: DEPOLARIZE1(3/2 x sx error);
  every qubit idle in a layer: Pauli twirled T1 and T2 decay over the layer's duration
  (two qubit gate time, sx time, readout time or reset time).
This is a simpler model than calib/noise.py (which keeps NVIDIA's fault locations for the pre-decoder study) and its
numbers are not compared with that study's.

usage: python examples/small_codes.py data/calibration/ibm_kingston_2026-04-15.csv [--d 1 2 3 5] [--rounds 5]
       [--shots 100000] [--scale 1.0] [--basis Z]
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import pymatching
import stim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib.ibm_csv import read_csv  # noqa: E402

ONE_Q = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG", "X", "Y", "Z"}
TWO_Q = {"CX", "CZ", "CNOT"}
MEAS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
RESET = {"R", "RX", "RY"}


class Median:
    def __init__(self, csv: str, scale: float):
        cal = read_csv(csv)
        q, c = cal.usable_qubits(), cal.usable_couplers()
        self.device, self.scale = cal.device, scale
        md = lambda d, keys: st.median(d[k] for k in keys)  # noqa: E731
        self.t1, self.t2 = md(cal.t1_us, q) * 1e3, min(md(cal.t2_us, q), 2 * md(cal.t1_us, q)) * 1e3
        self.ro, self.prep = md(cal.readout_error, q), md(cal.p10, q) / 2
        self.sx, self.sx_ns = md(cal.sx_error, q), md(cal.sx_ns, q)
        self.e2, self.t2q = md(cal.coupler_error, c), md(cal.coupler_ns, c)
        self.ro_ns, self.reset_ns = md(cal.readout_ns, q), md(cal.reset_ns, q)

    def p(self, v):
        return min(0.5, v * self.scale)

    def idle(self, t_ns):
        px = (1 - math.exp(-t_ns * self.scale / self.t1)) / 4
        pz = max(0.0, (1 - math.exp(-t_ns * self.scale / self.t2)) / 2 - px)
        return [px, px, pz]


def add_noise(clean: stim.Circuit, n: Median) -> stim.Circuit:
    """Noise layer by layer (layers separated by TICK) on a noiseless, flattened circuit."""
    out = stim.Circuit()
    layer: list[stim.CircuitInstruction] = []
    nq = clean.num_qubits

    def flush():
        busy, dur = set(), 0.0
        for ins in layer:
            t = [x.value for x in ins.targets_copy() if x.is_qubit_target]
            if ins.name in RESET:
                out.append(ins)
                out.append("X_ERROR" if ins.name == "R" else "Z_ERROR", t, n.p(n.prep))
                dur = max(dur, n.reset_ns)
            elif ins.name in MEAS:
                out.append("X_ERROR" if ins.name in ("M", "MR") else "Z_ERROR", t, n.p(n.ro))
                out.append(ins)
                if ins.name in ("MR", "MRX", "MRY"):
                    out.append("X_ERROR" if ins.name == "MR" else "Z_ERROR", t, n.p(n.prep))
                dur = max(dur, n.ro_ns + (n.reset_ns if ins.name.startswith("MR") else 0))
            elif ins.name in TWO_Q:
                out.append(ins)
                out.append("DEPOLARIZE2", t, n.p(1.25 * n.e2))
                dur = max(dur, n.t2q)
            elif ins.name in ONE_Q:
                out.append(ins)
                out.append("DEPOLARIZE1", t, n.p(1.5 * n.sx))
                dur = max(dur, n.sx_ns)
            else:
                out.append(ins)
                continue
            busy |= set(t)
        idle = [q for q in range(nq) if q not in busy]
        if idle and dur > 0:
            out.append("PAULI_CHANNEL_1", idle, n.idle(dur))
        layer.clear()

    for ins in clean.flattened():
        if ins.name == "TICK":
            flush()
            out.append(ins)
        elif ins.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            flush()
            out.append(ins)
        else:
            layer.append(ins)
    flush()
    return out


def round_ns(n: Median) -> float:
    """Duration of one rotated surface code round in Stim's circuit: reset, H, 4 CX layers, H, measurement."""
    return n.reset_ns + 2 * n.sx_ns + 4 * n.t2q + n.ro_ns


def unencoded(n: Median, rounds: int, basis: str, shots: int, seed: int) -> dict:
    c = stim.Circuit()
    c.append("R" if basis == "Z" else "RX", [0])
    c.append("X_ERROR" if basis == "Z" else "Z_ERROR", [0], n.p(n.prep))
    c.append("PAULI_CHANNEL_1", [0], n.idle(rounds * round_ns(n)))
    c.append("X_ERROR" if basis == "Z" else "Z_ERROR", [0], n.p(n.ro))
    c.append("M" if basis == "Z" else "MX", [0])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    _, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    k = int(obs.sum())
    return dict(d=1, rounds=rounds, shots=shots, logical_errors=k, ler=k / shots)


def surface(n: Median, d: int, rounds: int, basis: str, shots: int, seed: int) -> dict:
    clean = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=d, rounds=rounds)
    c = add_noise(clean, n)
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    if d == 2:
        keep = ~dets.any(axis=1)
        kept = int(keep.sum())
        k = int(obs[keep, 0].sum())
        return dict(d=2, rounds=rounds, shots=shots, discarded_fraction=1 - kept / shots, kept=kept,
                    logical_errors_kept=k, ler_kept=k / max(kept, 1), ler_no_decoding=float(obs[:, 0].mean()))
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    k = int((m.decode_batch(dets)[:, 0] != obs[:, 0]).sum())
    return dict(d=d, rounds=rounds, shots=shots, logical_errors=k, ler=k / shots, detector_density=float(dets.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--d", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every error rate (0.5 = a device twice as good)")
    ap.add_argument("--basis", default="Z", choices=["X", "Z"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default="", help="also write the results to this JSON file")
    a = ap.parse_args()
    n = Median(a.csv, a.scale)
    print(f"{n.device}: median 2q error {n.e2:.2e}, readout {n.ro:.2e}, T1 {n.t1 / 1e3:.0f} us, T2 {n.t2 / 1e3:.0f} us, "
          f"one round {round_ns(n) / 1e3:.2f} us, scale {a.scale}")
    rows = []
    for d in a.d:
        r = unencoded(n, a.rounds, a.basis, a.shots, a.seed) if d == 1 else surface(n, d, a.rounds, a.basis, a.shots, a.seed)
        rows.append(r)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    if a.json:
        import json
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(device=n.device, csv=Path(a.csv).name, scale=a.scale, basis=a.basis,
                                                rounds=a.rounds, seed=a.seed, results=rows), indent=1))


if __name__ == "__main__":
    main()
