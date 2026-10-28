"""Logical error rate on a calibration derived circuit: PyMatching alone and after NVIDIA's Fast pre-decoder (float32
and the INT8 integer model that the FPGA kernel computes bit for bit), all on the same test shots.

Author: Nasir Ali, C-DAC Noida.

Decoders (every one decodes the same shots of the calibrated circuit):
  pm_cal       PyMatching with the matching graph of the calibrated circuit (knows every qubit's error rates)
  pm_median    PyMatching with the graph of the same circuit at the device medians (knows only the averages)
  fast_fp32    NVIDIA Fast pre-decoder (float32) then pm_cal on the residual
  fast_int8    integer model (model/qmodel.py) then pm_cal; its decisions are what the FPGA computes
The integer model is calibrated on windows of separately drawn shots (seed + 1), never on test shots.
Per shot failure bits are stored (packed) for exact paired comparisons.

usage: eval_ler.py --csv data/calibration/ibm_kingston_2026-04-15.csv --mode median --scale 0.5 --d 5
                   [--basis X] [--shots 200000] [--seed 0] [--decoders pm_cal,pm_median,fast_fp32,fast_int8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calib.ibm_csv import read_csv  # noqa: E402
from calib.noise import calibrated  # noqa: E402
from harness.pipeline import Pipeline, fails, load_fast, matcher  # noqa: E402

SAMPLE_SEED = 1_000_003


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", default="median", choices=["median", "best", "screened"])
    ap.add_argument("--seed", type=int, default=0, help="placement seed")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=0, help="default: d")
    ap.add_argument("--basis", default="X", choices=["X", "Z"])
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--cal-windows", type=int, default=1024)
    ap.add_argument("--decoders", default="pm_cal,pm_median,fast_fp32,fast_int8")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    rounds = a.rounds or a.d
    cal = read_csv(a.csv)
    dec = a.decoders.split(",")
    t0 = time.time()
    circ, info = calibrated(cal, a.d, rounds, a.basis, a.mode, a.seed, a.scale)
    med, _ = calibrated(cal, a.d, rounds, a.basis, "median", 0, a.scale)
    seed = SAMPLE_SEED + 1000 * a.d + (0 if a.basis == "X" else 1)
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(a.shots, separate_observables=True)
    m = matcher(circ)
    rec = dict(tier="T4-SIM", csv=Path(a.csv).name, calibration=info, rounds=rounds, shots=a.shots, sample_seed=seed,
               detector_density=float(dets.mean()), decoders={})
    bits = {}

    def put(name, f, extra=None):
        k = int(f.sum())
        rec["decoders"][name] = dict(fails=k, ler=k / a.shots, ler_wilson95=wilson(k, a.shots), **(extra or {}))
        bits[name] = np.packbits(f)
        print(f"{name:10s} {k:7d} {k / a.shots:.3e}", flush=True)

    if "pm_cal" in dec:
        put("pm_cal", fails(m, dets, obs))
    if "pm_median" in dec:
        put("pm_median", fails(matcher(med), dets, obs))
    if "fast_fp32" in dec or "fast_int8" in dec:
        model = load_fast()
        if "fast_fp32" in dec:
            t1 = time.time()
            pre, res = Pipeline(a.d, rounds, a.basis, model).run(dets)
            put("fast_fp32", fails(m, res, obs, pre), dict(residual_density=float(res.mean()),
                                                          seconds=round(time.time() - t1, 1)))
        if "fast_int8" in dec:
            sys.path.insert(0, str(ROOT / "model"))
            from qmodel import calibrate
            cd, _ = circ.compile_detector_sampler(seed=seed + 7).sample(a.cal_windows, separate_observables=True)
            x_cal = Pipeline(a.d, rounds, a.basis).inputs(cd)
            qm = calibrate(model, x_cal)
            t1 = time.time()
            pre, res = Pipeline(a.d, rounds, a.basis, qm).run(dets)
            put("fast_int8", fails(m, res, obs, pre), dict(residual_density=float(res.mean()),
                                                          seconds=round(time.time() - t1, 1),
                                                          calibration_windows=a.cal_windows))
    rec["seconds"] = round(time.time() - t0, 1)
    out = Path(a.out or ROOT / "results" / "ler" / (f"{cal.device}_{a.mode}{a.seed}_s{a.scale:g}_d{a.d}_r{rounds}_{a.basis}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    # names contain the scale (s0.5), so the suffix is appended, never substituted
    Path(f"{out}.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(f"{out}.npz", **bits)
    print("wrote", f"{out}.json")


if __name__ == "__main__":
    main()
