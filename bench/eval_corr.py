"""Correlated matching on the test shots of bench/eval_ler.py: PyMatching with enable_correlations (pm_corr) alone and
after the float32 Fast pre-decoder (fast_fp32_corr). Same circuit, same sampling seed, same shots.

Author: Nasir Ali, C-DAC Noida.

Writes results/ler/<name>.corr.json and .corr.npz next to the eval_ler.py record of the same point.
usage: eval_corr.py --csv ... --mode ... --scale ... --d ... --basis ... [--shots 200000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pymatching

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.eval_ler import SAMPLE_SEED, wilson  # noqa: E402
from calib.ibm_csv import read_csv  # noqa: E402
from calib.noise import calibrated  # noqa: E402
from harness.pipeline import Pipeline, load_fast  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mode", default="median")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--basis", default="X")
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    cal = read_csv(a.csv)
    t0 = time.time()
    circ, info = calibrated(cal, a.d, a.d, a.basis, a.mode, a.seed, a.scale)
    seed = SAMPLE_SEED + 1000 * a.d + (0 if a.basis == "X" else 1)
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(a.shots, separate_observables=True)
    dem = circ.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    o = obs[:, 0]
    pre, res = Pipeline(a.d, a.d, a.basis, load_fast()).run(dets)
    f = {"pm_corr": m.decode_batch(dets, enable_correlations=True)[:, 0] != o,
         "fast_fp32_corr": (m.decode_batch(res, enable_correlations=True)[:, 0] ^ pre.astype(np.uint8)) != o}
    rec = dict(tier="T4-SIM", csv=Path(a.csv).name, calibration=info, rounds=a.d, shots=a.shots, sample_seed=seed,
               decoders={k: dict(fails=int(v.sum()), ler=float(v.mean()), ler_wilson95=wilson(int(v.sum()), a.shots))
                         for k, v in f.items()}, seconds=round(time.time() - t0, 1))
    out = ROOT / "results" / "ler" / f"{cal.device}_{a.mode}{a.seed}_s{a.scale:g}_d{a.d}_r{a.d}_{a.basis}"
    Path(f"{out}.corr.json").write_text(json.dumps(rec, indent=1))
    np.savez_compressed(f"{out}.corr.npz", **{k: np.packbits(v) for k, v in f.items()})
    print({k: v["fails"] for k, v in rec["decoders"].items()}, rec["seconds"])


if __name__ == "__main__":
    main()
