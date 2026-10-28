"""CPU throughput of the same network the card runs: NVIDIA's Fast model (float32, PyTorch eager) on calibration
derived windows, all host threads, for comparison with the card numbers of bench/run_card.py.

Author: Nasir Ali, C-DAC Noida.

Times the network forward pass only (no sampling, no input packing, no matching), batch --batch, best of --reps
after one warm up batch. Writes results/cpu/cpu_throughput.json.
usage: cpu_throughput.py [--csv data/calibration/ibm_kingston_2026-04-15.csv] [--d 3 5 7 9] [--threads 0]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calib.ibm_csv import read_csv  # noqa: E402
from calib.noise import calibrated  # noqa: E402
from harness.pipeline import Pipeline, load_fast  # noqa: E402


def cpu_name() -> str:
    for ln in Path("/proc/cpuinfo").read_text().splitlines():
        if ln.startswith("model name"):
            return ln.split(":", 1)[1].strip()
    return platform.processor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "calibration" / "ibm_kingston_2026-04-15.csv"))
    ap.add_argument("--d", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="0 = all logical CPUs")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads or os.cpu_count())
    cal = read_csv(a.csv)
    model = load_fast()
    rows = []
    for d in a.d:
        circ, _ = calibrated(cal, d, d, "X", "median", 0, 1.0)
        dets = circ.compile_detector_sampler(seed=1).sample(a.batch)
        x = Pipeline(d, d, "X", model).inputs(dets).float()
        with torch.no_grad():
            model(x)
            t = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                model(x)
                t.append(time.perf_counter() - t0)
        rows.append(dict(d=d, rounds=d, batch=a.batch, seconds_reps=t, windows_per_sec=a.batch / min(t)))
        print(d, f"{a.batch / min(t):,.0f} windows/s")
    out = dict(tier="T1-HW (CPU)", cpu=cpu_name(), logical_cpus=os.cpu_count(), threads=torch.get_num_threads(),
               torch=torch.__version__, precision="float32", csv=Path(a.csv).name, rows=rows,
               date=time.strftime("%Y-%m-%d %H:%M"))
    (ROOT / "results" / "cpu").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cpu" / "cpu_throughput.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
