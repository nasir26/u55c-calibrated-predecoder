"""One U55C card: NVIDIA's Fast pre-decoder (INT8 integer model) on calibration derived syndrome windows.

Author: Nasir Ali, C-DAC Noida.

For every configuration (calibration CSV, placement, error scale, distance, basis):
  1. the test shots of bench/eval_ler.py (same circuit, same sampling seed) and the integer model calibrated on the
     same separately drawn windows;
  2. every test window through the card (both kernel instances), the decision bits read back;
  3. bit exact check of both instances against the integer model (model/qmodel.py) on the first --check windows;
  4. logical error rate from the card's decisions (residual through NVIDIA's evaluation module, then PyMatching
     with the calibrated graph), compared shot by shot with the CPU integer model of bench/eval_ler.py when its
     .npz exists.
Then throughput (windows per second, both instances, two launches in flight) and latency (one window, one launch
in flight) at every distance in --perf-d, on windows of the first configuration's device.

usage: run_card.py --xclbin build/hw/predecoder.xclbin --device 2 --config CSV:MODE:SCALE:D:BASIS [...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model"), str(ROOT / "host")]

from bench.eval_ler import SAMPLE_SEED, wilson  # noqa: E402
from calib.ibm_csv import read_csv  # noqa: E402
from calib.noise import calibrated  # noqa: E402
from harness.pipeline import Pipeline, bits_to_logits, fails, load_fast, matcher  # noqa: E402
from pack import pack_model, pack_windows, unpack_bits  # noqa: E402
from qmodel import calibrate  # noqa: E402

HOST = Path(os.environ.get("NPD_HOST", ROOT / "runs" / "bin" / "xrt_predecoder"))


def host(xclbin, files, d, rounds, windows, launches, dev, dump=None, inflight=2):
    c = [str(HOST), str(xclbin), str(files["w"]), str(files["p"]), str(files["x"]), str(d), str(rounds), str(windows),
         str(launches), str(dev)] + ([str(dump)] if dump else [])
    r = subprocess.run(c, capture_output=True, text=True, env=dict(os.environ, NPD_INFLIGHT=str(inflight)))
    if r.returncode:
        raise RuntimeError(r.stderr)
    n_inst = int(re.search(r"instances (\d+)", r.stderr).group(1))
    m = re.search(r"window_ns (\d+) (\d+)", r.stderr)
    return n_inst, float(r.stdout.strip()), (int(m.group(2)) - int(m.group(1))) * 1e-9


def prepare(cal, mode, scale, d, basis, shots, cal_windows, model):
    circ, info = calibrated(cal, d, d, basis, mode, 0, scale)
    seed = SAMPLE_SEED + 1000 * d + (0 if basis == "X" else 1)
    dets, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    cd, _ = circ.compile_detector_sampler(seed=seed + 7).sample(cal_windows, separate_observables=True)
    pl = Pipeline(d, d, basis)
    qm = calibrate(model, pl.inputs(cd))
    return circ, info, dets, obs, qm, pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--device", type=int, default=2)
    ap.add_argument("--config", nargs="+", required=True, help="CSV:MODE:SCALE:D:BASIS")
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--check", type=int, default=4096)
    ap.add_argument("--cal-windows", type=int, default=1024)
    ap.add_argument("--perf-d", default="3,5,7,9")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "card" / f"card_{time.strftime('%Y%m%d_%H%M')}.json"))
    a = ap.parse_args()
    xclbin = Path(a.xclbin).resolve()
    sha = subprocess.run(["sha256sum", str(xclbin)], capture_output=True, text=True).stdout.split()[0]
    rec = dict(tier="T1-HW", start=time.strftime("%Y-%m-%d %H:%M:%S"), xclbin=xclbin.name, xclbin_sha256=sha,
               device=a.device, configs=[], perf=[])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    model = load_fast()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        first = None
        for spec in a.config:
            csv, mode, scale, d, basis = spec.split(":")
            scale, d = float(scale), int(d)
            cal = read_csv(csv)
            circ, info, dets, obs, qm, pl = prepare(cal, mode, scale, d, basis, a.shots, a.cal_windows, model)
            w, p = pack_model(qm)
            files = dict(w=td / "w.bin", p=td / "p.bin", x=td / "x.bin")
            w.tofile(files["w"])
            p.tofile(files["p"])
            first = first or (cal, mode, scale, basis, model)
            bits = np.zeros((len(dets), 4, d, d, d), dtype=np.uint8)
            exact = True
            t0 = time.time()
            for s in range(0, len(dets), a.chunk):
                xi = np.rint(pl.inputs(dets[s:s + a.chunk]).numpy() * 2).astype(np.int64)
                pack_windows(xi).tofile(files["x"])
                n_inst, _, _ = host(xclbin, files, d, d, len(xi), 1, a.device, td / "bits.bin")
                b = np.fromfile(td / "bits.bin", dtype=np.uint8).reshape(n_inst, len(xi), -1)
                got = [unpack_bits(b[k], d, d) for k in range(n_inst)]
                exact &= all(np.array_equal(g, got[0]) for g in got)       # instances agree on every window
                if s < a.check:
                    n = min(a.check - s, len(xi))
                    want, _ = qm.forward_int_fast(xi[:n])
                    exact &= np.array_equal(got[0][:n], want)
                bits[s:s + len(xi)] = got[0]
            card_s = time.time() - t0
            pre, res = Pipeline(d, d, basis).run(dets, bits_to_logits(bits))
            m = matcher(circ)
            f_card, f_pm = fails(m, res, obs, pre), fails(m, dets, obs)
            row = dict(csv=Path(csv).name, device_name=cal.device, mode=mode, scale=scale, d=d, rounds=d, basis=basis,
                       shots=len(dets), instances=n_inst, bit_exact_checked_windows=min(a.check, len(dets)),
                       bit_exact=bool(exact), wall_s_incl_host=round(card_s, 1),
                       card_int8=dict(fails=int(f_card.sum()), ler_wilson95=wilson(int(f_card.sum()), len(dets))),
                       pm_cal=dict(fails=int(f_pm.sum()), ler_wilson95=wilson(int(f_pm.sum()), len(dets))))
            npz = ROOT / "results" / "ler" / f"{cal.device}_{mode}0_s{scale:g}_d{d}_r{d}_{basis}.npz"
            if npz.exists():
                z = np.load(npz)
                if "fast_int8" in z:
                    cpu = np.unpackbits(z["fast_int8"])[:len(dets)].astype(bool)
                    row["same_failures_as_cpu_int8"] = bool(np.array_equal(cpu, f_card))
            tag = f"{cal.device}_{mode}0_s{scale:g}_d{d}_r{d}_{basis}"
            np.savez_compressed(Path(a.out).parent / f"{tag}.npz", card_int8=np.packbits(f_card), pm_cal=np.packbits(f_pm))
            row["failure_bits"] = f"{tag}.npz"
            rec["configs"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(rec, open(a.out, "w"), indent=1)
            if not exact:
                raise SystemExit("bit exact check failed")

        cal, mode, scale, basis, model = first
        for d in (int(x) for x in a.perf_d.split(",")):
            circ, info, dets, obs, qm, pl = prepare(cal, mode, scale, d, basis, 4096, a.cal_windows, model)
            w, p = pack_model(qm)
            files = dict(w=td / "w.bin", p=td / "p.bin", x=td / "x.bin")
            w.tofile(files["w"])
            p.tofile(files["p"])
            pack_windows(np.rint(pl.inputs(dets).numpy() * 2).astype(np.int64)).tofile(files["x"])
            n_inst, rate, _ = host(xclbin, files, d, d, 64, 4, a.device)
            launches = max(4, int(a.seconds * rate / (64 * n_inst)))
            reps = [host(xclbin, files, d, d, 64, launches, a.device)[1] for _ in range(3)]
            _, _, sec = host(xclbin, files, d, d, 1, 200, a.device, inflight=1)
            row = dict(d=d, rounds=d, sites=d ** 3, instances=n_inst, windows_per_launch=64, launches=launches,
                       windows_per_sec_reps=reps, latency_one_window_s=sec / 200)
            rec["perf"].append(row)
            print(json.dumps(row), flush=True)
            json.dump(rec, open(a.out, "w"), indent=1)
    rec["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
