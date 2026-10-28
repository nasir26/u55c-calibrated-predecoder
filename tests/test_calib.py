"""Calibration CSV, calibrated circuits and the decoding pipeline.

Author: Nasir Ali, C-DAC Noida.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pytest
import stim
import torch

from calib.ibm_csv import read_csv, same, write_csv
from calib.noise import P_IDLE, P_SPAM, P_TQ, calibrated, screened, template

ROOT = Path(__file__).resolve().parents[1]
CSVS = sorted(glob.glob(str(ROOT / "data" / "calibration" / "*.csv")))
KINGSTON = next(c for c in CSVS if "kingston" in c)
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR", "PAULI_CHANNEL_1"}


@pytest.mark.parametrize("csv", CSVS)
def test_csv_round_trip(csv, tmp_path):
    cal = read_csv(csv)
    write_csv(cal, tmp_path / "x.csv")
    assert same(cal, read_csv(tmp_path / "x.csv"))
    assert len(cal.usable_qubits()) >= 120 and len(cal.usable_couplers()) >= 120


def test_reader_accepts_header_variants(tmp_path):
    src = Path(KINGSTON).read_text(encoding="utf-8").splitlines()
    hdr = src[1].split(",")
    # trailing spaces, other capitalisation, no reset column, no comment line
    hdr = [h.upper() + " " for h in hdr]
    keep = [i for i, h in enumerate(hdr) if "RESET" not in h]
    rows = [",".join(hdr[i] for i in keep)]
    import csv as _csv
    for r in _csv.reader(src[2:]):
        rows.append(",".join('"' + r[i] + '"' for i in keep))
    p = tmp_path / "v.csv"
    p.write_text("\n".join(rows), encoding="utf-8")
    a, b = read_csv(KINGSTON), read_csv(p)
    assert a.coupler_error == b.coupler_error
    assert all(x == y or (math.isnan(x) and math.isnan(y)) for x, y in zip(a.t1_us.values(), b.t1_us.values()))
    assert all(b.reset_ns[q] == b.readout_ns[q] for q in b.qubits)       # missing reset length -> readout length


def _skeleton(c: stim.Circuit) -> list[str]:
    return [str(i) for i in c.flattened() if i.name not in NOISE]


@pytest.mark.parametrize("d,basis,mode", [(3, "X", "median"), (5, "Z", "best"), (5, "X", "screened")])
def test_calibrated_circuit_keeps_nvidia_structure(d, basis, mode):
    """Same operations, detectors and observable as NVIDIA's circuit; one calibrated channel per fault location."""
    cal = read_csv(KINGSTON)
    tpl, _ = template(d, d, basis)
    c, info = calibrated(cal, d, d, basis, mode, 3)
    assert _skeleton(c) == _skeleton(tpl)
    n_tq = sum(len(i.targets_copy()) // 2 for i in tpl if i.name == "DEPOLARIZE2")
    n_idle = sum(len(i.targets_copy()) for i in tpl if i.name == "DEPOLARIZE1")
    n_spam = sum(len(i.targets_copy()) for i in tpl if i.name in ("X_ERROR", "Z_ERROR"))
    assert info["locations"]["two_qubit"] == n_tq
    assert info["locations"]["idle"] == n_idle
    assert info["locations"]["readout"] + info["locations"]["prep"] == n_spam
    # placeholders are gone
    for i in c:
        if i.name in NOISE:
            assert not any(math.isclose(p, x) for p in i.gate_args_copy() for x in (P_IDLE, P_TQ, P_SPAM))


def test_last_round_is_noiseless():
    """NVIDIA's convention: no noise after the last noisy round's measurement idle."""
    cal = read_csv(KINGSTON)
    c, _ = calibrated(cal, 5, 5, "X", "median", 0)
    ins = list(c)
    meas = [k for k, i in enumerate(ins) if i.name in ("MR", "MRX")]
    last_round_start = meas[-3] + 1           # after the MRX and MR of round T - 1
    tail = [i for i in ins[last_round_start:] if i.name in NOISE]
    assert all(i.name == "PAULI_CHANNEL_1" for i in tail)      # only the measurement idle of round T - 1
    after = [i for i in ins[meas[-2]:] if i.name in NOISE]
    assert not after


def test_scale_zero_is_noiseless():
    cal = read_csv(KINGSTON)
    c, _ = calibrated(cal, 3, 3, "Z", "screened", 1, scale=0.0)
    dets, obs = c.compile_detector_sampler(seed=1).sample(2000, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_measurement_and_preparation_flips_are_told_apart():
    cal = read_csv(KINGSTON)
    c, info = calibrated(cal, 3, 3, "X", "median", 0)
    med_ro = float(np.median([cal.readout_error[q] for q in cal.usable_qubits()]))
    med_prep = float(np.median([cal.p10[q] for q in cal.usable_qubits()])) / 2
    assert math.isclose(info["mean_probability"]["readout"], med_ro)
    assert math.isclose(info["mean_probability"]["prep"], med_prep)
    # every flip directly before a measurement of its qubit carries the readout error
    ins = list(c)
    for k, i in enumerate(ins):
        if i.name in ("X_ERROR", "Z_ERROR"):
            nxt = next(x for x in ins[k + 1:] if x.name not in NOISE)
            if nxt.name in ("MR", "MRX") and set(t.value for t in i.targets_copy()) <= set(t.value for t in nxt.targets_copy()):
                assert math.isclose(i.gate_args_copy()[0], med_ro)


def test_screened_pool_and_placement_are_distinct_qubits():
    cal = read_csv(KINGSTON)
    q, c = screened(cal)
    assert 0 < len(q) < len(cal.usable_qubits()) and 0 < len(c) < len(cal.usable_couplers())
    _, info = calibrated(cal, 5, 5, "X", "screened", 4)
    v = list(info["placement_qubits"].values())
    assert len(set(v)) == len(v) and set(v) <= set(q)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_pipeline_without_corrections_is_pymatching(basis):
    """With no pre-decoder corrections the residual loses only the last round's basis layer, which PyMatching does
    not need: failures equal PyMatching on the raw detectors."""
    from harness.pipeline import Pipeline, fails, matcher
    cal = read_csv(KINGSTON)
    c, _ = calibrated(cal, 3, 3, basis, "median", 0)
    dets, obs = c.compile_detector_sampler(seed=5).sample(20000, separate_observables=True)
    pre, res = Pipeline(3, 3, basis).run(dets, torch.full((len(dets), 4, 3, 3, 3), -1.0))
    m = matcher(c)
    assert not pre.any()
    assert np.array_equal(fails(m, res, obs, pre), fails(m, dets, obs))


def test_small_codes_example_runs():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "small_codes.py"), KINGSTON, "--d", "1", "2", "3",
                        "--shots", "2000"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "'d': 1" in r.stdout and "discarded_fraction" in r.stdout and "'d': 3" in r.stdout
