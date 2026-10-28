"""Calibration derived circuit noise on NVIDIA's surface code memory circuit.

Author: Nasir Ali, C-DAC Noida.

The circuit is NVIDIA's `MemoryCircuit` (Ising-Decoding, Apache-2.0) with boundary detectors, the circuit their
pre-decoder was trained and evaluated on. It is generated with distinct placeholder probabilities, which mark every
fault location of NVIDIA's noise model (noise in every round except the last, which is perfect, and a noiseless
final data readout). Each location keeps its place and gets a probability from the calibration CSV of the device
qubit or coupler it is assigned to. Detector and observable definitions are untouched, so the network input, the
FPGA kernel and PyMatching's detector order are those of NVIDIA's pipeline.

Location classes and their probabilities (r: reported error per gate from randomized benchmarking):

| class | NVIDIA instruction | calibrated instruction |
|---|---|---|
| two qubit gate | DEPOLARIZE2 after CX | DEPOLARIZE2(5 r_2q / 4) on the pair, DEPOLARIZE1(3 r_sx) on the target (the two Hadamards that turn CZ or ECR into CX) |
| idle in a gate layer | DEPOLARIZE1 after CX | Pauli twirled amplitude and phase damping over the layer duration (longest coupler of the layer plus two sx) |
| idle in measurement | DEPOLARIZE1 on data after MR | same, over the longest ancilla readout |
| idle in reset | DEPOLARIZE1 on data at round start | same, over the longest ancilla reset |
| measurement | X_ERROR or Z_ERROR before MR or MRX | flip with the readout assignment error |
| preparation | X_ERROR or Z_ERROR after R, RX, MR, MRX | flip with half of Prob meas1 prep0 (equal split between reset and readout) |

Pauli twirl of amplitude damping (T1) and dephasing (T2 <= 2 T1) for a duration t:
p_x = p_y = (1 - exp(-t/T1)) / 4, p_z = (1 - exp(-t/T2)) / 2 - p_x.
Conversions of an average gate error r to depolarizing probability: 3r/2 (one qubit), 5r/4 (two qubits).

Placement. The rotated surface code needs a square grid; IBM devices are heavy hexagonal, so the code is not
embedded in the device graph (no SWAPs, no flag qubits). Instead each code qubit is assigned a distinct device qubit
and each code coupler a distinct device coupler (with replacement only if too few are available), in one of three
modes (see `place`): "median" (every location gets the device median), "best" (the lowest error qubits and
couplers) and "screened" (a seeded random draw after removing the worst 10% of every metric). The noise is therefore
drawn from the device's measured distribution, not a model of running the code on that chip.
"""
from __future__ import annotations

import math
import os
import statistics as st
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
REF = Path(os.environ.get("ISING_DECODING_DIR", HERE.parent / "third_party" / "Ising-Decoding")).resolve()
if str(REF / "code") not in sys.path:
    sys.path.insert(0, str(REF / "code"))

from calib.ibm_csv import Calibration  # noqa: E402

# placeholders: distinct values that mark NVIDIA's location classes
P_IDLE, P_TQ, P_SPAM = 1.1e-4, 1.3e-4, 1.7e-4
MEAS = {"M", "MR", "MX", "MRX", "MY", "MRY"}
RESET = {"R", "RX", "RY"}
MEAS_RESET = {"MR", "MRX", "MRY"}


def template(d: int, rounds: int, basis: str, rotation: str = "XV") -> tuple[stim.Circuit, object]:
    """NVIDIA's memory circuit with placeholder probabilities, flattened; and the MemoryCircuit object."""
    from qec.surface_code.memory_circuit import MemoryCircuit
    mc = MemoryCircuit(distance=d, idle_error=P_IDLE, sqgate_error=0.0, tqgate_error=P_TQ, spam_error=P_SPAM,
                       n_rounds=rounds, basis=basis, code_rotation=rotation, add_boundary_detectors=True)
    mc.set_error_rates()
    return mc.stim_circuit.flattened(), mc


@dataclass
class Placement:
    mode: str
    seed: int
    qubit: dict[int, int] = field(default_factory=dict)              # code qubit -> device qubit
    coupler: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)   # code pair -> device coupler


def code_pairs(c: stim.Circuit) -> list[tuple[int, int]]:
    pairs = set()
    for ins in c:
        if ins.name == "CX":
            t = [x.value for x in ins.targets_copy()]
            pairs.update((min(a, b), max(a, b)) for a, b in zip(t[::2], t[1::2]))
    return sorted(pairs)


SCREEN_QUANTILE = 0.9


def screened(cal: Calibration) -> tuple[list[int], list[tuple[int, int]]]:
    """Usable qubits and couplers after removing the worst 10% of each metric (readout error, sx error, Prob meas1
    prep0, and the shortest 10% of T1 and of T2; coupler error)."""
    uq, uc = cal.usable_qubits(), cal.usable_couplers()
    bad = set()
    for key, worse_high in (("readout_error", True), ("sx_error", True), ("p10", True), ("t1_us", False), ("t2_us", False)):
        v = np.array([getattr(cal, key)[q] for q in uq])
        cut = np.quantile(v, SCREEN_QUANTILE if worse_high else 1 - SCREEN_QUANTILE)
        bad |= {q for q, x in zip(uq, v) if (x > cut if worse_high else x < cut)}
    ce = np.array([cal.coupler_error[c] for c in uc])
    cut = np.quantile(ce, SCREEN_QUANTILE)
    return [q for q in uq if q not in bad], [c for c, e in zip(uc, ce) if e <= cut]


def qubit_score(cal: Calibration, q: int, t_ns: float) -> float:
    """Error of one round for ranking qubits: readout error plus idle error over t_ns."""
    t1, t2 = cal.t1_us[q] * 1e3, min(cal.t2_us[q], 2 * cal.t1_us[q]) * 1e3
    return cal.readout_error[q] + cal.p10[q] / 2 + (1 - math.exp(-t_ns / t2)) / 2 + (1 - math.exp(-t_ns / t1)) / 4


def place(cal: Calibration, n_qubits: int, pairs: list[tuple[int, int]], mode: str, seed: int) -> Placement:
    """median: every location gets the device median. best: the n lowest error usable qubits (readout, preparation
    and idle over one readout) and the lowest error couplers that pass `screened`, assigned at
    random among themselves. screened: a random draw among qubits and couplers that pass `screened`. Couplers are
    reused only when the pool is smaller than the code's coupler count."""
    pl = Placement(mode, seed)
    if mode == "median":
        return pl
    rng = np.random.default_rng(seed)
    if mode == "best":
        uc = screened(cal)[1]
        uq = cal.usable_qubits()
        t = st.median(cal.readout_ns[q] for q in uq)
        uq = sorted(uq, key=lambda q: qubit_score(cal, q, t))[:n_qubits]
        uc = sorted(uc, key=lambda c: cal.coupler_error[c])[:len(pairs)]
    elif mode == "screened":
        uq, uc = screened(cal)
    else:
        raise ValueError(mode)
    if len(uq) < n_qubits:
        raise ValueError(f"{cal.device}: {len(uq)} qubits available for mode {mode}, the code needs {n_qubits}")
    qs = rng.choice(len(uq), n_qubits, replace=False)
    pl.qubit = {i: uq[k] for i, k in enumerate(qs)}
    cs = rng.choice(len(uc), len(pairs), replace=len(uc) < len(pairs))
    pl.coupler = {p: uc[k] for p, k in zip(pairs, cs)}
    return pl


class Params:
    """Per code qubit and per code coupler values for one placement (medians for "median")."""

    def __init__(self, cal: Calibration, pl: Placement, scale: float = 1.0):
        self.cal, self.pl, self.scale = cal, pl, scale
        uq, uc = cal.usable_qubits(), cal.usable_couplers()
        self.med = {k: st.median(getattr(cal, k)[q] for q in uq)
                    for k in ("t1_us", "t2_us", "readout_error", "p10", "readout_ns", "sx_error", "sx_ns", "reset_ns")}
        self.med["coupler_error"] = st.median(cal.coupler_error[c] for c in uc)
        self.med["coupler_ns"] = st.median(cal.coupler_ns[c] for c in uc)

    def q(self, key: str, i: int) -> float:
        if self.pl.mode == "median":
            return self.med[key]
        return getattr(self.cal, key)[self.pl.qubit[i]]

    def c(self, key: str, pair: tuple[int, int]) -> float:
        if self.pl.mode == "median":
            return self.med[key]
        return getattr(self.cal, key)[self.pl.coupler[(min(pair), max(pair))]]

    def idle(self, i: int, t_ns: float) -> tuple[float, float, float]:
        t1 = self.q("t1_us", i) * 1e3
        t2 = min(self.q("t2_us", i), 2 * self.q("t1_us", i)) * 1e3
        x = t_ns * self.scale      # scaling the error rates scales the decay rates 1/T1 and 1/T2
        px = (1 - math.exp(-x / t1)) / 4
        pz = max(0.0, (1 - math.exp(-x / t2)) / 2 - px)
        return px, px, pz

    def p(self, v: float) -> float:
        return min(0.5, v * self.scale)


def _targets(ins) -> list[int]:
    return [t.value for t in ins.targets_copy()]


def calibrated(cal: Calibration, d: int, rounds: int, basis: str, mode: str = "screened", seed: int = 0,
               scale: float = 1.0) -> tuple[stim.Circuit, dict]:
    """NVIDIA's circuit with every fault location's probability taken from the calibration. Returns the circuit and
    a record of the placement and of the per class probabilities used (for the ledger)."""
    tpl, mc = template(d, rounds, basis)
    pairs = code_pairs(tpl)
    pl = place(cal, tpl.num_qubits, pairs, mode, seed)
    pr = Params(cal, pl, scale)
    out = stim.Circuit()
    ins_list = list(tpl)
    noise = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR"}
    anc = sorted({q for x in ins_list if x.name in MEAS_RESET for q in _targets(x)})
    t_meas = max(pr.q("readout_ns", i) for i in anc)     # a round's readout lasts as long as its slowest ancilla
    t_reset = max(pr.q("reset_ns", i) for i in anc)
    prev_op = None            # last instruction that is neither noise nor annotation
    seen_det_since_meas = False
    layer_ns = 0.0
    used = {"two_qubit": [], "idle": [], "readout": [], "prep": []}
    for k, ins in enumerate(ins_list):
        name = ins.name
        if name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"):
            seen_det_since_meas |= name == "DETECTOR"
            out.append(ins)
            continue
        if name not in noise:
            if name == "CX":
                tg = _targets(ins)
                layer_ns = max(pr.c("coupler_ns", (a, b)) + 2 * pr.q("sx_ns", b) for a, b in zip(tg[::2], tg[1::2]))
            if name in MEAS:
                seen_det_since_meas = False
            prev_op = ins
            out.append(ins)
            continue
        ph, tg = ins.gate_args_copy()[0], _targets(ins)
        if name == "DEPOLARIZE2":
            assert math.isclose(ph, P_TQ) and prev_op.name == "CX"
            for a, b in zip(tg[::2], tg[1::2]):
                p2 = pr.p(5 / 4 * pr.c("coupler_error", (a, b)))
                out.append("DEPOLARIZE2", [a, b], p2)
                out.append("DEPOLARIZE1", [b], pr.p(3 * pr.q("sx_error", b)))
                used["two_qubit"].append(p2)
        elif name == "DEPOLARIZE1":
            assert math.isclose(ph, P_IDLE), (name, ph)
            if prev_op.name == "CX":
                t_ns = layer_ns
            elif prev_op.name in MEAS and not seen_det_since_meas:
                t_ns = t_meas
            else:
                t_ns = t_reset
            for i in tg:
                px, py, pz = pr.idle(i, t_ns)
                out.append("PAULI_CHANNEL_1", [i], [px, py, pz])
                used["idle"].append(px + py + pz)
        else:
            # a flip is a measurement error when the next operation measures these qubits, else a preparation error
            assert math.isclose(ph, P_SPAM), (name, ph)
            nxt = next((x for x in ins_list[k + 1:] if x.name not in noise), None)
            is_meas = nxt is not None and nxt.name in MEAS and set(tg) <= set(_targets(nxt))
            for i in tg:
                v = pr.p(pr.q("readout_error", i)) if is_meas else pr.p(pr.q("p10", i) / 2)
                out.append(name, [i], v)
                used["readout" if is_meas else "prep"].append(v)
    info = dict(device=cal.device, snapshot=cal.snapshot, d=d, rounds=rounds, basis=basis, mode=mode, seed=seed,
                scale=scale, qubits=tpl.num_qubits, code_couplers=len(pairs),
                placement_qubits={str(k): v for k, v in pl.qubit.items()},
                placement_couplers={f"{a}_{b}": f"{c[0]}_{c[1]}" for (a, b), c in pl.coupler.items()},
                mean_probability={k: float(np.mean(v)) if v else 0.0 for k, v in used.items()},
                locations={k: len(v) for k, v in used.items()})
    return out, info


def nominal(d: int, rounds: int, basis: str, p: float) -> stim.Circuit:
    """NVIDIA's own circuit noise at physical error rate p (the 25 parameter model with its public relations), for
    decoders that ignore the calibration."""
    from qec.noise_model import NoiseModel  # noqa: F401  (imported for the version check of the reference)
    from qec.surface_code.memory_circuit import MemoryCircuit
    mc = MemoryCircuit(distance=d, idle_error=p, sqgate_error=p, tqgate_error=p, spam_error=2 * p / 3, n_rounds=rounds,
                       basis=basis, code_rotation="XV", add_boundary_detectors=True)
    mc.set_error_rates()
    return mc.stim_circuit
