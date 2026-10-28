"""Device calibration as CSV: export from IBM backend snapshots, and a reader for any CSV in the same layout.

Author: Nasir Ali, C-DAC Noida.

Layout (one row per qubit), following the column names of the IBM Quantum calibration CSV:

    Qubit, T1 (us), T2 (us), Readout assignment error, Prob meas0 prep1, Prob meas1 prep0,
    Readout length (ns), √x (sx) error, Single-qubit gate length (ns), <2Q> error, Gate length (ns),
    Reset length (ns), Operational

where <2Q> is CZ or ECR, and the two qubit columns hold "a_b:value" entries separated by ";" for the couplers of
that qubit. "Reset length (ns)" is an addition of this project (the reset duration enters the idle noise); when it is
absent the reader uses the readout length.

The reader matches columns by case-insensitive keywords, so CSVs with extra columns or slightly different headers
(for example "Readout assignment error " with a trailing space) load unchanged. A coupler listed under both of its
qubits with different values (directional gates) gets the mean.

Export: python calib/ibm_csv.py export --device fez --out data/calibration
(needs qiskit-ibm-runtime, which ships the snapshots under fake_provider/backends/<name>/props_<name>.json,
Apache License 2.0; no IBM account is used.)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

TWO_Q = ("cz", "ecr", "cx")


@dataclass
class Calibration:
    device: str
    snapshot: str                      # last_update_date of the snapshot, or "" for a hand made CSV
    two_qubit_gate: str
    t1_us: dict[int, float]
    t2_us: dict[int, float]
    readout_error: dict[int, float]
    p01: dict[int, float]              # Prob meas0 prep1
    p10: dict[int, float]              # Prob meas1 prep0
    readout_ns: dict[int, float]
    sx_error: dict[int, float]
    sx_ns: dict[int, float]
    reset_ns: dict[int, float]
    operational: dict[int, bool]
    coupler_error: dict[tuple[int, int], float] = field(default_factory=dict)
    coupler_ns: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def qubits(self) -> list[int]:
        return sorted(self.t1_us)

    def usable_qubits(self) -> list[int]:
        """Qubits marked operational with finite, positive T1 and T2 and a readout error below 0.5."""
        return [q for q in self.qubits if self.operational.get(q, True) and self.t1_us[q] > 0 and self.t2_us[q] > 0
                and all(math.isfinite(v) for v in (self.t1_us[q], self.t2_us[q], self.readout_error[q], self.sx_error[q],
                                                 self.readout_ns[q]))
                and self.readout_error[q] < 0.5]

    def usable_couplers(self) -> list[tuple[int, int]]:
        """Couplers with a finite reported error below 0.5 (IBM reports 1.0 for couplers out of service) between usable
        qubits."""
        ok = set(self.usable_qubits())
        return sorted(c for c, e in self.coupler_error.items() if e is not None and e < 0.5 and c[0] in ok and c[1] in ok)


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


# ---------------------------------------------------------------- export from a snapshot


def props_path(device: str) -> Path:
    import qiskit_ibm_runtime.fake_provider as fp
    p = Path(fp.__file__).parent / "backends" / device / f"props_{device}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def from_props(props: dict) -> Calibration:
    name = props["backend_name"]
    n = len(props["qubits"])
    nan = float("nan")
    qv = [{x["name"]: x["value"] for x in q} for q in props["qubits"]]
    qv = [{k: v.get(k, nan) for k in ("T1", "T2", "readout_error", "readout_length")} | v for v in qv]
    gates = props["gates"]
    kinds = {g["gate"] for g in gates}
    g2 = next(k for k in TWO_Q if k in kinds)

    def gp(g, key):
        return next((p["value"] for p in g["parameters"] if p["name"] == key), None)

    sx = {g["qubits"][0]: g for g in gates if g["gate"] == "sx"}
    reset = {g["qubits"][0]: g for g in gates if g["gate"] == "reset"}
    ce: dict[tuple[int, int], list[float]] = {}
    cl: dict[tuple[int, int], list[float]] = {}
    for g in gates:
        if g["gate"] == g2:
            k = _pair(*g["qubits"])
            ce.setdefault(k, []).append(gp(g, "gate_error"))
            cl.setdefault(k, []).append(gp(g, "gate_length"))
    oper = {}
    for i in range(n):
        v = qv[i].get("operational")
        oper[i] = True if v is None else bool(v)
    return Calibration(
        device=name, snapshot=props.get("last_update_date", ""), two_qubit_gate=g2,
        t1_us={i: qv[i]["T1"] for i in range(n)}, t2_us={i: qv[i]["T2"] for i in range(n)},
        readout_error={i: qv[i]["readout_error"] for i in range(n)},
        p01={i: qv[i].get("prob_meas0_prep1", qv[i]["readout_error"]) for i in range(n)},
        p10={i: qv[i].get("prob_meas1_prep0", qv[i]["readout_error"]) for i in range(n)},
        readout_ns={i: qv[i]["readout_length"] for i in range(n)},
        sx_error={i: gp(sx[i], "gate_error") if i in sx else nan for i in range(n)},
        sx_ns={i: gp(sx[i], "gate_length") if i in sx else nan for i in range(n)},
        reset_ns={i: gp(reset[i], "gate_length") if i in reset else qv[i]["readout_length"] for i in range(n)},
        operational=oper,
        coupler_error={k: sum(v) / len(v) for k, v in ce.items()},
        coupler_ns={k: sum(v) / len(v) for k, v in cl.items()})


HEADER = ["Qubit", "T1 (us)", "T2 (us)", "Readout assignment error", "Prob meas0 prep1", "Prob meas1 prep0",
          "Readout length (ns)", "√x (sx) error", "Single-qubit gate length (ns)", "{G} error", "Gate length (ns)",
          "Reset length (ns)", "Operational"]


def write_csv(cal: Calibration, path: Path) -> None:
    g = cal.two_qubit_gate.upper()
    by_q: dict[int, list[tuple[int, int]]] = {q: [] for q in cal.qubits}
    for c in sorted(cal.coupler_error):
        by_q[c[0]].append(c)
        by_q[c[1]].append(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# device={cal.device} snapshot={cal.snapshot} source=qiskit-ibm-runtime fake_provider props\n")
        w = csv.writer(f)
        w.writerow([h.format(G=g) for h in HEADER])
        for q in cal.qubits:
            cs = by_q[q]
            w.writerow([q, repr(cal.t1_us[q]), repr(cal.t2_us[q]), repr(cal.readout_error[q]), repr(cal.p01[q]),
                        repr(cal.p10[q]), repr(cal.readout_ns[q]), repr(cal.sx_error[q]), repr(cal.sx_ns[q]),
                        ";".join(f"{a}_{b}:{cal.coupler_error[(a, b)]!r}" for a, b in cs),
                        ";".join(f"{a}_{b}:{cal.coupler_ns[(a, b)]!r}" for a, b in cs),
                        repr(cal.reset_ns[q]), "true" if cal.operational[q] else "false"])


# ---------------------------------------------------------------- reader


def _find(header: list[str], *keys: str, exclude: tuple[str, ...] = ()) -> int | None:
    for i, h in enumerate(header):
        lh = h.strip().lower()
        if all(k in lh for k in keys) and not any(x in lh for x in exclude):
            return i
    return None


def read_csv(path: str | Path) -> Calibration:
    path = Path(path)
    meta = {}
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    body = []
    for ln in lines:
        if ln.startswith("#"):
            for tok in ln[1:].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    meta[k] = v
        elif ln.strip():
            body.append(ln)
    rows = list(csv.reader(body))
    hdr = rows[0]
    col = dict(
        q=_find(hdr, "qubit"), t1=_find(hdr, "t1"), t2=_find(hdr, "t2"),
        ro=_find(hdr, "readout", "error") if _find(hdr, "readout assignment") is None else _find(hdr, "readout assignment"),
        p01=_find(hdr, "meas0", "prep1"), p10=_find(hdr, "meas1", "prep0"), rol=_find(hdr, "readout length"),
        sx=_find(hdr, "sx", "error"), sxl=_find(hdr, "single-qubit gate length"),
        rst=_find(hdr, "reset length"), op=_find(hdr, "operational"),
        gl=_find(hdr, "gate length", exclude=("single",)) if _find(hdr, "gate time") is None else _find(hdr, "gate time"))
    g2 = None
    for k in TWO_Q:
        i = _find(hdr, f"{k} error")
        if i is not None:
            g2, col["ge"] = k, i
            break
    missing = [k for k in ("q", "t1", "t2", "ro", "sx") if col[k] is None] + ([] if g2 else ["two qubit error"])
    if missing:
        raise ValueError(f"{path}: columns not found: {missing}; header {hdr}")

    def num(r, k, default=None):
        i = col.get(k)
        if i is None or i >= len(r) or not r[i].strip():
            return default
        return float(r[i])  # "nan" stays nan and marks the qubit unusable

    cal = Calibration(device=meta.get("device", path.stem), snapshot=meta.get("snapshot", ""), two_qubit_gate=g2,
                      t1_us={}, t2_us={}, readout_error={}, p01={}, p10={}, readout_ns={}, sx_error={}, sx_ns={},
                      reset_ns={}, operational={})
    ce: dict[tuple[int, int], list[float]] = {}
    cl: dict[tuple[int, int], list[float]] = {}
    for r in rows[1:]:
        q = int(float(r[col["q"]]))
        cal.t1_us[q] = num(r, "t1")
        cal.t2_us[q] = num(r, "t2")
        cal.readout_error[q] = num(r, "ro")
        cal.p01[q] = num(r, "p01", cal.readout_error[q])
        cal.p10[q] = num(r, "p10", cal.readout_error[q])
        cal.readout_ns[q] = num(r, "rol", 1000.0)
        cal.sx_error[q] = num(r, "sx")
        cal.sx_ns[q] = num(r, "sxl", 32.0)
        cal.reset_ns[q] = num(r, "rst", cal.readout_ns[q])
        cal.operational[q] = (r[col["op"]].strip().lower() not in ("false", "no", "0")) if col["op"] is not None else True
        for k, store in (("ge", ce), ("gl", cl)):
            i = col.get(k)
            if i is None or i >= len(r):
                continue
            for ent in filter(None, (e.strip() for e in r[i].split(";"))):
                pair, val = ent.split(":")
                a, b = (int(x) for x in pair.replace("-", "_").split("_"))
                store.setdefault(_pair(a, b), []).append(float(val))
    cal.coupler_error = {k: sum(v) / len(v) for k, v in ce.items()}
    cal.coupler_ns = {k: sum(v) / len(v) for k, v in cl.items()}
    for k in cal.coupler_error:
        cal.coupler_ns.setdefault(k, 68.0 if g2 == "cz" else 500.0)
    return cal


def summary(cal: Calibration) -> dict:
    import statistics as st
    q, c = cal.usable_qubits(), cal.usable_couplers()

    def med(d, keys):
        return st.median(d[k] for k in keys)
    return dict(device=cal.device, snapshot=cal.snapshot, two_qubit_gate=cal.two_qubit_gate, qubits=len(cal.qubits),
                usable_qubits=len(q), couplers=len(cal.coupler_error), usable_couplers=len(c),
                median_t1_us=med(cal.t1_us, q), median_t2_us=med(cal.t2_us, q),
                median_readout_error=med(cal.readout_error, q), median_sx_error=med(cal.sx_error, q),
                median_2q_error=med(cal.coupler_error, c), median_2q_ns=med(cal.coupler_ns, c),
                median_readout_ns=med(cal.readout_ns, q), median_reset_ns=med(cal.reset_ns, q))


def same(a: Calibration, b: Calibration) -> bool:
    """Field by field equality, with nan equal to nan."""
    def eq(x, y):
        return x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
    for f in ("t1_us", "t2_us", "readout_error", "p01", "p10", "readout_ns", "sx_error", "sx_ns", "reset_ns",
              "operational", "coupler_error", "coupler_ns"):
        da, db = getattr(a, f), getattr(b, f)
        if da.keys() != db.keys() or not all(eq(float(da[k]), float(db[k])) for k in da):
            return False
    return a.two_qubit_gate == b.two_qubit_gate


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--device", nargs="+", required=True)
    e.add_argument("--out", default="data/calibration")
    s = sub.add_parser("summary")
    s.add_argument("csv", nargs="+")
    a = ap.parse_args()
    if a.cmd == "export":
        for dev in a.device:
            props = json.loads(props_path(dev).read_text())
            cal = from_props(props)
            out = Path(a.out) / f"ibm_{dev}_{cal.snapshot[:10]}.csv"
            write_csv(cal, out)
            back = read_csv(out)
            assert same(back, cal), "round trip failed"
            print(out, json.dumps(summary(back)))
    else:
        for p in a.csv:
            print(json.dumps(summary(read_csv(p))))


if __name__ == "__main__":
    main()
