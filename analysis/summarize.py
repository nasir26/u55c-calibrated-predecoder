"""Tables, figures and paper macros from results/ler (CPU) and results/card (FPGA).

Author: Nasir Ali, C-DAC Noida.

For every (device, placement, scale, d) the X and Z basis experiments are pooled (2 x 200,000 shots). Reported per
decoder: failures, logical error per shot, error per round eps = (1 - (1 - 2P)^(1/r)) / 2. Paired comparisons use
the per shot failure bits: n10 = shots only the first decoder fails, n01 = only the second; ratio = fails ratio; the
two sided exact sign test on (n10, n01).

Outputs: results/summary.json, results/tables/*.csv, results/figures/*.pdf, paper/generated/*.tex.
usage: python analysis/summarize.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calib.ibm_csv import read_csv, summary  # noqa: E402

LER = ROOT / "results" / "ler"
CARD = ROOT / "results" / "card"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
GEN = ROOT / "paper" / "generated"
DECODERS = ("pm_cal", "pm_median", "pm_corr", "fast_fp32", "fast_int8", "fast_fp32_corr")
ORDER = ["ibm_aachen", "ibm_kingston", "ibm_marrakesh", "ibm_fez", "ibm_torino", "ibm_sherbrooke", "ibm_brisbane",
         "ibm_kyiv"]
FAMILY = {"ibm_aachen": "Heron r3", "ibm_kingston": "Heron r3", "ibm_marrakesh": "Heron r2", "ibm_fez": "Heron r2",
          "ibm_torino": "Heron r1", "ibm_sherbrooke": "Eagle r3", "ibm_brisbane": "Eagle r3", "ibm_kyiv": "Eagle r3"}


def per_round(p: float, r: int) -> float:
    return 0.5 * (1 - (1 - 2 * p) ** (1 / r)) if p < 0.5 else float("nan")


def sign_test(n10: int, n01: int) -> float:
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    # log space binomial tail (exact big integer sums are too slow at tens of thousands of discordant shots)
    lt = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    top = max(lt)
    return min(1.0, 2 * math.exp(top) * sum(math.exp(x - top) for x in lt))


def load_points() -> dict:
    """Records of bench/eval_ler.py, with the correlated matching decoders of bench/eval_corr.py merged in when the
    point has them (same shots by construction; checked through the sampling seed)."""
    pts = {}
    for f in sorted(glob.glob(str(LER / "*.json"))):
        if f.endswith(".corr.json"):
            continue
        j = json.loads(Path(f).read_text())
        c = j["calibration"]
        key = (c["device"], f"{c['mode']}{c['seed'] if c['mode'] == 'screened' else ''}", c["scale"], c["d"])
        z = np.load(f[:-5] + ".npz")
        bits = {k: np.unpackbits(z[k])[:j["shots"]].astype(bool) for k in z.files}
        cf = Path(f[:-5] + ".corr.json")
        if cf.exists():
            jc = json.loads(cf.read_text())
            assert jc["sample_seed"] == j["sample_seed"] and jc["shots"] == j["shots"], f
            zc = np.load(f[:-5] + ".corr.npz")
            bits.update({k: np.unpackbits(zc[k])[:j["shots"]].astype(bool) for k in zc.files})
        pts.setdefault(key, {})[c["basis"]] = dict(rec=j, bits=bits)
    return pts


def pooled(entry: dict) -> dict | None:
    if set(entry) != {"X", "Z"}:
        return None
    r = entry["X"]["rec"]["rounds"]
    n = entry["X"]["rec"]["shots"] + entry["Z"]["rec"]["shots"]
    out = dict(rounds=r, shots=n, decoders={})
    keys = set(entry["X"]["bits"]) & set(entry["Z"]["bits"])
    bits = {k: np.concatenate([entry["X"]["bits"][k], entry["Z"]["bits"][k]]) for k in sorted(keys)}
    for k, b in bits.items():
        f = int(b.sum())
        out["decoders"][k] = dict(fails=f, ler=f / n, eps_round=per_round(f / n, r),
                                  fails_X=int(entry["X"]["bits"][k].sum()), fails_Z=int(entry["Z"]["bits"][k].sum()))
    for a, b in (("fast_fp32", "pm_cal"), ("fast_int8", "pm_cal"), ("pm_median", "pm_cal"), ("fast_int8", "fast_fp32"),
                 ("pm_corr", "pm_cal"), ("fast_fp32", "pm_corr"), ("fast_int8", "pm_corr"),
                 ("fast_fp32_corr", "pm_corr")):
        if a in bits and b in bits:
            n10 = int((bits[a] & ~bits[b]).sum())
            n01 = int((~bits[a] & bits[b]).sum())
            out[f"{a}_vs_{b}"] = dict(ratio=bits[a].sum() / max(bits[b].sum(), 1), n10=n10, n01=n01,
                                      p_sign=sign_test(n10, n01))
    if {"pm_cal", "fast_int8", "pm_corr"} <= set(bits):
        fixed = bits["pm_cal"] & ~bits["fast_int8"]          # shots the network pipeline corrects and PM does not
        out["network_fixes"] = dict(n=int(fixed.sum()), also_pm_corr=int((fixed & ~bits["pm_corr"]).sum()))
    out["detector_density"] = 0.5 * (entry["X"]["rec"]["detector_density"] + entry["Z"]["rec"]["detector_density"])
    return out


def fmt(x: float, digits: int = 2) -> str:
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    return f"{x / 10 ** e:.{digits - 1}f}e{e}"


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    devs = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "calibration" / "*.csv"))):
        s = summary(read_csv(f))
        s["csv"] = Path(f).name
        devs[s["device"]] = s
    pts = load_points()
    rows = []
    for key in sorted(pts, key=lambda k: (ORDER.index(k[0]), k[1], -k[2], k[3])):
        p = pooled(pts[key])
        if p is None:
            continue
        rows.append(dict(device=key[0], placement=key[1], scale=key[2], d=key[3], **p))
    card = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(CARD / "card_main*.json")))]
    # card decisions against the CPU integer model, shot by shot, from the stored failure bits (also for configs
    # whose CPU record was written after the card run)
    for cc in card:
        for c in cc.get("configs", []):
            cpu = LER / c["failure_bits"]
            if cpu.exists() and "fast_int8" in np.load(cpu).files:
                a = np.unpackbits(np.load(CARD / c["failure_bits"])["card_int8"])[:c["shots"]]
                b = np.unpackbits(np.load(cpu)["fast_int8"])[:c["shots"]]
                c["same_failures_as_cpu_int8"] = bool(np.array_equal(a, b))
    cpu = json.loads((ROOT / "results" / "cpu" / "cpu_throughput.json").read_text()) \
        if (ROOT / "results" / "cpu" / "cpu_throughput.json").exists() else None
    out = dict(tier="T4-SIM (CPU) and T1-HW (card)", devices=devs, points=rows, card=card, cpu_throughput=cpu)
    (ROOT / "results" / "summary.json").write_text(json.dumps(out, indent=1, default=float))

    # device table
    with open(TAB / "devices.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "family", "snapshot", "qubits", "usable", "2q gate", "median 2q error", "median sx error",
                    "median readout error", "median T1 us", "median T2 us", "median readout ns"])
        for d in ORDER:
            s = devs[d]
            w.writerow([d, FAMILY[d], s["snapshot"][:10], s["qubits"], s["usable_qubits"], s["two_qubit_gate"].upper(),
                        f"{s['median_2q_error']:.2e}", f"{s['median_sx_error']:.2e}", f"{s['median_readout_error']:.2e}",
                        f"{s['median_t1_us']:.0f}", f"{s['median_t2_us']:.0f}", f"{s['median_readout_ns']:.0f}"])
    # decoder table
    with open(TAB / "decoders.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "placement", "scale", "d", "rounds", "shots", "detector density"]
                   + [f"{k} fails" for k in DECODERS] + [f"{k} eps/round" for k in DECODERS]
                   + ["fp32/pm_cal", "p fp32", "int8/pm_cal", "p int8", "pm_median/pm_cal", "p median",
                      "pm_corr/pm_cal", "p corr", "int8/pm_corr", "p int8 vs corr", "fp32_corr/pm_corr", "p fp32_corr"])
        for r in rows:
            dd = r["decoders"]
            w.writerow([r["device"], r["placement"], r["scale"], r["d"], r["rounds"], r["shots"],
                        f"{r['detector_density']:.4f}"]
                       + [dd[k]["fails"] if k in dd else "" for k in DECODERS]
                       + [fmt(dd[k]["eps_round"], 3) if k in dd else "" for k in DECODERS]
                       + sum(([f"{r[c]['ratio']:.3f}", f"{r[c]['p_sign']:.2g}"] if c in r else ["", ""]
                              for c in ("fast_fp32_vs_pm_cal", "fast_int8_vs_pm_cal", "pm_median_vs_pm_cal",
                                        "pm_corr_vs_pm_cal", "fast_int8_vs_pm_corr", "fast_fp32_corr_vs_pm_corr")), []))

    figures(rows, devs, card)
    macros(rows, devs, card, cpu)
    latex_tables(rows, devs, card, cpu)
    print(f"{len(rows)} pooled points; tables in {TAB}")


def figures(rows, devs, card):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    med = [r for r in rows if r["placement"] == "median"]
    # 1: ratio of the pre-decoder pipeline to PyMatching alone, per device, scale and d
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, d in zip(axs, (3, 5)):
        for i, dev in enumerate(ORDER):
            rr = sorted([r for r in med if r["device"] == dev and r["d"] == d and "fast_int8_vs_pm_cal" in r],
                        key=lambda r: r["scale"])
            if rr:
                ax.plot([r["scale"] for r in rr], [r["fast_int8_vs_pm_cal"]["ratio"] for r in rr], marker="o", ms=3,
                        lw=1, color=f"C{i}", label=f"{dev[4:]} ({FAMILY[dev]})")
        ax.axhline(1, color="k", lw=0.7, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1], ["0.25", "0.5", "1"])
        ax.set_xlabel("error scale s (1 = as calibrated)")
        ax.set_title(f"d = {d}, {d} rounds, median placement")
    axs[0].set_ylabel("failures: Fast INT8 + PyMatching\n/ PyMatching alone")
    axs[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ratio.pdf", metadata={"CreationDate": None})
    # 2: error per round against d and scale for PyMatching alone and the pre-decoder pipeline
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, s in zip(axs, (1.0, 0.5, 0.25)):
        for i, dev in enumerate(ORDER):
            rr = sorted([r for r in med if r["device"] == dev and r["scale"] == s], key=lambda r: r["d"])
            if rr:
                ax.plot([r["d"] for r in rr], [r["decoders"]["pm_cal"]["eps_round"] for r in rr], color=f"C{i}",
                        marker="o", ms=3, lw=1, label=dev[4:])
                if all("pm_corr" in r["decoders"] for r in rr):
                    ax.plot([r["d"] for r in rr], [r["decoders"]["pm_corr"]["eps_round"] for r in rr], color=f"C{i}",
                            marker="x", ms=3, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xticks([3, 5])
        ax.set_xlabel("code distance d")
        ax.set_title(f"s = {s:g}")
    axs[0].set_ylabel("logical error per round")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="center right")
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(FIG / "fig_eps_round.pdf", metadata={"CreationDate": None})
    # 3: value of per qubit matching weights (heterogeneous placements)
    het = [r for r in rows if r["placement"] != "median" and "pm_median_vs_pm_cal" in r]
    if het:
        fig, ax = plt.subplots(figsize=(5.0, 2.5))
        for i, dev in enumerate(ORDER):
            rr = [r for r in het if r["device"] == dev]
            ax.scatter([r["decoders"]["pm_cal"]["ler"] for r in rr], [r["pm_median_vs_pm_cal"]["ratio"] for r in rr],
                       s=8, color=f"C{i}", label=dev[4:])
        ax.axhline(1, color="k", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("logical error per shot, PyMatching, calibrated weights")
        ax.set_ylabel("failures, median weights\n/ calibrated weights")
        ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(FIG / "fig_weights.pdf", metadata={"CreationDate": None})
    # 5: the pre-decoder pipeline against both baselines, every point, by the error rate of plain matching
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, d in zip(axs, (3, 5)):
        rr = [r for r in rows if r["d"] == d and "fast_int8_vs_pm_corr" in r]
        x = [r["decoders"]["pm_cal"]["ler"] for r in rr]
        ax.scatter(x, [r["fast_int8_vs_pm_cal"]["ratio"] for r in rr], s=9, marker="o", color="C0",
                   label="Fast INT8 + PyMatching / PyMatching")
        ax.scatter(x, [r["fast_int8_vs_pm_corr"]["ratio"] for r in rr], s=9, marker="^", color="C3",
                   label="Fast INT8 + PyMatching / correlated PyMatching")
        ax.axhline(1, color="k", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("logical error per shot, PyMatching")
        ax.set_title(f"d = {d}, {d} rounds, all placements and scales")
    axs[0].set_ylabel("failure ratio (< 1: pre-decoder better)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_baselines.pdf", metadata={"CreationDate": None})
    # 4: card throughput and latency against d
    perf = [p for c in card for p in c.get("perf", [])]
    if perf:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        x = [p["d"] for p in perf]
        ax.plot(x, [np.median(p["windows_per_sec_reps"]) for p in perf], "o-", ms=3, lw=1, color="C0")
        ax.set_yscale("log")
        ax.set_xlabel("code distance d (= rounds)")
        ax.set_ylabel("windows per second, one card", color="C0")
        ax2 = ax.twinx()
        ax2.plot(x, [p["latency_one_window_s"] * 1e3 for p in perf], "s--", ms=3, lw=1, color="C3")
        ax2.set_yscale("log")
        ax2.set_ylabel("latency of one window (ms)", color="C3")
        fig.tight_layout()
        fig.savefig(FIG / "fig_card.pdf", metadata={"CreationDate": None})


def latex_tables(rows, devs, card, cpu):
    """paper/generated/tab_*.tex: devices, wins and losses per baseline, card runs and throughput."""
    def w(name, lines):
        (GEN / name).write_text("% generated by analysis/summarize.py; do not edit\n" + "\n".join(lines) + "\n")
    t = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Device & Family & Usable & 2q & 2q error & sx error & Readout & $T_1$ & $T_2$ \\\\",
         " & (snapshot) & qubits & gate & (\\%) & (\\%) & error (\\%) & ($\\mu$s) & ($\\mu$s) \\\\", "\\midrule"]
    for d in ORDER:
        v = devs[d]
        t.append(f"{d.replace('ibm_', '').capitalize()} & {FAMILY[d]} ({v['snapshot'][:7]}) & {v['usable_qubits']} & "
                 f"{v['two_qubit_gate'].upper()} & {100 * v['median_2q_error']:.2f} & {100 * v['median_sx_error']:.3f} & "
                 f"{100 * v['median_readout_error']:.2f} & {v['median_t1_us']:.0f} & {v['median_t2_us']:.0f} \\\\")
    w("tab_devices.tex", t + ["\\bottomrule", "\\end{tabular}"])
    t = ["\\begin{tabular}{llrrrr}", "\\toprule",
         "Comparison & $d$ & Points & Better & Worse & Failure ratio \\\\", "\\midrule"]
    for lab, cmp in (("INT8 pre-decoder + PM vs PM", "fast_int8_vs_pm_cal"),
                     ("INT8 pre-decoder + PM vs PM-corr", "fast_int8_vs_pm_corr"),
                     ("float pre-decoder + PM vs PM-corr", "fast_fp32_vs_pm_corr"),
                     ("float pre-decoder + PM-corr vs PM-corr", "fast_fp32_corr_vs_pm_corr"),
                     ("PM-corr vs PM", "pm_corr_vs_pm_cal"),
                     ("PM, median weights vs calibrated", "pm_median_vs_pm_cal")):
        for dd in (3, 5):
            rr = [r[cmp] for r in rows if r["d"] == dd and cmp in r
                  and (cmp != "pm_median_vs_pm_cal" or r["placement"] != "median")]
            if rr:
                t.append(f"{lab} & {dd} & {len(rr)} & {sum(1 for c in rr if c['ratio'] < 1 and c['p_sign'] < 0.05)} & "
                         f"{sum(1 for c in rr if c['ratio'] > 1 and c['p_sign'] < 0.05)} & "
                         f"{min(c['ratio'] for c in rr):.2f} to {max(c['ratio'] for c in rr):.2f} \\\\")
    w("tab_wins.tex", t + ["\\bottomrule", "\\end{tabular}"])
    cfgs = [c for cc in card for c in cc.get("configs", [])]
    t = ["\\begin{tabular}{lrrlrrrl}", "\\toprule",
         "Device & Scale & $d$ & Basis & Shots & PM & Card + PM & Same as CPU \\\\", "\\midrule"]
    for c in cfgs:
        same = {True: "yes", False: "no"}.get(c.get("same_failures_as_cpu_int8"), "n/a")
        t.append(f"{c['device_name'].replace('ibm_', '').capitalize()} & {c['scale']:g} & {c['d']} & {c['basis']} & "
                 f"{c['shots']:,} & {c['pm_cal']['fails']:,} & {c['card_int8']['fails']:,} & {same} \\\\")
    w("tab_card.tex", t + ["\\bottomrule", "\\end{tabular}"])
    perf = {p["d"]: p for cc in card for p in cc.get("perf", [])}
    cpu_d = {r["d"]: r for r in (cpu or {}).get("rows", [])}
    t = ["\\begin{tabular}{rrrrr}", "\\toprule",
         "$d$ & Card (windows/s) & One window (ms) & CPU (windows/s) & CPU / card \\\\", "\\midrule"]
    for dd in sorted(perf):
        cr = float(np.median(perf[dd]["windows_per_sec_reps"]))
        cp = cpu_d.get(dd)
        t.append(f"{dd} & {cr:,.0f} & {perf[dd]['latency_one_window_s'] * 1e3:.2f} & "
                 + (f"{cp['windows_per_sec']:,.0f} & {cp['windows_per_sec'] / cr:.1f}" if cp else "n/a & n/a")
                 + " \\\\")
    w("tab_perf.tex", t + ["\\bottomrule", "\\end{tabular}"])


def macros(rows, devs, card, cpu=None):
    """LaTeX macros for every number the paper quotes (paper/generated/macros.tex)."""
    m = {}
    m["NumDevices"] = str(len(devs))
    m["NumPoints"] = str(len(rows))
    med = [r for r in rows if r["placement"] == "median" and "fast_int8_vs_pm_cal" in r]
    if med:
        ratios = [r["fast_int8_vs_pm_cal"]["ratio"] for r in med]
        m["RatioMin"], m["RatioMax"] = f"{min(ratios):.2f}", f"{max(ratios):.2f}"
        m["NumBetter"] = str(sum(1 for r in med if r["fast_int8_vs_pm_cal"]["ratio"] < 1
                                 and r["fast_int8_vs_pm_cal"]["p_sign"] < 0.05))
        m["NumWorse"] = str(sum(1 for r in med if r["fast_int8_vs_pm_cal"]["ratio"] > 1
                                and r["fast_int8_vs_pm_cal"]["p_sign"] < 0.05))
        m["NumMedianPoints"] = str(len(med))
    # counts of significant (sign test p < 0.05) wins and losses over all points, per d and baseline
    for cmp, tag in (("fast_int8_vs_pm_cal", "Pm"), ("fast_int8_vs_pm_corr", "Corr"), ("pm_corr_vs_pm_cal", "CorrPm"),
                     ("fast_fp32_corr_vs_pm_corr", "FcCorr")):
        for d, dt in ((3, "Three"), (5, "Five")):
            rr = [r[cmp] for r in rows if r["d"] == d and cmp in r]
            if not rr:
                continue
            m[f"{tag}{dt}N"] = str(len(rr))
            m[f"{tag}{dt}Better"] = str(sum(1 for c in rr if c["ratio"] < 1 and c["p_sign"] < 0.05))
            m[f"{tag}{dt}Worse"] = str(sum(1 for c in rr if c["ratio"] > 1 and c["p_sign"] < 0.05))
            m[f"{tag}{dt}Min"] = f"{min(c['ratio'] for c in rr):.2f}"
            m[f"{tag}{dt}Max"] = f"{max(c['ratio'] for c in rr):.2f}"
    het = [r["pm_median_vs_pm_cal"]["ratio"] for r in rows if r["placement"] != "median" and "pm_median_vs_pm_cal" in r]
    if het:
        m["MedWeightsMin"], m["MedWeightsMax"] = f"{min(het):.2f}", f"{max(het):.2f}"
    ks = [r["fast_int8_vs_pm_cal"]["ratio"] for r in rows if r["device"] == "ibm_kingston"
          and r["placement"].startswith("screened") and "fast_int8_vs_pm_cal" in r]
    if ks:
        m["KingScreenMin"], m["KingScreenMax"] = f"{min(ks):.2f}", f"{max(ks):.2f}"
        m["KingScreenN"] = str(len(ks))
        m["KingScreenWorse"] = str(sum(1 for r in rows if r["device"] == "ibm_kingston"
                                       and r["placement"].startswith("screened")
                                       and r["fast_int8_vs_pm_cal"]["ratio"] > 1
                                       and r["fast_int8_vs_pm_cal"]["p_sign"] < 0.05))
    for d, dt in ((3, "Three"), (5, "Five")):
        rr = [r["pm_median_vs_pm_cal"] for r in rows if r["d"] == d and r["placement"] != "median"
              and "pm_median_vs_pm_cal" in r]
        m[f"MedW{dt}N"] = str(len(rr))
        m[f"MedW{dt}Worse"] = str(sum(1 for c in rr if c["ratio"] > 1 and c["p_sign"] < 0.05))
    m["TotalShots"] = f"{sum(r['shots'] for r in rows):,}"
    # distance 3 at the lowest error scale: correlated matching against plain, the pre-decoder against correlated
    low = [r for r in rows if r["d"] == 3 and r["scale"] == 0.25 and "pm_corr_vs_pm_cal" in r]
    if low:
        m["LowN"] = str(len(low))
        m["LowCorrWorse"] = str(sum(1 for r in low if r["pm_corr_vs_pm_cal"]["ratio"] > 1))
        m["LowFastBetterCorr"] = str(sum(1 for r in low if r["fast_int8_vs_pm_corr"]["ratio"] < 1
                                         and r["fast_int8_vs_pm_corr"]["p_sign"] < 0.05))
    # devices whose median circuit is below threshold (error per round falls from d = 3 to d = 5), per decoder and scale
    for sc, st_ in ((1.0, "One"), (0.5, "Half"), (0.25, "Quarter")):
        for dec, dt in (("pm_cal", "Pm"), ("pm_corr", "Corr"), ("fast_int8", "Fast")):
            pr = {}
            for r in rows:
                if r["placement"] == "median" and r["scale"] == sc and dec in r["decoders"]:
                    pr.setdefault(r["device"], {})[r["d"]] = r["decoders"][dec]["eps_round"]
            m[f"Below{dt}{st_}"] = str(sum(1 for v in pr.values() if 3 in v and 5 in v and v[5] < v[3]))
    ex = [r for r in rows if r["device"] == "ibm_aachen" and r["placement"] == "median" and r["scale"] == 1.0
          and r["d"] == 5]
    if ex:
        for k, t in (("pm_cal", "Pm"), ("pm_corr", "Corr"), ("fast_int8", "Fast"), ("fast_fp32_corr", "FastCorr")):
            m[f"ExAachen{t}"] = f"{ex[0]['decoders'][k]['fails']:,}"
    q = [r["fast_int8_vs_fast_fp32"] for r in rows if "fast_int8_vs_fast_fp32" in r]
    if q:
        m["QuantMin"], m["QuantMax"] = f"{min(c['ratio'] for c in q):.2f}", f"{max(c['ratio'] for c in q):.2f}"
        m["QuantBetter"] = str(sum(1 for c in q if c["ratio"] < 1 and c["p_sign"] < 0.05))
        m["QuantWorse"] = str(sum(1 for c in q if c["ratio"] > 1 and c["p_sign"] < 0.05))
        m["QuantN"] = str(len(q))
    for d, dt in ((3, "Three"), (5, "Five")):
        nf = [r["network_fixes"] for r in rows if r["d"] == d and "network_fixes" in r]
        if nf:
            m[f"Overlap{dt}"] = f"{100 * sum(x['also_pm_corr'] for x in nf) / sum(x['n'] for x in nf):.0f}"
    best5 = [r["fast_int8_vs_pm_cal"]["ratio"] for r in rows if r["d"] == 5 and r["placement"] == "best"]
    med5 = [r["fast_int8_vs_pm_cal"]["ratio"] for r in rows if r["d"] == 5 and r["placement"] == "median"]
    m["MedianFiveMeanRatio"], m["BestFiveMeanRatio"] = f"{np.mean(med5):.2f}", f"{np.mean(best5):.2f}"
    # spread of the calibrations: worst usable coupler over the median one, median readout over median 2q error
    import statistics as st
    worst, ro = [], []
    for f in sorted(glob.glob(str(ROOT / "data" / "calibration" / "*.csv"))):
        c = read_csv(f)
        e = [c.coupler_error[k] for k in c.usable_couplers()]
        worst.append(max(e) / st.median(e))
        ro.append(st.median(c.readout_error[q] for q in c.usable_qubits()) / st.median(e))
    m["WorstCouplerMin"], m["WorstCouplerMax"] = f"{min(worst):.0f}", f"{max(worst):.0f}"
    m["ReadoutOverGateMin"], m["ReadoutOverGateMax"] = f"{min(ro):.1f}", f"{max(ro):.1f}"
    cfgs = [c for cc in card for c in cc.get("configs", [])]
    if cfgs:
        m["CardConfigs"] = str(len(cfgs))
        m["CardAllExact"] = "all" if all(c["bit_exact"] for c in cfgs) else "not all"
        m["CardSameAsCPU"] = str(sum(1 for c in cfgs if c.get("same_failures_as_cpu_int8")))
        m["CardCompared"] = str(sum(1 for c in cfgs if "same_failures_as_cpu_int8" in c))
    perf = [p for cc in card for p in cc.get("perf", [])]
    for p in perf:
        tag = "abcdefghijklmn"[p["d"]]
        m[f"CardRateD{tag}"] = f"{np.median(p['windows_per_sec_reps']):,.0f}"
        m[f"CardLatD{tag}"] = f"{p['latency_one_window_s'] * 1e3:.2f}"
    if cpu:
        m["CpuName"] = cpu["cpu"].replace("(R)", "").replace("(TM)", "").replace(" Processor", "")
        m["CpuThreads"] = str(cpu["threads"])
        perf_d = {p["d"]: float(np.median(p["windows_per_sec_reps"])) for p in perf}
        for r in cpu["rows"]:
            tag = "abcdefghijklmn"[r["d"]]
            m[f"CpuRateD{tag}"] = f"{r['windows_per_sec']:,.0f}"
            if r["d"] in perf_d:
                m[f"CpuOverCardD{tag}"] = f"{r['windows_per_sec'] / perf_d[r['d']]:.1f}"
    teach = ROOT / "results" / "teaching"
    for sc, tag in (("1", "One"), ("0.5", "Half"), ("0.25", "Quarter")):
        f = teach / f"kingston_s{sc}.json"
        if f.exists():
            res = {x["d"]: x for x in json.loads(f.read_text())["results"]}
            m[f"TeachDOne{tag}"] = f"{100 * res[1]['ler']:.2f}"
            m[f"TeachDTwoKept{tag}"] = f"{100 * res[2]['ler_kept']:.3f}"
            m[f"TeachDTwoDiscard{tag}"] = f"{100 * res[2]['discarded_fraction']:.0f}"
            m[f"TeachDThree{tag}"] = f"{100 * res[3]['ler']:.2f}"
            m[f"TeachDFive{tag}"] = f"{100 * res[5]['ler']:.3f}"
    with open(GEN / "macros.tex", "w") as fh:
        fh.write("% generated by analysis/summarize.py; do not edit\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


if __name__ == "__main__":
    main()
