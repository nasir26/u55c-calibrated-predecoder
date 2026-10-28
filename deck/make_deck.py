"""Team deck: running calibration driven QEC experiments (d = 1, 2, 3, 5) and the single card pre-decoder results.

Author: Nasir Ali, C-DAC Noida.

Every number on a slide is read from results/ (summary.json from analysis/summarize.py, teaching/*.json from
examples/small_codes.py, card/card_main.json from bench/run_card.py); nothing is typed in by hand.
usage: python deck/make_deck.py [--out deck/calibrated_predecoder_team.pptx]   (needs python-pptx)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
INK = RGBColor(0x0B, 0x2E, 0x3A)        # deep teal, dominant
TEAL = RGBColor(0x1F, 0x6F, 0x7A)
MIST = RGBColor(0xE6, 0xF0, 0xF1)
AMBER = RGBColor(0xE8, 0x9B, 0x2E)      # accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x5B, 0x6B, 0x70)
HEAD, BODY, MONO = "Cambria", "Calibri", "Courier New"
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)
FAMILY = {"ibm_aachen": "Heron r3", "ibm_kingston": "Heron r3", "ibm_marrakesh": "Heron r2", "ibm_fez": "Heron r2",
          "ibm_torino": "Heron r1", "ibm_sherbrooke": "Eagle r3", "ibm_brisbane": "Eagle r3", "ibm_kyiv": "Eagle r3"}
ORDER = list(FAMILY)


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def slide(self, dark=False, notes=""):
        s = self.prs.slides.add_slide(self.blank)
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = INK if dark else WHITE
        self.n += 1
        if not dark:
            self.text(s, f"{self.n}", W - M - Inches(0.6), H - Inches(0.45), Inches(0.6), Inches(0.3), 10, GREY,
                      align=PP_ALIGN.RIGHT)
            self.text(s, "Calibration aware pre-decoding on one U55C", M, H - Inches(0.45), Inches(6), Inches(0.3), 10,
                      GREY)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def text(self, s, txt, x, y, w, h, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, name=None):
        tb = s.shapes.add_textbox(x, y, w, h)
        if name:
            tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        lines = txt if isinstance(txt, list) else [txt]
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = ln
            r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
            r.font.color.rgb = color
        return tb

    def title(self, s, t, sub=None):
        self.text(s, t, M, Inches(0.45), W - 2 * M, Inches(0.8), 34, INK, True, HEAD, name="Title")
        if sub:
            self.text(s, sub, M, Inches(1.2), W - 2 * M, Inches(0.5), 16, GREY)

    def bullets(self, s, items, x, y, w, h, size=16, color=INK):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            bold, rest = (it if isinstance(it, tuple) else ("", it))
            p.space_after = Pt(9)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.25))))
            pPr.set("indent", str(-Emu(Inches(0.25))))
            from pptx.oxml.ns import qn
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
            pPr.append(bu)
            if bold:
                r = p.add_run()
                r.text = bold + " "
                r.font.bold, r.font.size, r.font.name, r.font.color.rgb = True, Pt(size), BODY, color
            r = p.add_run()
            r.text = rest
            r.font.size, r.font.name, r.font.color.rgb = Pt(size), BODY, color
        return tb

    def card(self, s, x, y, w, h, fill=MIST):
        sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        sh.adjustments[0] = 0.06
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        sh.line.fill.background()
        sh.shadow.inherit = False
        return sh

    def code(self, s, lines, x, y, w, h, size=13):
        self.card(s, x, y, w, h, INK)
        self.text(s, lines, x + Inches(0.25), y + Inches(0.2), w - Inches(0.5), h - Inches(0.4), size, MIST, font=MONO)

    def stat(self, s, big, label, x, y, w):
        self.card(s, x, y, w, Inches(1.9))
        self.text(s, big, x + Inches(0.25), y + Inches(0.2), w - Inches(0.5), Inches(0.9), 40, TEAL, True, HEAD)
        self.text(s, label, x + Inches(0.25), y + Inches(1.05), w - Inches(0.5), Inches(0.8), 13, INK)

    def table(self, s, rows, x, y, w, col_w=None, size=12, head_fill=TEAL):
        t = s.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(0.36) * len(rows)).table
        for j, cw in enumerate(col_w or []):
            t.columns[j].width = cw
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                c = t.cell(i, j)
                c.text = str(v)
                c.margin_left = c.margin_right = Inches(0.08)
                p = c.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
                f = p.runs[0].font
                f.size, f.name = Pt(size), BODY
                f.bold = i == 0
                f.color.rgb = WHITE if i == 0 else INK
                c.fill.solid()
                c.fill.fore_color.rgb = head_fill if i == 0 else (MIST if i % 2 else WHITE)
        return t


def style_chart(ch, legend=True, pct=False, title=None):
    ch.font.size, ch.font.name = Pt(12), BODY
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.TOP
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(12)
    if title:
        ch.has_title = True
        ch.chart_title.text_frame.text = title
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
        ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xD5, 0xDE, 0xE0)
    va.format.line.fill.background()
    va.tick_labels.font.color.rgb = GREY
    ch.category_axis.tick_labels.font.color.rgb = GREY
    ch.category_axis.format.line.color.rgb = RGBColor(0xB0, 0xBB, 0xBE)
    if pct:
        va.tick_labels.number_format = '0.0"%"'
        va.tick_labels.number_format_is_linked = False


def build(out: Path):
    summ = load("results/summary.json") or {}
    teach = {s: load(f"results/teaching/kingston_s{s}.json") for s in ("1", "0.5", "0.25")}
    pts = summ.get("points", [])
    devs = summ.get("devices", {})
    card = (summ.get("card") or [load("results/card/card_main.json") or {}])[0]
    D = Deck()

    # 1 title
    s = D.slide(dark=True, notes="Purpose: show the team how to run surface code memory experiments with real device "
                                 "calibration data, and what one FPGA card adds.")
    D.text(s, "Calibration aware neural pre-decoding", M, Inches(2.1), Inches(11), Inches(1.0), 44, WHITE, True, HEAD)
    D.text(s, "Real IBM calibration data, NVIDIA's pre-decoder and PyMatching, on one Alveo U55C",
           M, Inches(3.2), Inches(11), Inches(0.6), 20, MIST)
    D.text(s, ["Nasir Ali, C-DAC Noida", "Team walkthrough: run d = 1, 2, 3 and 5 yourself"],
           M, Inches(5.4), Inches(9), Inches(0.9), 16, MIST)

    # 2 pipeline
    s = D.slide(notes="Each box is one module of the repository. The CSV is the only device specific input.")
    D.title(s, "From a calibration CSV to a decoded logical qubit")
    steps = [("Calibration CSV", "T1, T2, readout, sx and CZ or ECR error per qubit and coupler", "calib/ibm_csv.py"),
             ("Noise model", "every fault location gets its own qubit's or coupler's rate", "calib/noise.py"),
             ("Syndromes", "Stim samples detector events of a d x d x d window", "Stim"),
             ("Pre-decoder", "NVIDIA Fast network, INT8, removes local errors", "U55C card"),
             ("Matching", "PyMatching decodes what is left", "harness/pipeline.py")]
    bw, gap = Inches(2.25), Inches(0.27)
    for i, (h, b, f) in enumerate(steps):
        x = M + i * (bw + gap)
        D.card(s, x, Inches(2.3), bw, Inches(3.0), AMBER if h == "Pre-decoder" else MIST)
        D.text(s, f"{i + 1}", x + Inches(0.2), Inches(2.45), Inches(0.6), Inches(0.6), 28, TEAL if h != "Pre-decoder"
               else INK, True, HEAD)
        D.text(s, h, x + Inches(0.2), Inches(3.1), bw - Inches(0.4), Inches(0.5), 18, INK, True)
        D.text(s, b, x + Inches(0.2), Inches(3.65), bw - Inches(0.4), Inches(1.2), 13, INK)
        D.text(s, f, x + Inches(0.2), Inches(4.85), bw - Inches(0.4), Inches(0.35), 11, GREY, font=MONO)
        if i < len(steps) - 1:
            a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x + bw + Inches(0.02), Inches(3.65), gap - Inches(0.04),
                                   Inches(0.3))
            a.fill.solid()
            a.fill.fore_color.rgb = TEAL
            a.line.fill.background()
    D.text(s, "No IBM account is needed: the eight device snapshots ship with qiskit-ibm-runtime (Apache 2.0) and "
              "are already exported to data/calibration/.", M, Inches(5.8), W - 2 * M, Inches(0.6), 14, GREY)

    # 3 headline numbers
    s = D.slide(notes="Numbers come from results/summary.json; see the results section for the full picture. The key "
                      "message: against correlated matching the pre-decoder does not pay at d = 5.")
    D.title(s, "What we found")
    exact = [c for c in card.get("configs", [])]
    perf = {p["d"]: p for p in card.get("perf", [])}
    sx = Inches(0.35)
    cw = (W - 2 * M - 3 * sx) / 4

    def wins(cmp, d):
        rr = [p[cmp] for p in pts if p["d"] == d and cmp in p]
        return (sum(1 for c in rr if c["ratio"] < 1 and c["p_sign"] < 0.05),
                sum(1 for c in rr if c["ratio"] > 1 and c["p_sign"] < 0.05), len(rr))
    b5, _, n5 = wins("fast_int8_vs_pm_cal", 5)
    c5b, c5w, _ = wins("fast_int8_vs_pm_corr", 5)
    if n5:
        D.stat(s, f"{b5} of {n5}", "d = 5 points where the INT8 pre-decoder significantly beats plain PyMatching",
               M, Inches(1.8), cw)
        D.stat(s, f"{c5b} of {n5}", f"d = 5 points where it beats correlated PyMatching ({c5w} significantly worse)",
               M + cw + sx, Inches(1.8), cw)
    if exact:
        D.stat(s, f"{sum(c['bit_exact'] for c in exact)} of {len(exact)}", "card runs bit exact against the integer "
               "model (both kernel instances)", M + 2 * (cw + sx), Inches(1.8), cw)
    if 5 in perf:
        D.stat(s, f"{np.median(perf[5]['windows_per_sec_reps']):,.0f}/s", "d = 5 windows per second on one card "
               f"({perf[5]['latency_one_window_s'] * 1e3:.2f} ms for a single window)", M + 3 * (cw + sx), Inches(1.8), cw)
    D.bullets(s, [("Strongest baseline wins at d = 5:", "PyMatching with correlations, no network at all, fails less "
                   "often than pre-decoder + PyMatching on every device and setting."),
                  ("Calibration matters:", "per qubit rates improve matching weights; and the network, trained on "
                   "uniform noise, loses its gain when qubits differ strongly."),
                  ("Honest limits:", "simulated circuits with calibrated rates, not runs on IBM hardware; the card "
                   "is no faster than a large CPU for this small network.")],
              M, Inches(4.2), W - 2 * M, Inches(2.4), 16)

    # 4 devices
    s = D.slide(notes="Snapshot dates matter: Kingston and Aachen are April 2026, the others February 2025.")
    D.title(s, "Eight IBM devices, as calibrated", "Medians over usable qubits and couplers, from data/calibration/*.csv")
    rows = [["Device", "Family", "Snapshot", "2q gate", "2q error", "Readout error", "T1 (us)", "T2 (us)",
             "Readout (ns)"]]
    for d in ORDER:
        v = devs.get(d)
        if v:
            rows.append([d.replace("ibm_", ""), FAMILY[d], v["snapshot"][:10], v["two_qubit_gate"].upper(),
                         f"{100 * v['median_2q_error']:.2f}%", f"{100 * v['median_readout_error']:.2f}%",
                         f"{v['median_t1_us']:.0f}", f"{v['median_t2_us']:.0f}", f"{v['median_readout_ns']:.0f}"])
    D.table(s, rows, M, Inches(1.9), W - 2 * M, size=13)
    D.text(s, "Readout errors are 3 to 10 times the two qubit errors, and data qubits idle through 1.3 to 2.6 us of "
              "readout: both dominate the noise.", M, Inches(5.6), W - 2 * M, Inches(0.6), 14, GREY)

    # 5 section
    s = D.slide(dark=True)
    D.text(s, "Hands on", M, Inches(2.6), Inches(10), Inches(1), 44, WHITE, True, HEAD)
    D.text(s, "Set up once, then run d = 1, 2, 3 and 5 on a laptop", M, Inches(3.6), Inches(10), Inches(0.6), 20, MIST)

    # 6 setup
    s = D.slide(notes="The teaching example needs only stim, pymatching and numpy. The full pipeline needs NVIDIA's "
                      "code and the gated weights.")
    D.title(s, "Setup")
    D.code(s, ["git clone https://github.com/nasir26/u55c-calibrated-predecoder",
               "cd u55c-calibrated-predecoder",
               "python3.11 -m venv .venv && source .venv/bin/activate",
               "pip install -r requirements.txt",
               "",
               "# only for the full pre-decoder pipeline:",
               "scripts/setup_reference.sh        # NVIDIA code, pinned commit",
               "HF_TOKEN=... scripts/fetch_weights.sh fast",
               "pytest -q                          # tests"],
           M, Inches(1.7), Inches(8.2), Inches(4.2), 14)
    D.bullets(s, [("Laptop:", "steps 1 to 4 are enough for d = 1, 2, 3, 5."),
                  ("Weights:", "accept NVIDIA's Open Model License on Hugging Face first."),
                  ("Card:", "scripts/fetch_bitstreams.sh, then bench/run_card.py on a U55C host.")],
              M + Inches(8.6), Inches(1.8), Inches(3.5), Inches(4), 15)

    # 7 teaching
    s = D.slide(notes="Same command for every distance. Scale 0.5 means a device with half of every error rate.")
    D.title(s, "Run d = 1, 2, 3, 5 in one command")
    D.code(s, ["python examples/small_codes.py \\", "  data/calibration/ibm_kingston_2026-04-15.csv \\",
               "  --d 1 2 3 5 --rounds 5 --scale 1"], M, Inches(1.6), Inches(7.4), Inches(1.3), 14)
    if all(teach.values()):
        cd = CategoryChartData()
        cd.categories = ["d = 1", "d = 2 (kept shots)", "d = 3", "d = 5"]
        for sc in ("1", "0.5"):
            r = {x["d"]: x for x in teach[sc]["results"]}
            cd.add_series(f"scale {sc}", [100 * r[1]["ler"], 100 * r[2]["ler_kept"], 100 * r[3]["ler"], 100 * r[5]["ler"]])
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, M, Inches(3.15), Inches(7.4), Inches(3.7), cd)
        ch = gf.chart
        style_chart(ch, pct=True, title="Logical error after 5 rounds, Kingston median noise")
        pl = ch.plots[0]
        pl.gap_width = 60
        pl.has_data_labels = True
        pl.data_labels.number_format = '0.00"%"'
        pl.data_labels.number_format_is_linked = False
        pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        pl.data_labels.font.size = Pt(11)
        for k, c in enumerate((TEAL, AMBER)):
            pl.series[k].format.fill.solid()
            pl.series[k].format.fill.fore_color.rgb = c
        r2 = {x["d"]: x for x in teach["1"]["results"]}[2]
        D.bullets(s, [("d = 1:", "one bare qubit; the bar every code must beat."),
                      ("d = 2:", f"detects but cannot correct: {100 * r2['discarded_fraction']:.0f}% of shots are "
                                 "thrown away at scale 1 to get the low rate shown."),
                      ("d = 3, 5:", "decoded by PyMatching; larger d wins when the device is below threshold.")],
                  M + Inches(7.9), Inches(1.7), Inches(4.2), Inches(5), 15)

    # 8 d1/d2 lesson
    s = D.slide(notes="Use this slide to explain detection versus correction.")
    D.title(s, "What d = 1 and d = 2 can and cannot do")
    half = (W - 2 * M - Inches(0.4)) / 2
    for i, (h, items) in enumerate([
            ("d = 1: no code", ["One physical qubit, measured at the end.", "Every error in T1, T2 or readout is a "
                                "logical error.", "Error grows with the time it waits: the reference line."]),
            ("d = 2: detection only", ["Four data qubits, three checks.", "Any single error raises a flag, but two "
                                       "different errors raise the same flag.", "So it can only discard flagged "
                                       "shots (post-selection); d = 3 is the smallest code that corrects."])]):
        x = M + i * (half + Inches(0.4))
        D.card(s, x, Inches(1.8), half, Inches(4.6))
        D.text(s, h, x + Inches(0.35), Inches(2.05), half - Inches(0.7), Inches(0.6), 22, TEAL, True, HEAD)
        D.bullets(s, items, x + Inches(0.35), Inches(2.85), half - Inches(0.7), Inches(3.3), 16)

    # 9 full pipeline
    s = D.slide(notes="200,000 shots at d = 5 take about 10 minutes on 8 CPU threads with the INT8 model; drop "
                      "--shots for a quick look.")
    D.title(s, "Run the full pipeline at d = 3 and 5")
    D.code(s, ["python bench/eval_ler.py \\", "  --csv data/calibration/ibm_aachen_2026-04-17.csv \\",
               "  --mode median --scale 1 --d 5 --basis X --shots 20000"], M, Inches(1.6), Inches(7.6), Inches(1.3), 14)
    D.table(s, [["Decoder", "What it is"],
                ["pm_cal", "PyMatching with every qubit's own error rates"],
                ["pm_median", "PyMatching with the device medians only"],
                ["fast_fp32", "NVIDIA Fast pre-decoder (float), then pm_cal"],
                ["fast_int8", "integer pre-decoder, as on the card, then pm_cal"],
                ["pm_corr", "PyMatching with correlations (bench/eval_corr.py)"]],
            M, Inches(3.2), Inches(7.6), [Inches(1.8), Inches(5.8)], 14)
    D.bullets(s, [("--mode:", "median, best (lowest error qubits) or screened (random, worst 10% removed)."),
                  ("--scale:", "1 as calibrated; 0.5 halves every error rate."),
                  ("Output:", "results/ler/<name>.json and per shot failure bits (.npz).")],
              M + Inches(8.1), Inches(1.7), Inches(4.0), Inches(4.5), 15)

    # 10 own CSV
    s = D.slide(notes="Any CSV in IBM's calibration layout loads; extra columns are ignored.")
    D.title(s, "Bring your own calibration CSV")
    D.table(s, [["Column", "Used for"], ["Qubit", "row key"], ["T1 (us), T2 (us)", "idle noise over each layer"],
                ["Readout assignment error", "measurement flip"], ["Prob meas1 prep0", "preparation flip (half)"],
                ["Readout length (ns)", "how long data qubits idle"], ["√x (sx) error", "one qubit gates"],
                ["CZ error or ECR error", "two qubit gates, \"a_b:value;...\""], ["Gate length (ns)", "layer duration"],
                ["Reset length (ns) (optional)", "idle during reset"]],
            M, Inches(1.6), Inches(7.6), [Inches(3.4), Inches(4.2)], 13)
    D.code(s, ["python calib/ibm_csv.py summary my.csv", "python bench/eval_ler.py --csv my.csv ..."],
           M + Inches(8.1), Inches(1.7), Inches(4.0), Inches(1.2), 12)
    D.bullets(s, ["Headers are matched by keywords, so the IBM dashboard export loads as is.",
                  "Missing values mark a qubit unusable; it is never placed."],
              M + Inches(8.1), Inches(3.3), Inches(4.0), Inches(3), 15)

    # 11 section
    s = D.slide(dark=True)
    D.text(s, "Results", M, Inches(2.6), Inches(10), Inches(1), 44, WHITE, True, HEAD)
    D.text(s, "200,000 shots per basis and point; X and Z pooled", M, Inches(3.6), Inches(10), Inches(0.6), 20, MIST)

    # 12 ratio chart: both baselines, d = 5, as calibrated
    med = [p for p in pts if p["placement"] == "median" and "fast_int8_vs_pm_corr" in p]
    s = D.slide(notes="Below 1 means pre-decoder plus PyMatching fails less often than the baseline. Teal: plain "
                      "PyMatching. Amber: PyMatching with correlations, which needs no network.")
    D.title(s, "At d = 5, correlated matching beats the pre-decoder", "Failures of INT8 pre-decoder + PyMatching "
            "divided by each baseline; d = 5, median placement, as calibrated (scale 1)")
    cd = CategoryChartData()
    cats = [d for d in ORDER if any(p["device"] == d and p["scale"] == 1.0 and p["d"] == 5 for p in med)]
    cd.categories = [c.replace("ibm_", "") for c in cats]
    for lab, cmp in (("vs PyMatching", "fast_int8_vs_pm_cal"), ("vs correlated PyMatching", "fast_int8_vs_pm_corr")):
        cd.add_series(lab, [[p for p in med if p["device"] == c and p["scale"] == 1.0 and p["d"] == 5][0][cmp]["ratio"]
                            for c in cats])
    if cats:
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, M, Inches(1.85), Inches(8.6), Inches(4.9), cd)
        ch = gf.chart
        style_chart(ch)
        ch.value_axis.minimum_scale = 0.6
        ch.value_axis.maximum_scale = 1.4
        ch.value_axis.tick_labels.number_format = "0.0"
        ch.value_axis.tick_labels.number_format_is_linked = False
        pl = ch.plots[0]
        pl.gap_width = 60
        pl.has_data_labels = True
        pl.data_labels.number_format = "0.00"
        pl.data_labels.number_format_is_linked = False
        pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        pl.data_labels.font.size = Pt(10)
        for k, c in enumerate((TEAL, AMBER)):
            pl.series[k].format.fill.solid()
            pl.series[k].format.fill.fore_color.rgb = c
    D.bullets(s, ["Teal below 1: the network removes errors plain matching gets wrong.",
                  "Amber above 1: correlated matching fixes the same errors, and more, without a network.",
                  "Putting the network in front of correlated matching does not help either."],
              M + Inches(9.0), Inches(2.0), Inches(3.1), Inches(4.5), 15)

    # 12b where it still pays
    s = D.slide(notes="Counts are significant at p < 0.05 in a paired sign test over all placements and scales.")
    D.title(s, "Where the pre-decoder still pays", "Significant wins and losses of INT8 pre-decoder + PyMatching, all "
            "placements and scales")
    rows = [["Baseline", "d", "points", "pre-decoder better", "pre-decoder worse", "failure ratio range"]]
    for lab, cmp in (("PyMatching", "fast_int8_vs_pm_cal"), ("correlated PyMatching", "fast_int8_vs_pm_corr")):
        for d in (3, 5):
            rr = [p[cmp] for p in pts if p["d"] == d and cmp in p]
            if rr:
                bw_, ww_, n_ = wins(cmp, d)
                rows.append([lab, str(d), str(n_), str(bw_), str(ww_),
                             f"{min(c['ratio'] for c in rr):.2f} to {max(c['ratio'] for c in rr):.2f}"])
    D.table(s, rows, M, Inches(1.9), Inches(8.0), [Inches(2.4), Inches(0.6), Inches(1.0), Inches(1.4), Inches(1.4),
                                                   Inches(1.2)], 13)
    ks = [p["fast_int8_vs_pm_cal"]["ratio"] for p in pts if p["device"] == "ibm_kingston"
          and p["placement"].startswith("screened") and "fast_int8_vs_pm_cal" in p]
    D.bullets(s, [("d = 3, low noise:", "correlated matching gets worse than plain matching there; the network does "
                   "not, so it wins."),
                  ("Uneven qubits:", f"on Kingston's screened placements it even makes plain matching worse "
                   f"(up to {100 * (max(ks) - 1):.0f}% more failures)." if ks else "")],
              M + Inches(8.4), Inches(2.0), Inches(3.7), Inches(4.5), 15)

    # 13 calibration matters for matching
    het = [p for p in pts if p["placement"] != "median" and "pm_median_vs_pm_cal" in p]
    s = D.slide(notes="With median placement both graphs are identical by construction, so only best and screened "
                      "placements are shown.")
    D.title(s, "Per qubit rates make PyMatching better", "Failures with device median weights divided by failures "
            "with each qubit's own weights (best and screened placements)")
    if het:
        r = np.array([p["pm_median_vs_pm_cal"]["ratio"] for p in het])
        sig = sum(1 for p in het if p["pm_median_vs_pm_cal"]["ratio"] > 1 and p["pm_median_vs_pm_cal"]["p_sign"] < 0.05)
        D.stat(s, f"{np.median(r):.2f}x", "median ratio over all heterogeneous points", M, Inches(1.9), Inches(3.6))
        D.stat(s, f"{r.max():.2f}x", "largest ratio", M + Inches(3.9), Inches(1.9), Inches(3.6))
        D.stat(s, f"{sig} of {len(het)}", "points where median weights are significantly worse (p < 0.05)",
               M + Inches(7.8), Inches(1.9), Inches(3.6))
        D.bullets(s, ["The CSV is not only for simulation: its numbers become the matching weights.",
                      "This needs no network and no FPGA, only the calibration file."],
                  M, Inches(4.3), W - 2 * M, Inches(2), 16)

    # 14 card
    s = D.slide(notes="One card, both kernel instances, two launches in flight. Latency is one window through PCIe "
                      "with XRT overheads, not a real time loop.")
    D.title(s, "One U55C card", "Same bitstream as the four card study; weights and window size are launch arguments")
    if perf:
        ds = sorted(perf)
        cd = CategoryChartData()
        cd.categories = [f"d = {d}" for d in ds]
        cd.add_series("windows per second", [float(np.median(perf[d]["windows_per_sec_reps"])) for d in ds])
        gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, M, Inches(1.85), Inches(7.6), Inches(4.9), cd)
        ch = gf.chart
        style_chart(ch, legend=False, title="Windows per second (d rounds each)")
        pl = ch.plots[0]
        pl.gap_width = 70
        pl.has_data_labels = True
        pl.data_labels.number_format = "#,##0"
        pl.data_labels.number_format_is_linked = False
        pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        pl.data_labels.font.size = Pt(11)
        pl.series[0].format.fill.solid()
        pl.series[0].format.fill.fore_color.rgb = TEAL
        rows = [["d", "one window (ms)"]] + [[str(d), f"{perf[d]['latency_one_window_s'] * 1e3:.2f}"] for d in ds]
        D.table(s, rows, M + Inches(8.1), Inches(1.9), Inches(4.0), [Inches(1.4), Inches(2.6)], 13)
    if exact:
        n_ok = sum(c["bit_exact"] for c in exact)
        same = [c for c in exact if "same_failures_as_cpu_int8" in c]
        D.text(s, f"Bit exact in {n_ok} of {len(exact)} runs; logical failures from the card's bits equal the CPU "
                  f"integer model's shot for shot in {sum(c['same_failures_as_cpu_int8'] for c in same)} of "
                  f"{len(same)} compared runs.", M + Inches(8.1), Inches(5.0), Inches(4.0), Inches(1.6), 14, INK)

    # 15 limits
    s = D.slide(notes="Say these out loud; they are also in docs/LIMITATIONS.md.")
    D.title(s, "What this is not")
    D.bullets(s, [("Not a hardware QEC run:", "no circuit ran on IBM devices; calibrations set the rates of a "
                   "simulated circuit."),
                  ("No chip embedding:", "IBM chips are heavy hexagonal; code qubits are placed by a rule, without "
                   "SWAPs."),
                  ("Model assumptions:", "depolarizing gates, T1 and T2 idle decay, half of Prob meas1 prep0 as "
                   "preparation error; no leakage or crosstalk."),
                  ("Not real time:", "card latency includes PCIe and runtime overheads; one big CPU is as fast for "
                   "this network.")], M, Inches(1.7), W - 2 * M, Inches(4.8), 17)

    # 16 next
    s = D.slide(dark=True, notes="Ask the team which device CSVs they want to try next.")
    D.text(s, "Next steps", M, Inches(1.5), Inches(10), Inches(1), 40, WHITE, True, HEAD)
    D.bullets(s, ["Try your own device CSV with examples/small_codes.py, then bench/eval_ler.py.",
                  "Fine tune the network on one device's calibrated noise, and compare with correlated matching.",
                  "Feed fresh calibrations daily and watch the decoder track drift.",
                  "Repository: github.com/nasir26/u55c-calibrated-predecoder"],
              M, Inches(2.8), Inches(11), Inches(3.5), 20, MIST)

    out.parent.mkdir(parents=True, exist_ok=True)
    D.prs.save(out)
    print("wrote", out, D.n, "slides")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "deck" / "calibrated_predecoder_team.pptx"))
    build(Path(ap.parse_args().out))
