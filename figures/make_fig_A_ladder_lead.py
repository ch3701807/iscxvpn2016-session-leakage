"""Fig. 3 (fig_v3_11/out/fig_A.pdf): the leakage ladder and RF's vanishing lead.

(a) macro-F1 by split; the tinted wedge is RF's lead over Fusion (the stronger deep model), which
    closes at P* (all classes). (b) difference vs. P* (split minus P*) with 95% paired cluster-bootstrap CIs.
Sources: paperA/gen/ladder.csv, paperA/gen/tab_rank.tex; every printed number is a numbers.tex macro.
"""
from __future__ import annotations

import csv

import numpy as np

import figstyle as fs
from figstyle import plt

fs.setup()
nums = fs.read_numbers()

# ---------------------------------------------------------------- data + checks
rows = list(csv.DictReader((fs.GEN / "ladder.csv").open()))
assert len(rows) == 12, len(rows)
D = {(r["model"], r["condition"]): r for r in rows}
assert len(D) == 12, "duplicate (model, condition) rows"
assert {m for m, _ in D} == {"RF", "Fu", "Cn"} and {c for _, c in D} == set(fs.SPLITS)

F1 = {(m, c): float(D[m, c]["macro_f1"]) for m in fs.MODELS for c in fs.SPLITS}
DELTA = {}
for m in fs.MODELS:
    assert D[m, "Pstar"]["delta"] == "" and D[m, "Pstar"]["lo"] == "", "P* row must carry no delta"
    for c in ("leak", "no_session", "evidence"):
        d, lo, hi = (float(D[m, c][k]) for k in ("delta", "lo", "hi"))
        assert lo <= d <= hi, (m, c, d, lo, hi)
        # delta is split minus P*, computed from unrounded values
        assert abs((F1[m, c] - F1[m, "Pstar"]) - d) < 0.0015, (m, c)
        DELTA[m, c] = (d, lo, hi)

TAG = {"RF": "RF", "Fu": "Fu", "Cn": "Cn"}
LAD = {"leak": "Flow", "no_session": "File", "Pstar": "Suffix", "evidence": "Evid"}
for m in fs.MODELS:
    for c in fs.SPLITS:
        fs.assert_macro(nums, f"Lad{TAG[m]}{LAD[c]}", F1[m, c])
    for c in ("leak", "no_session", "evidence"):
        d, lo, hi = DELTA[m, c]
        fs.assert_macro(nums, f"D{TAG[m]}{LAD[c]}", d)
        fs.assert_macro(nums, f"D{TAG[m]}{LAD[c]}Lo", lo)
        fs.assert_macro(nums, f"D{TAG[m]}{LAD[c]}Hi", hi)
assert nums["LadRFFlow"] == "72.9" and nums["DRFFlow"] == "39.7" and nums["DRFFile"] == "22.9"

# RF's lead over Fusion (paired difference, tab_rank.tex): the wedge label
body = fs.tabular_body(fs.GEN / "tab_rank.tex")
fu_minus_rf = {}
for i, r in enumerate(body):
    for name, split in (("Flow-random", "leak"), ("File-grouped", "no_session"), ("Suffix session (P*)", "Pstar")):
        if r[0] == name:
            fu_minus_rf[split] = fs.tex_num(r[1])
assert fu_minus_rf == {"leak": fs.macro_float(nums, "RFuFlow"), "no_session": fs.macro_float(nums, "RFuFile"),
                       "Pstar": fs.macro_float(nums, "RFuSuffix")}
assert fu_minus_rf["leak"] == fu_minus_rf["no_session"] == -11.2 and fu_minus_rf["Pstar"] == 0.6
# Fusion is the stronger deep model under the leaky splits, so the wedge is RF minus Fusion
for c in ("leak", "no_session"):
    assert F1["Fu", c] > F1["Cn", c] and F1["RF", c] > F1["Fu", c]
    assert abs((F1["RF", c] - F1["Fu", c]) + fu_minus_rf[c]) < 0.06, c     # same lead, rounded inputs
assert F1["Fu", "Pstar"] > F1["RF", "Pstar"]                                # lead gone at P*
LEAD_TXT = f"RF +{-fu_minus_rf['leak']:.1f} pp"

# ---------------------------------------------------------------- figure
W, H = 4.80, 1.80
fig = plt.figure(figsize=(W, H))
X = np.arange(len(fs.SPLITS))
BOT, TOP_PAD = 0.33, 0.24
PH = H - BOT - TOP_PAD

# (a) ------------------------------------------------------------------------
axa = fs.axes_in(fig, 0.30, BOT, 2.30, PH)
fs.session_band(axa, 1.5, 3.45)
for y in (20, 40, 60):
    axa.axhline(y, color=fs.GRID, lw=0.5, zorder=0.5)
rf = np.array([F1["RF", c] for c in fs.SPLITS])
fu = np.array([F1["Fu", c] for c in fs.SPLITS])
xx = np.linspace(0, 2, 201)
rfi, fui = np.interp(xx, X[:3], rf[:3]), np.interp(xx, X[:3], fu[:3])
axa.fill_between(xx, fui, rfi, where=rfi >= fui, interpolate=True, color=fs.MODELS["RF"]["color"],
                 alpha=0.16, lw=0, zorder=1)
for m, st in fs.MODELS.items():
    axa.plot(X, [F1[m, c] for c in fs.SPLITS], color=st["color"], ls=st["ls"], lw=1.25,
             marker=st["marker"], ms=st["ms"], mec="white", mew=0.6, zorder=3, clip_on=False)
lab_y = {"RF": F1["RF", "leak"], "Fu": F1["Fu", "leak"] + 2.6, "Cn": F1["Cn", "leak"] - 2.6}
for m, st in fs.MODELS.items():
    axa.text(-0.16, lab_y[m], st["name"], ha="right", va="center", fontsize=fs.FS_SMALL, color=fs.TEXT)
axa.set_xlim(-0.95, 3.45)
axa.set_ylim(0, 80)
axa.set_yticks([0, 20, 40, 60, 80])
axa.set_xticks(list(X), [fs.SPLIT_2LINE[c] for c in fs.SPLITS], linespacing=1.0)
axa.tick_params(axis="x", length=0, pad=2.5)
axa.spines["left"].set_visible(False)
axa.tick_params(axis="y", length=0, pad=1.5)
# wedge label, written along the wedge
fig.canvas.draw()
xm = 0.5
ym = (np.interp(xm, X, rf) + np.interp(xm, X, fu)) / 2
p0, p1 = axa.transData.transform([(0, (rf[0] + fu[0]) / 2), (1, (rf[1] + fu[1]) / 2)])
ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
axa.text(xm, ym, LEAD_TXT, ha="center", va="center", rotation=ang, rotation_mode="anchor",
         fontsize=fs.FS_SMALL, color=fs.TEXT, fontweight="bold", zorder=4)
for x0, x1, lab in ((-0.4, 1.4, "leaky splits"), (1.6, 3.4, "session splits")):
    axa.text((x0 + x1) / 2, 5.0, lab, ha="center", va="center", fontsize=fs.FS_SMALL,
             color=fs.MUTED, style="italic")
fs.panel_heading(fig, 0.02, H - 0.05, "(a)", "Macro-F1 (%)")

# (b) ------------------------------------------------------------------------
axb = fs.axes_in(fig, 3.06, BOT, 1.68, PH)
fs.session_band(axb, 1.5, 3.45)
for y in (20, 40):
    axb.axhline(y, color=fs.GRID, lw=0.5, zorder=0.5)
OFF = {"RF": -0.24, "Fu": 0.0, "Cn": 0.24}
BW = 0.2
data_rows = []
for m, st in fs.MODELS.items():
    for i, c in enumerate(fs.SPLITS):
        if c == "Pstar":
            continue
        d, lo, hi = DELTA[m, c]
        x = i + OFF[m]
        axb.bar(x, d, width=BW, color=st["color"], lw=0, zorder=2)
        axb.plot([x, x], [lo, hi], color=fs.ANNOT, lw=0.7, solid_capstyle="butt", zorder=3)
        data_rows.append(dict(model=st["long"], split=fs.SPLIT_LONG[c], macro_f1=F1[m, c],
                              delta_vs_Pstar=d, ci_lo=lo, ci_hi=hi))
    data_rows.append(dict(model=st["long"], split=fs.SPLIT_LONG["Pstar"], macro_f1=F1[m, "Pstar"],
                          delta_vs_Pstar="", ci_lo="", ci_hi=""))
axb.axhline(0, color=fs.ZERO, lw=0.6, zorder=4)
for c, xo in (("leak", 0), ("no_session", 1)):
    d, lo, hi = DELTA["RF", c]
    txt = fs.fmt_signed(d)
    assert txt.replace("−", "-").lstrip("+") == nums["DRFFlow" if c == "leak" else "DRFFile"]
    axb.text(xo + OFF["RF"], hi + 1.4, txt, ha="center", va="bottom", fontsize=fs.FS_SMALL, color=fs.TEXT)
axb.text(2.0, 1.5, "P*", ha="center", va="bottom", fontsize=fs.FS_SMALL, color=fs.MUTED)
axb.set_xlim(-0.55, 3.45)
axb.set_ylim(-10, 50)
axb.set_yticks([0, 20, 40])
axb.set_xticks(list(X), [fs.SPLIT_2LINE[c] for c in fs.SPLITS], linespacing=1.0)
axb.tick_params(axis="x", length=0, pad=2.5)
for sp in ("left", "bottom"):
    axb.spines[sp].set_visible(False)
axb.tick_params(axis="y", length=0, pad=1.5)
fs.panel_heading(fig, 2.80, H - 0.05, "(b)", "Difference vs. P* (pp), 95% CI")

fs.save(fig, "fig_A", data_rows)
