"""Fig. 4 (fig_v3_11/out/fig_D.pdf): deep model minus RF (len+IAT), macro-F1 pp, 95% CIs.

(a) per split on the same split axis as Fig. 3, all classes (filled, solid) and without the email
    term (open, dashed); (b) under P*, with the F1 term of one class omitted: only intervals that
    exclude zero keep their model colour, the rest fade to grey.
Sources: paperA/gen/tab_rank.tex, paperA/gen/facts.json (loco_Pstar_deep_minus_RF);
every plotted value is checked against gen/numbers.tex.
"""
from __future__ import annotations

import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figstyle as fs
from figstyle import plt

fs.setup()
nums = fs.read_numbers()
facts = fs.read_facts()

# ---------------------------------------------------------------- (a) data
body = fs.tabular_body(fs.GEN / "tab_rank.tex")
SPLIT_ROW = {"Flow-random": "leak", "File-grouped": "no_session",
             "Suffix session (P*)": "Pstar", "Evidence session": "evidence"}
COLS = [("Fu", "all"), ("Fu", "ne"), ("Cn", "all"), ("Cn", "ne")]
RANK = {}
for i, r in enumerate(body):
    if r[0] in SPLIT_ROW:
        ci = body[i + 1]
        assert ci[0] == "" and len(r) == 5 and len(ci) == 5, (r, ci)
        for j, key in enumerate(COLS):
            lo, hi = (fs.tex_num(x) for x in ci[j + 1].strip("[]").split(","))
            RANK[(SPLIT_ROW[r[0]],) + key] = (fs.tex_num(r[j + 1]), lo, hi)
assert len(RANK) == 16, len(RANK)
MAC = {"leak": "Flow", "no_session": "File", "Pstar": "Suffix", "evidence": "Evid"}
for (c, m, v), (d, lo, hi) in RANK.items():
    assert lo <= d <= hi, (c, m, v)
    name = f"R{m}{MAC[c]}" + ("NE" if v == "ne" else "")
    for suf, val in (("", d), ("Lo", lo), ("Hi", hi)):
        fs.assert_macro(nums, name + suf, val)
assert RANK["Pstar", "Fu", "all"] == (0.6, -2.9, 3.4)
assert RANK["Pstar", "Cn", "all"] == (-1.5, -6.7, 2.4)

# ---------------------------------------------------------------- (b) data
loco = facts["loco_Pstar_deep_minus_RF"]
assert facts["loco_mode"] == {"strat": True, "drop": "term"} and facts["n_boot"] == 10000
assert set(loco) == set(fs.CLASS_SHORT), sorted(loco)
for cls, d in loco.items():
    for m in ("Fu", "Cn"):
        est, lo, hi = d[m]
        assert lo <= est <= hi, (cls, m)
assert loco["email"]["Fu"] == [-5.56, -7.78, -2.66] and loco["email"]["Cn"] == [-8.63, -12.2, -4.43]
for m in ("Fu", "Cn"):
    est, lo, hi = loco["vpn_file_transfer"][m]
    for suf, val in (("", est), ("Lo", lo), ("Hi", hi)):
        fs.assert_macro(nums, f"LocoVft{m}{suf}", val)
decisive = sorted(c for c, d in loco.items() if any(d[m][1] > 0 or d[m][2] < 0 for m in ("Fu", "Cn")))
assert decisive == sorted(facts["loco_Pstar_classes_whose_removal_excludes_zero"]) == \
    ["email", "vpn_file_transfer"], decisive
assert int(nums["NLocoDecisive"]) == len(decisive)
# the "none" row equals (a) P*, all classes, and no all-class P* interval excludes zero
for m in ("Fu", "Cn"):
    _, lo, hi = RANK["Pstar", m, "all"]
    assert lo < 0 < hi

# row order = Fig. 2 (suffix sessions ascending, then name)
spc = facts["sessions_per_class"]
ORDER = sorted(loco, key=lambda c: (spc[c]["suffix"], fs.CLASS_SHORT[c]))
assert [fs.CLASS_SHORT[c] for c in ORDER] == ["Email", "V-FT", "V-Stream", "V-Chat", "V-VoIP",
                                               "Chat", "Stream", "VoIP", "FT"]



def excl0(lo, hi):
    return lo > 0 or hi < 0


# focus set of (b): exactly the intervals that exclude zero
FOCUS = {(c, m) for c in loco for m in ("Fu", "Cn") if excl0(*loco[c][m][1:])}
assert FOCUS == {("email", "Fu"), ("email", "Cn"), ("vpn_file_transfer", "Cn")}, FOCUS
assert not any(excl0(*RANK["Pstar", m, "all"][1:]) for m in ("Fu", "Cn"))

# ---------------------------------------------------------------- figure
W, H = 4.80, 2.00
fig = plt.figure(figsize=(W, H))
BOT, PH = 0.36, 1.26
FU, CN = fs.MODELS["Fu"], fs.MODELS["Cn"]
rows_out = []


def marker(ax, x, y, st, filled, color=None, z=3):
    col = color or st["color"]
    ax.plot([x], [y], marker=st["marker"], ms=st["ms"] - (0.3 if filled else 1.1), ls="none", zorder=z,
            mfc=col if filled else "white", mec="white" if filled else col, mew=0.5 if filled else 0.9)


# (a) one small panel per deep model: all classes vs. without the email term -------
# The fill between the two lines is what the email term alone contributes; it opens up under the
# session splits, which is where the ranking flips.
for m in ("Fu", "Cn"):
    gap = [RANK[c, m, "all"][0] - RANK[c, m, "ne"][0] for c in fs.SPLITS]
    assert min(gap[2:]) > max(abs(g) for g in gap[:2]), (m, gap)          # widest under session splits
    assert all(not excl0(*RANK[c, m, "all"][1:]) for c in ("Pstar", "evidence")), m
    assert all(RANK[c, m, "ne"][2] < 0 for c in ("Pstar", "evidence")), m   # w/o email: RF leads
AX0, AW, AG = 0.30, 0.97, 0.13
XS = np.arange(4)
DX = {"all": -0.1, "ne": 0.1}
for k, m in enumerate(("Fu", "Cn")):
    st = fs.MODELS[m]
    ax = fs.axes_in(fig, AX0 + k * (AW + AG), BOT, AW, PH)
    ax.patch.set_alpha(0)
    fs.session_band(ax, 1.5, 3.5)
    for y in (-20, -10):
        ax.axhline(y, color=fs.GRID, lw=0.5, zorder=0.5)
    ax.axhline(0, color=fs.ZERO, lw=0.6, zorder=1)
    est = {v: np.array([RANK[c, m, v][0] for c in fs.SPLITS]) for v in DX}
    xx = np.linspace(XS[0] + DX["all"], XS[-1] + DX["ne"], 401)
    ya, yn = np.interp(xx, XS + DX["all"], est["all"]), np.interp(xx, XS + DX["ne"], est["ne"])
    ax.fill_between(xx, yn, ya, where=ya >= yn, interpolate=True, color=st["color"], alpha=0.2, lw=0,
                    zorder=1.5)
    for v, dx in DX.items():
        ax.plot(XS + dx, est[v], color=st["color"], lw=0.9, ls="-" if v == "all" else (0, (2.4, 1.6)),
                zorder=2)
        for j, c in enumerate(fs.SPLITS):
            e, lo, hi = RANK[c, m, v]
            ax.plot([j + dx, j + dx], [lo, hi], color=st["color"], lw=0.9, solid_capstyle="butt", zorder=2.5)
            marker(ax, j + dx, e, st, v == "all")
            rows_out.append(dict(panel="a", row=fs.SPLIT_LONG[c], model=st["long"],
                                 classes="all" if v == "all" else "w/o email", est=e, lo=lo, hi=hi,
                                 ci_excludes_0=excl0(lo, hi)))
    ax.set_xlim(-0.45, 3.5)
    ax.set_ylim(-22.6, 4.4)
    # horizontal fill key (solid top / shaded fill / dashed bottom + upright "Email"), aligned at
    # the same clean baseline in both panels inside the open bottom of the session-split band
    y_key = -16.5
    sw_w, sw_h, gap = 0.42, 1.65, 0.14
    t_tmp = ax.text(2.5, y_key, "Email", ha="left", va="center",
                    fontsize=fs.FS_SMALL, color=fs.TEXT, zorder=4)
    fig.canvas.draw()
    tw = t_tmp.get_window_extent().transformed(ax.transData.inverted()).width
    x_sw = 2.5 - (sw_w + gap + tw) / 2
    t_tmp.set_position((x_sw + sw_w + gap, y_key))
    ax.add_patch(Rectangle((x_sw, y_key - sw_h / 2), sw_w, sw_h,
                           facecolor=st["color"], alpha=0.2, edgecolor="none", zorder=3))
    ax.plot([x_sw, x_sw + sw_w], [y_key + sw_h / 2, y_key + sw_h / 2],
            color=st["color"], lw=0.85, solid_capstyle="butt", zorder=3.2)
    ax.plot([x_sw, x_sw + sw_w], [y_key - sw_h / 2, y_key - sw_h / 2],
            color=st["color"], lw=0.85, ls=(0, (1.8, 1.1)), solid_capstyle="butt", zorder=3.2)
    fig.canvas.draw()
    bb = t_tmp.get_window_extent().transformed(ax.transData.inverted())
    min_lo_sess = min(RANK[c, m, v][1] for c in ("Pstar", "evidence") for v in DX)
    assert 1.5 + 0.15 < x_sw and bb.x1 < 3.5 - 0.15, (m, x_sw, bb)                 # centred in session band
    assert max(bb.y1, y_key + sw_h / 2) < min_lo_sess - 2.0, (m, bb, min_lo_sess)  # clear of session whiskers
    assert min(bb.y0, y_key - sw_h / 2) > -20.0 + 1.5, (m, bb)                     # clear above -20 gridline
    assert min(RANK[c, m, v][1] for c in fs.SPLITS for v in DX) > -22.6 + 0.3
    assert max(RANK[c, m, v][2] for c in fs.SPLITS for v in DX) < 4.4 - 0.8
    ax.set_yticks([-20, -10, 0])
    if k:
        ax.set_yticklabels([])
    ax.set_xticks(range(4), [fs.SPLIT_SHORT[c] for c in fs.SPLITS])
    ax.tick_params(axis="x", length=0, pad=2.5, labelsize=fs.FS_SMALL)
    ax.tick_params(axis="y", length=0, pad=1.5)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.4, 4.0, st["name"], ha="left", va="top", fontsize=fs.FS_SMALL, color=fs.TEXT,
            fontweight="bold")
    if k == 0:
        ax.text(-0.4, -1.3, "RF better ↓", ha="left", va="top", fontsize=fs.FS_SMALL, color=fs.MUTED,
                style="italic")

# (b) leave one class out, focus + context -------------------------------------
axb = fs.axes_in(fig, 2.92, BOT, 1.82, PH)
labels = ["None"] + [fs.CLASS_SHORT[c] for c in ORDER]
keys = [None] + ORDER
n = len(labels)
for i, (lab, c) in enumerate(zip(labels, keys)):
    y = n - 1 - i
    for m, dy in (("Fu", 0.17), ("Cn", -0.17)):
        st = fs.MODELS[m]
        est, lo, hi = RANK["Pstar", m, "all"] if c is None else loco[c][m]
        focus = c is not None and (c, m) in FOCUS
        col = st["color"] if focus else fs.FADE
        axb.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.3 if focus else 0.8, solid_capstyle="butt",
                 zorder=3 if focus else 2)
        marker(axb, est, y + dy, st, True, color=col, z=4 if focus else 2.5)
        rows_out.append(dict(panel="b", row=lab, model=st["long"],
                             classes="all" if c is None else f"w/o {lab}", est=est, lo=lo, hi=hi,
                             ci_excludes_0=excl0(lo, hi)))
axb.axhline(n - 1.5, color=fs.GRID, lw=0.5, zorder=0)
axb.axvline(0, color=fs.ZERO, lw=0.6, zorder=1)
axb.set_ylim(-0.6, n - 0.4)
axb.set_yticks(range(n - 1, -1, -1), labels)
for t, c in zip(axb.get_yticklabels(), keys):
    if c is None:
        t.set_fontstyle("italic")
        t.set_color(fs.MUTED)
    elif c in decisive:
        t.set_fontweight("bold")
        t.set_color(fs.TEXT)
    else:
        t.set_color(fs.FADE_TEXT)
axb.set_xlim(-14, 8)
axb.set_xticks([-10, -5, 0, 5])
axb.spines["left"].set_visible(False)
axb.tick_params(axis="y", length=0, pad=2.0)
axb.text(-0.5, -0.19, "← RF better", transform=axb.get_xaxis_transform(), ha="right", va="top",
         fontsize=fs.FS_SMALL, color=fs.MUTED, style="italic")
axb.text(0.5, -0.19, "deep better →", transform=axb.get_xaxis_transform(), ha="left", va="top",
         fontsize=fs.FS_SMALL, color=fs.MUTED, style="italic")


# key row, then headings -------------------------------------------------------------
def mk(st, filled):
    return Line2D([], [], ls="none", marker=st["marker"], ms=st["ms"] - (0.3 if filled else 1.1),
                  mfc=st["color"] if filled else "white", mec="white" if filled else st["color"],
                  mew=0.5 if filled else 0.9)


def model(st):
    return Line2D([], [], color=st["color"], lw=0.9, marker=st["marker"], ms=st["ms"] - 0.3,
                  mec="white", mew=0.5)


grey = Line2D([], [], color=fs.FADE, lw=0.9, marker="s", ms=3.0, mec="white", mew=0.4)
fig.legend([model(FU), model(CN), (mk(FU, True), mk(CN, True)), (mk(FU, False), mk(CN, False)), grey],
           ["Fusion", "1D-CNN", "all classes", "w/o Email", "CI covers 0"],
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
           loc="center", bbox_to_anchor=(0.5, (H - 0.085) / H), ncol=5, frameon=False,
           fontsize=fs.FS_SMALL, handlelength=2.0, handletextpad=0.5, columnspacing=1.9,
           borderaxespad=0, borderpad=0)
top = H - 0.205
fs.panel_heading(fig, 0.02, top, "(a)", "Deep − RF (pp) by split")
fs.panel_heading(fig, 2.40, top, "(b)", "Under P*, one class omitted")

fs.save(fig, "fig_D", rows_out)
