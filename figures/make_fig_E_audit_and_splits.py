"""Fig. 1 (fig_v3_11/out/fig_E.pdf): why split units matter, then what each one removes.

(a) Measured: byte-identical packets (% of the smaller file) in each of the 33 sibling a/b pairs,
    sorted, with time overlap marked below (source: gpu_returns/G01_final/.../sibling_pairs.csv,
    the audit table behind Section 3; every count is checked against numbers.tex).
(b) Synthetic schematic: the four split units on two sessions, one orange arc per leakage kind.
"""
from __future__ import annotations

import csv

from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

import figstyle as fs
from figstyle import plt

fs.setup()
nums = fs.read_numbers()

# ---------------------------------------------------------------- (a) audit data + checks
rows = list(csv.DictReader(fs.SIBLING_CSV.open(newline="")))
suf = [r for r in rows if "suffix" in r["candidate_origins"]]
assert len(suf) == int(nums["NSuffixPairs"]) == 33, len(suf)
PAIRS = []
for r in suf:
    PAIRS.append(dict(
        file_a=r["file_a"], file_b=r["file_b"], cls=r["class"], time=r["time_relation"],
        shared=int(r["common_tcp_five_tuples"]) + int(r["common_udp_five_tuples"]),
        split_tcp=int(r["split_tcp_connections"]), ident_n=int(r["identical_packet_count"]),
        ident_pct=100.0 * float(r["identical_packet_fraction_min"])))
TIME_RANK = {"overlap": 0, "adjacent": 1, "separate": 2}
assert {p["time"] for p in PAIRS} <= set(TIME_RANK)
cnt = {k: sum(p["time"] == k for p in PAIRS) for k in TIME_RANK}
assert cnt["overlap"] == int(nums["NSuffixOverlap"]) == 28
assert cnt["adjacent"] == int(nums["NSuffixAdjacent"]) == 2
assert cnt["separate"] == int(nums["NSuffixSeparate"]) == 3
ident = [p for p in PAIRS if p["ident_n"] > 0]
assert len(ident) == int(nums["NSuffixIdent"]) == 13
assert all(p["time"] == "overlap" for p in ident)          # identical packets only in overlapping pairs
top = max(p["ident_pct"] for p in PAIRS)
assert f"{top:.1f}" == nums["SuffixIdentMax"] == "84.6"
assert sum(p["ident_pct"] >= 40 for p in PAIRS) == int(nums["NSuffixIdentForty"]) == 8
assert sum(p["shared"] >= 1 for p in PAIRS) == int(nums["NSuffixShared"]) == 18
assert sum(p["split_tcp"] >= 1 for p in PAIRS) == int(nums["NSuffixSplitTCP"]) == 2
PAIRS.sort(key=lambda p: (-p["ident_pct"], TIME_RANK[p["time"]], -p["shared"], p["file_a"]))
assert [p["ident_n"] > 0 for p in PAIRS] == [True] * 13 + [False] * 20

# ---------------------------------------------------------------- (b) schematic spec + checks
FILES = ["a", "b", "c", "d"]
SESSION = {"a": "S1", "b": "S1", "c": "S1", "d": "S2"}
T, t = "train", "test"
ROWS = [
    ("leak", None,
     {"a": [T, t, T], "b": [t, T, t], "c": [T, t, T], "d": [T, t, T]},
     [(("a", 2), ("b", 0)), (("c", 0), ("b", 2)), (("d", 0), ("d", 1))]),
    ("no_session", [["a"], ["b"], ["c"], ["d"]],
     {"a": [T] * 3, "b": [t] * 3, "c": [T] * 3, "d": [t] * 3},
     [(("a", 2), ("b", 0)), (("c", 0), ("b", 2))]),
    ("Pstar", [["a", "b"], ["c"], ["d"]],
     {"a": [T] * 3, "b": [T] * 3, "c": [t] * 3, "d": [t] * 3},
     [(("b", 2), ("c", 0))]),
    ("evidence", [["a", "b", "c"], ["d"]],
     {"a": [T] * 3, "b": [T] * 3, "c": [T] * 3, "d": [t] * 3},
     []),
]


def kind(f1, f2):
    if f1 == f2:
        return "intra-file"
    if {f1, f2} == {"a", "b"}:
        return "cross-sibling"
    return "cross-stem"


EXPECTED = [{"intra-file", "cross-sibling", "cross-stem"}, {"cross-sibling", "cross-stem"},
            {"cross-stem"}, set()]
assert [s for s, *_ in ROWS] == fs.SPLITS
for (split, units, roles, arcs), want in zip(ROWS, EXPECTED):
    if units is not None:
        assert sorted(f for u in units for f in u) == FILES
        for u in units:
            assert len({r for f in u for r in roles[f]}) == 1, (split, u)
            assert len({SESSION[f] for f in u}) == 1, (split, u)
    present = {kind(f1, f2) for f1 in FILES for f2 in FILES
               for r1 in roles[f1] for r2 in roles[f2]
               if r1 == t and r2 == T and SESSION[f1] == SESSION[f2]}
    drawn = []
    for (f1, i1), (f2, i2) in arcs:
        assert SESSION[f1] == SESSION[f2], (split, f1, f2)
        assert {roles[f1][i1], roles[f2][i2]} == {T, t}, (split, f1, i1, f2, i2)
        drawn.append(kind(f1, f2))
    assert sorted(drawn) == sorted(want) == sorted(present), (split, drawn, present)

# ---------------------------------------------------------------- geometry (pt; 1 data unit = 1 pt)
W_IN, H_IN = 4.80, 1.88
WP, HP = W_IN * 72, H_IN * 72
B_TOP = 97.92                        # height of the (b) block: identical to the restrained Fig. 1
DY = B_TOP - 93.6
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WP)
ax.set_ylim(0, HP)
ax.set_aspect("equal")
ax.axis("off")

CAP_W, CAP_H, R = 52.0, 8.6, 2.15
X0 = {"a": 72.0, "b": 134.0, "c": 196.0, "d": 276.0}
ROW_Y = [66.5 + DY, 50.5 + DY, 34.5 + DY, 18.5 + DY]
PAD = 2.3
INK = "#3a5170"
UNIT_FILL = "#e1ebf7"
CAP_EDGE = "#a9b1bc"
X_L, X_R = X0["a"], X0["d"] + CAP_W          # (a) bars share the file grid of (b)
LABEL_X = 62.0
renderer = fig.canvas.get_renderer()


def text_w(s, size=fs.FS_SMALL):
    tt = ax.text(0, 0, s, fontsize=size)
    w = tt.get_window_extent(renderer).width * 72.0 / fig.dpi
    tt.remove()
    return w


def tag(y, letter, title):
    t1 = ax.text(1.5, y, f"({letter})", ha="left", va="center", fontsize=fs.FS_PANEL, fontweight="bold")
    w = t1.get_window_extent(renderer).width * 72.0 / fig.dpi
    ax.text(1.5 + w + 3.0, y, title, ha="left", va="center", fontsize=fs.FS)


# ---------------------------------------------------------------- (a)
Y0 = B_TOP + 13.2                    # bar baseline
SCALE = 18.0 / 100.0                 # pt per percentage point
YD = Y0 - 4.6                        # time-overlap dots
pitch = (X_R - X_L) / len(PAIRS)
STUB = 1.1
bar_w = pitch * 0.68
rows_out = []
ax.plot([X_L, X_R], [Y0, Y0], color=fs.AXIS, lw=0.5, solid_capstyle="butt", zorder=1)
for i, p in enumerate(PAIRS):
    xc = X_L + pitch * (i + 0.5)
    h = p["ident_pct"] * SCALE
    if p["ident_n"] > 0:             # below 1 %, a fixed stub marks "some identical packets"
        ax.add_patch(Rectangle((xc - bar_w / 2, Y0), bar_w, max(h, STUB) if p["ident_pct"] < 1 else h,
                               lw=0, facecolor=fs.LEAK, zorder=2))
    over = p["time"] == "overlap"
    ax.add_patch(Circle((xc, YD), 1.45, facecolor=INK if over else "white", edgecolor=INK, lw=0.6,
                        zorder=3))
    rows_out.append(dict(panel="a", rank=i + 1, file_a=p["file_a"], file_b=p["file_b"], cls=p["cls"],
                         time_relation=p["time"], identical_pct=round(p["ident_pct"], 3),
                         shared_five_tuples=p["shared"], split_tcp=p["split_tcp"]))
x1 = X_L + pitch * 0.5
ax.text(x1, Y0 + top * SCALE + 1.2, f"{top:.1f}%", ha="center", va="bottom",
        fontsize=fs.FS_SMALL, color=fs.TEXT)
# summary to the right of the 13 pairs with identical packets (counts are numbers.tex macros)
xs = X_L + pitch * len(ident) + 5.0                         # right after the 13 pairs
ax.text(xs, Y0 + 10.6, f"{len(ident)} of {len(PAIRS)} pairs share byte-identical packets",
        ha="left", va="center", fontsize=fs.FS_SMALL, color=fs.TEXT)
ax.text(xs, Y0 + 3.8, f"{cnt['overlap']} of {len(PAIRS)} overlap in time", ha="left", va="center",
        fontsize=fs.FS_SMALL, color=fs.TEXT)
assert xs + text_w(f"{len(ident)} of {len(PAIRS)} pairs share byte-identical packets") < WP - 2
tag(HP - 6.5, "a", "Sibling pairs")
ax.text(LABEL_X, Y0 + 6.0, "identical packets", ha="right", va="center", fontsize=fs.FS_SMALL,
        color=fs.MUTED)
ax.text(LABEL_X, YD, "time overlap", ha="right", va="center", fontsize=fs.FS_SMALL, color=fs.MUTED)
assert Y0 + top * SCALE + 7.5 < HP - 0.3
assert YD - 1.45 > B_TOP - 2.0


# ---------------------------------------------------------------- (b)
def dot_x(f, i):
    return X0[f] + CAP_W * (0.22, 0.50, 0.78)[i]


def arc(x1_, x2_, y, lift):
    verts = [(x1_, y), ((x1_ + x2_) / 2, y + 2 * lift), (x2_, y)]
    return PathPatch(MPath(verts, [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]), fill=False,
                     edgecolor=fs.LEAK, lw=1.0, capstyle="round", zorder=5)


yb = 84.0 + DY
for xs_, xe, lab in ((X0["a"], X0["c"] + CAP_W, "Session S1"), (X0["d"], X0["d"] + CAP_W, "Session S2")):
    ax.plot([xs_, xs_, xe, xe], [yb - 2.2, yb, yb, yb - 2.2], color=fs.AXIS, lw=0.6,
            solid_joinstyle="miter")
    ax.text((xs_ + xe) / 2, yb + 1.6, lab, ha="center", va="bottom", fontsize=fs.FS_SMALL,
            color=fs.TEXT)
for f in FILES:
    ax.text(X0[f] + CAP_W / 2, 76.4 + DY, f"file {f}", ha="center", va="center", fontsize=fs.FS_SMALL,
            color=fs.MUTED)
tag(yb + 4.6, "b", "Split units")

for (split, units, roles, arcs), yc in zip(ROWS, ROW_Y):
    ax.text(LABEL_X, yc, fs.SPLIT_LONG[split], ha="right", va="center", fontsize=fs.FS_SMALL)
    for u in units or []:
        xs_, xe = X0[u[0]] - PAD, X0[u[-1]] + CAP_W + PAD
        ax.add_patch(FancyBboxPatch((xs_, yc - CAP_H / 2 - PAD), xe - xs_, CAP_H + 2 * PAD,
                                    boxstyle="round,pad=0,rounding_size=4.0", lw=0,
                                    facecolor=UNIT_FILL, zorder=1))
    for f in FILES:
        ax.add_patch(FancyBboxPatch((X0[f], yc - CAP_H / 2), CAP_W, CAP_H,
                                    boxstyle=f"round,pad=0,rounding_size={CAP_H / 2}",
                                    lw=0.5, edgecolor=CAP_EDGE, facecolor="white", zorder=2))
        for i, role in enumerate(roles[f]):
            ax.add_patch(Circle((dot_x(f, i), yc), R, zorder=4,
                                facecolor=INK if role == T else "white", edgecolor=INK, lw=0.8))
            rows_out.append(dict(panel="b", rank="", file_a=f"{SESSION[f]}:{f}", file_b=i + 1,
                                 cls=fs.SPLIT_LONG[split], time_relation=role, identical_pct="",
                                 shared_five_tuples="", split_tcp="synthetic"))
    for (f1, i1), (f2, i2) in arcs:
        xa, xb = sorted((dot_x(f1, i1), dot_x(f2, i2)))
        ax.add_patch(arc(xa, xb, yc + R + 0.3, 3.0 if f1 == f2 else 4.1))

# legend for (b): one row, measured widths, equal gaps, centred under the file grid
YL, ICON_GAP, ITEM_GAP, ICON_W = 6.0, 3.6, 16.0, 11.0
DOT_W = 2 * R + 0.8


def icon_dot(x, filled):
    ax.add_patch(Circle((x + DOT_W / 2, YL), R, facecolor=INK if filled else "white",
                        edgecolor=INK, lw=0.8))


def icon_arc(x):
    ax.add_patch(arc(x, x + ICON_W, YL - 1.2, 2.4))


def icon_unit(x):
    ax.add_patch(FancyBboxPatch((x, YL - 3.4), ICON_W, 6.8, boxstyle="round,pad=0,rounding_size=2.2",
                                lw=0, facecolor=UNIT_FILL))


items = [(lambda x: icon_dot(x, True), DOT_W, "train flow"),
         (lambda x: icon_dot(x, False), DOT_W, "test flow"),
         (icon_arc, ICON_W, "same-session leakage"), (icon_unit, ICON_W, "split unit")]
widths = [iw + ICON_GAP + text_w(lab) for _, iw, lab in items]
total = sum(widths) + ITEM_GAP * (len(items) - 1)
x = (X_L - PAD + X_R + PAD) / 2 - total / 2
assert x > 2 and x + total < WP - 2, (x, total)
assert YL + 4.0 < ROW_Y[-1] - CAP_H / 2 - PAD - 3.0
for (draw, iw, lab), w in zip(items, widths):
    draw(x)
    ax.text(x + iw + ICON_GAP, YL, lab, ha="left", va="center", fontsize=fs.FS_SMALL, color=fs.TEXT)
    x += w + ITEM_GAP

fs.save(fig, "fig_E", rows_out)
