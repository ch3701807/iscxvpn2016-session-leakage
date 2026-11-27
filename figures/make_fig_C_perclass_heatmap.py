"""Fig. 2 (fig_C.pdf): flows, sessions, applications and per-class F1 in the nine-class task.

Sources: paperA/gen/tab_perclass.tex (RF), tab_perclass_deep.tex (Fusion, 1D-CNN),
tab_sessions.tex (flows; suffix/evidence/E3), neon_fig/data/session_app_map.csv (applications);
cross-checked against gen/facts.json and gen/numbers.tex.
"""
from __future__ import annotations

import csv
import re

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import figstyle as fs
from figstyle import plt

fs.setup()
nums = fs.read_numbers()
facts = fs.read_facts()

# ---------------------------------------------------------------- data
F1 = {"RF": {}, "Fu": {}, "Cn": {}}
for r in fs.tabular_body(fs.GEN / "tab_perclass.tex"):
    assert len(r) == 5, r
    F1["RF"][r[0].replace("\\_", "_")] = [fs.tex_num(x) for x in r[1:]]
for r in fs.tabular_body(fs.GEN / "tab_perclass_deep.tex"):
    assert len(r) == 9, r
    c = r[0].replace("\\_", "_")
    F1["Fu"][c] = [fs.tex_num(x) for x in r[1:5]]
    F1["Cn"][c] = [fs.tex_num(x) for x in r[5:9]]
CLASSES = sorted(F1["RF"])
assert len(CLASSES) == 9 and set(CLASSES) == set(fs.CLASS_SHORT) == set(F1["Fu"]) == set(F1["Cn"])

# column order of the source tables must be the ladder order
hdr = (fs.GEN / "tab_perclass.tex").read_text()
assert re.search(r"Class & Flow-random & File-grouped & Suffix session \(P\*\) & Evidence session", hdr)
assert "Class & Flow & File & P* & Evid." in (fs.GEN / "tab_perclass_deep.tex").read_text()

# cross-check against facts.json (independent copies written by make_numbers.py)
alpha = sorted(fs.CLASS_SHORT)
for c in CLASSES:
    pr = facts[f"perclass_RF_{c}"]
    assert F1["RF"][c] == [pr[s] for s in fs.SPLITS], c
    for m in ("Fu", "Cn"):
        assert F1[m][c] == [facts["v35_perclass_deep"][m][s][alpha.index(c)] for s in fs.SPLITS], (m, c)

# flows and session counts (tab_sessions.tex is transposed: classes are columns)
sess_rows = fs.tabular_body(fs.GEN / "tab_sessions.tex")
head = [h.strip() for h in re.search(r"^\s*&(.*Total)\s*\\\\", (fs.GEN / "tab_sessions.tex").read_text(),
                                      re.M).group(1).split("&")]
assert head[-1] == "Total" and len(head) == 10, head
cols = [fs.SHORT_CLASS[h] for h in head[:-1]]
S = {}
for r in sess_rows:
    S[r[0]] = dict(zip(cols + ["Total"], (int(x) for x in r[1:])))
assert set(S) == {"Flows", "Suffix sess.", "Evidence sess.", "E3 groups"}
for k, v in S.items():
    assert sum(v[c] for c in cols) == v["Total"], k
for c in CLASSES:
    spc = facts["sessions_per_class"][c]
    assert (S["Flows"][c], S["Suffix sess."][c], S["Evidence sess."][c], S["E3 groups"][c]) == \
        (spc["flows"], spc["suffix"], spc["evidence"], spc["e3"]), c
assert S["Flows"]["Total"] == int(nums["NFlows"]) == 7371
assert S["Suffix sess."]["Total"] == int(nums["NSessMulti"])
assert S["Evidence sess."]["Total"] == int(nums["NEvSessMulti"])
assert S["E3 groups"]["Total"] == int(nums["NEThreeMulti"])

# applications per class from the session -> application map
apps = {}
with fs.SESSION_APP_CSV.open() as f:
    for r in csv.DictReader(f):
        apps.setdefault(r["class"], set()).add(r["application"])
NAPP = {c: len(apps[c]) for c in CLASSES}
assert NAPP == {c: facts["v37_applications_per_class"][c]["applications"] for c in CLASSES}
assert NAPP == {"chat": 6, "email": 1, "file_transfer": 4, "streaming": 4, "voip": 4, "vpn_chat": 5,
                "vpn_file_transfer": 3, "vpn_streaming": 4, "vpn_voip": 4}
STAR = {c for c in CLASSES if NAPP[c] == S["Suffix sess."][c]}
assert STAR == {"vpn_chat", "vpn_file_transfer", "vpn_streaming"} and len(STAR) == int(nums["NClassesDistinctApp"])

# text-quoted values (numbers.tex)
assert nums["PcEmailRFP"] == f'{F1["RF"]["email"][2]:.1f}' == nums["PcEmailSuffix"]
assert nums["PcEmailFlow"] == f'{F1["RF"]["email"][0]:.1f}'
assert nums["PcEmailFuP"] == f'{F1["Fu"]["email"][2]:.1f}'
assert nums["PcEmailCnP"] == f'{F1["Cn"]["email"][2]:.1f}'
assert nums["PcVpnFtFlow"] == f'{F1["RF"]["vpn_file_transfer"][0]:.1f}'
assert nums["PcVpnFtSuffix"] == f'{F1["RF"]["vpn_file_transfer"][2]:.1f}'
assert F1["RF"]["email"] == [60.6, 86.2, 0.5, 0.2]
assert int(nums["NSessEmail"]) == S["Suffix sess."]["email"] and int(nums["NSessVft"]) == S["Suffix sess."]["vpn_file_transfer"]
assert int(nums["NClassesFewSess"]) == sum(S["Suffix sess."][c] <= 3 for c in CLASSES)
assert int(nums["NClassesFewEThree"]) == sum(1 <= S["E3 groups"][c] <= 4 for c in CLASSES)
assert int(nums["NClassesLarge"]) == sum(S["Suffix sess."][c] >= int(nums["NSessLargeMin"]) for c in CLASSES)

# rows: suffix sessions ascending; the first block is the four classes the text singles out
ORDER = sorted(CLASSES, key=lambda c: (S["Suffix sess."][c], fs.CLASS_SHORT[c]))
TOP = ORDER[:4]
assert set(TOP) == {"email"} | STAR, TOP

# ---------------------------------------------------------------- figure (coordinates in pt)
W_IN, H_IN = 4.80, 1.80
WP, HP = W_IN * 72, H_IN * 72
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WP)
ax.set_ylim(0, HP)
ax.axis("off")

CMAP = LinearSegmentedColormap.from_list("f1", ["#f6f9fe", "#d3e5f9", "#94c1ed", "#5b9ce0"])
ROW_H, GROUP_GAP = 11.3, 4.0
CELL_W, CELL_GAP, SESS_GAP, BLOCK_GAP = 16.8, 1.0, 3.2, 6.6
X_HEAT = 113.0
Y_COLHDR = HP - 16.8          # column labels baseline area
Y_TOP = HP - 22.2             # top of first row
X_CLASS, X_FLOWS, X_SESS, X_APPS = 1.5, 50.5, 85.5, 104.5


def row_y(i):
    return Y_TOP - (i + 1) * ROW_H - (GROUP_GAP if i >= len(TOP) else 0.0)


def col_x(b, j):
    block_w = 4 * CELL_W + 2 * CELL_GAP + SESS_GAP
    x = X_HEAT + b * (block_w + BLOCK_GAP) + j * CELL_W
    return x + (CELL_GAP if j >= 1 else 0) + (SESS_GAP if j >= 2 else 0) + (CELL_GAP if j >= 3 else 0)


right_edge = col_x(2, 3) + CELL_W
assert right_edge <= WP - 0.5, right_edge
assert row_y(8) >= 1.0, row_y(8)

# left table
hdr_kw = dict(fontsize=fs.FS_SMALL, color=fs.MUTED, va="center")
ax.text(X_CLASS, Y_COLHDR, "Class", ha="left", **hdr_kw)
ax.text(X_FLOWS, Y_COLHDR, "Flows", ha="right", **hdr_kw)
ax.text(X_SESS, Y_COLHDR, "Sess.", ha="right", **hdr_kw)
ax.text(X_APPS - 0.6, Y_COLHDR, "Apps", ha="right", **hdr_kw)
rows_out = []
for i, c in enumerate(ORDER):
    yc = row_y(i) + ROW_H / 2
    kw = dict(fontsize=fs.FS_SMALL, va="center", color=fs.TEXT)
    ax.text(X_CLASS, yc, fs.CLASS_SHORT[c], ha="left", **kw)
    ax.text(X_FLOWS, yc, f'{S["Flows"][c]}', ha="right", **kw)
    ax.text(X_SESS, yc, f'{S["Suffix sess."][c]}/{S["Evidence sess."][c]}/{S["E3 groups"][c]}',
            ha="right", **kw)
    ax.text(X_APPS - 3.0, yc, f"{NAPP[c]}", ha="right", **kw)      # digits stay aligned
    if c in STAR:
        ax.text(X_APPS - 2.6, yc, "*", ha="left", **kw)

# heat blocks
for b, m in enumerate(("RF", "Fu", "Cn")):
    x0, x1 = col_x(b, 0), col_x(b, 3) + CELL_W
    ax.text((x0 + x1) / 2, HP - 4.6, fs.MODELS[m]["long"], ha="center", va="center",
            fontsize=fs.FS, fontweight="bold")
    ax.plot([x0, x1], [HP - 10.0, HP - 10.0], color=fs.AXIS, lw=0.5)
    for j, s in enumerate(fs.SPLITS):
        xc = col_x(b, j) + CELL_W / 2
        ax.text(xc, Y_COLHDR, fs.SPLIT_SHORT[s], ha="center", **hdr_kw)
        for i, c in enumerate(ORDER):
            v = F1[m][c][j]
            y = row_y(i)
            ax.add_patch(Rectangle((col_x(b, j), y + 0.5), CELL_W, ROW_H - 1.0, lw=0,
                                   facecolor=CMAP(v / 100.0)))
            ax.text(xc, y + ROW_H / 2, f"{v:.1f}", ha="center", va="center",
                    fontsize=fs.FS_SMALL, color=fs.TEXT)
            rows_out.append(dict(model=fs.MODELS[m]["long"], cls=c, short=fs.CLASS_SHORT[c],
                                 split=fs.SPLIT_LONG[s], f1=v, flows=S["Flows"][c],
                                 suffix=S["Suffix sess."][c], evidence=S["Evidence sess."][c],
                                 e3=S["E3 groups"][c], apps=NAPP[c], star=c in STAR))

fs.save(fig, "fig_C", rows_out)
