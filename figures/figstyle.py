"""Shared style, data readers and export checks for the four figures of draft_v3_11.

Every figure script in this folder imports this module, so fonts, colours, model encodings, class
and split names, and the session-split shading are identical across Fig. 1-4. Outputs go to ./out/.

Inputs are read in place (read-only):
  outputs/ (or expected/ if outputs/ has not been generated yet)
  manifest/session_app_map.csv
  manifest/sibling_pairs.csv
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402

# ----------------------------------------------------------------------------
# Paths (sources are read-only)
# ----------------------------------------------------------------------------
import os
ROOT = Path(os.environ["ARTEFACT_ROOT"]) if "ARTEFACT_ROOT" in os.environ else Path(__file__).resolve().parent.parent
GEN = ROOT / "outputs" if (ROOT / "outputs" / "numbers.tex").exists() else ROOT / "expected"
NUMBERS_TEX = GEN / "numbers.tex"
SESSION_APP_CSV = ROOT / "manifest" / "session_app_map.csv"
SIBLING_CSV = ROOT / "manifest" / "sibling_pairs.csv"

OUT = Path(__file__).resolve().parent / "out"

# ----------------------------------------------------------------------------
# Typography: Arimo (TrueType, Helvetica metrics) for every glyph.
# ----------------------------------------------------------------------------
FONT_DIR = Path(os.environ["ARIMO_FONT_DIR"]) if "ARIMO_FONT_DIR" in os.environ else None
if FONT_DIR is not None and FONT_DIR.is_dir():
    for _f in ("Arimo-Regular.ttf", "Arimo-Bold.ttf", "Arimo-Italic.ttf", "Arimo-BoldItalic.ttf"):
        if (FONT_DIR / _f).exists():
            font_manager.fontManager.addfont(str(FONT_DIR / _f))
_HAS_ARIMO = any(f.name == "Arimo" for f in font_manager.fontManager.ttflist)
FONT_FAMILY = "Arimo" if _HAS_ARIMO else "DejaVu Sans"

FS = 7.0          # base text
FS_SMALL = 6.5    # ticks, annotations (minimum size used anywhere)
FS_PANEL = 7.5    # panel labels

# Ink
TEXT = "#1a1a1a"
MUTED = "#5c5c5c"
AXIS = "#8f8f8f"
GRID = "#e4e4e4"
ZERO = "#7a7a7a"
BAND = "#f0f0f0"

# Models: colour + marker + dash (identity never relies on colour alone).
# Palette validated with the dataviz validator (all-pairs CVD dE >= 11.8, normal-vision dE >= 22.0);
# grey levels 100/169/138, so the three models stay apart in greyscale print.
MODELS = {
    "RF": dict(name="RF", long="RF (len+IAT)", color="#2f6eba", marker="o",
               ls="-", ms=4.2),
    "Fu": dict(name="Fusion", long="Fusion", color="#f5973f", marker="s",
               ls=(0, (3.6, 1.6)), ms=3.8),
    "Cn": dict(name="1D-CNN", long="1D-CNN (len+IAT)", color="#2fb4a1", marker="^",
               ls=(0, (1.0, 1.3)), ms=4.4),
}
# Session-split shading and focus+context greys (Fig. 3, Fig. 4)
SESSION_TINT = "#eef1f5"      # background of the session-split region (P*, evidence)
ANNOT = "#404040"             # annotation ink (CI whiskers on bars)
FADE = "#c2c2c2"              # context marks: CIs that cover zero in Fig. 4b
FADE_TEXT = "#8a8a8a"
LEAK = "#f5973f"      # orange leakage arcs in Fig. 1 (caption: "orange arcs")

# Splits (same order and names in every figure)
SPLITS = ["leak", "no_session", "Pstar", "evidence"]
SPLIT_LONG = {"leak": "Flow-random", "no_session": "File-grouped",
              "Pstar": "Suffix (P*)", "evidence": "Evidence session"}
SPLIT_2LINE = {"leak": "Flow-\nrandom", "no_session": "File-\ngrouped",
               "Pstar": "Suffix\n(P*)", "evidence": "Evidence\nsession"}
SPLIT_SHORT = {"leak": "Flow", "no_session": "File", "Pstar": "P*", "evidence": "Evid."}

# Classes: abbreviations defined in the Fig. 2 caption (FT: file transfer; V-: VPN)
CLASS_SHORT = {
    "chat": "Chat", "email": "Email", "file_transfer": "FT", "streaming": "Stream",
    "voip": "VoIP", "vpn_chat": "V-Chat", "vpn_file_transfer": "V-FT",
    "vpn_streaming": "V-Stream", "vpn_voip": "V-VoIP",
}
SHORT_CLASS = {v: k for k, v in CLASS_SHORT.items()}


def setup() -> None:
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "butt",
        "savefig.dpi": 300,
        "figure.dpi": 100,
    })


def axes_in(fig, left, bottom, width, height, **kw):
    """Add axes with geometry given in inches (figure size is fixed)."""
    W, H = fig.get_size_inches()
    return fig.add_axes([left / W, bottom / H, width / W, height / H], **kw)


def panel_heading(fig, x_in, y_in, tag, title):
    """Panel tag in bold plus a short plain title, top-aligned at (x_in, y_in) inches."""
    W, H = fig.get_size_inches()
    t = fig.text(x_in / W, y_in / H, tag, fontsize=FS_PANEL, fontweight="bold", ha="left", va="top")
    fig.canvas.draw()
    bb = t.get_window_extent()
    x2 = bb.x1 / fig.dpi + 0.045
    fig.text(x2 / W, y_in / H, title, fontsize=FS, ha="left", va="top", color=TEXT)


def fmt_signed(v: float, nd: int = 1) -> str:
    s = f"{v:+.{nd}f}"
    return s.replace("-", "\u2212")


# ----------------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------------
def tex_num(s: str) -> float:
    s = s.strip().replace("$-$", "-").replace("$+$", "+").replace("\\,", "")
    return float(s)


def read_numbers() -> dict[str, str]:
    """Macros of the compiled manuscript: name -> raw TeX string."""
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", NUMBERS_TEX.read_text(), re.M):
        out[m.group(1)] = m.group(2)
    assert len(out) > 400, f"numbers.tex parsed only {len(out)} macros"
    return out


def macro_float(nums: dict[str, str], name: str) -> float:
    return tex_num(nums[name])


def read_facts() -> dict:
    return json.loads((GEN / "facts.json").read_text())


def tabular_body(path: Path) -> list[list[str]]:
    rows, body = [], False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("\\midrule"):
            body = True
            continue
        if line.startswith("\\bottomrule"):
            break
        if not body or not line:
            continue
        line = re.sub(r"\\\\(\[[^\]]*\])?\s*$", "", line).strip()
        if line:
            rows.append([c.strip() for c in line.split("&")])
    return rows


def assert_macro(nums, name, value, nd=1):
    """Rounded plotted value must equal the manuscript macro exactly (as text)."""
    want = tex_num(nums[name])
    got = round(value + 0.0, nd)
    assert abs(got - want) < 1e-9, f"{name}: data {value} -> {got} != numbers.tex {want}"


# ----------------------------------------------------------------------------
# Export + checks
# ----------------------------------------------------------------------------
def _check_pdf(pdf: Path) -> list[str]:
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True, check=True).stdout
    lines = fonts.strip().splitlines()[2:]
    assert lines, f"{pdf.name}: no fonts listed"
    names = []
    for ln in lines:
        assert "Type 3" not in ln, f"{pdf.name}: Type 3 font: {ln}"
        assert "TrueType" in ln, f"{pdf.name}: non-TrueType font: {ln}"
        cols = ln.split()
        # columns: name type... encoding emb sub uni objID gen
        assert cols[-5] == "yes", f"{pdf.name}: font not embedded: {ln}"
        if _HAS_ARIMO:
            assert cols[0].split("+")[-1].startswith("Arimo"), f"{pdf.name}: unexpected font {cols[0]}"
        names.append(cols[0])
    return names


def save(fig, stem: str, rows: list[dict] | None = None, out_dir: Path | None = None) -> None:
    out = OUT if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W, H = fig.get_size_inches()
    assert abs(W - 4.80) < 1e-9, f"{stem}: width {W} in != 4.80 in"
    pdf = out / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                               "Creator": None})
    plt.close(fig)

    fonts = _check_pdf(pdf)
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info)
    assert m and abs(float(m.group(1)) - 345.6) < 0.01, f"{stem}: page width {m.group(1) if m else '?'} pt"

    # pdftoppm must render the PDF without any warning (font or otherwise).
    r = subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(pdf)],
                       capture_output=True)
    assert r.returncode == 0 and r.stderr.strip() == b"", f"{stem}: pdftoppm stderr: {r.stderr!r}"
    # PNGs are rasterised from the vector PDF itself (cairo: no stroke snapping), so they show
    # exactly what LaTeX embeds.
    png = out / f"{stem}.png"
    r = subprocess.run(["pdftocairo", "-r", "300", "-png", "-singlefile", str(pdf), str(png.with_suffix(""))],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr.strip() == "", f"{stem}: pdftocairo stderr: {r.stderr!r}"
    im = Image.open(png)
    assert im.size[0] == 1440, f"{stem}: PNG width {im.size[0]} px != 1440"
    im.convert("RGB").save(png, dpi=(300, 300))
    im.convert("L").save(out / f"{stem}_gray.png", dpi=(300, 300))

    if rows:
        with (out / f"{stem}_data.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{stem}: {W:.2f} x {H:.2f} in, fonts {sorted(set(n.split('+')[-1] for n in fonts))}, "
          f"PNG {im.size[0]}x{im.size[1]} px, pdftoppm clean")


def session_band(ax, lo=1.5, hi=3.6):
    """Tint the session splits (x = 2, 3 in ladder order)."""
    ax.axvspan(lo, hi, color=SESSION_TINT, lw=0, zorder=0)
