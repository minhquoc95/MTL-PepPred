"""Rebuild Figure 1 (MTL-PepPred workflow) as vector SVG and render it at print quality.

Canvas is 2000 x 2140 user units, rendered to 190 mm width at 600 DPI (4488 px).
1 unit = 0.0952 mm, so a 30-unit font prints at about 8 pt.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results"
W, H = 2000, 2140

INK = "#1b2a36"
SUB = "#5d6f7c"
BLUE_F, BLUE_S = "#dce9f8", "#2f6fb4"
GREEN_F, GREEN_S, GREEN_T = "#dcedd6", "#4f9a4f", "#3f8a4a"
TEAL_F, TEAL_S = "#d3ebeb", "#2f8e8e"
GREY_F, GREY_S = "#eef1f3", "#9fadb7"
CONT_F, CONT_S = "#f7f9fb", "#dbe2e8"
OR_F, OR_S, OR_T = "#fdf1dd", "#e09a3e", "#c98120"
ARROW = "#4a5763"
FONT = "Liberation Sans, Arial, Helvetica, sans-serif"

el = []


def rect(x, y, w, h, fill, stroke, rx=14, sw=3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=30, fill=INK, weight="normal", anchor="middle"):
    el.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
              f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')


def arrow(x1, y1, x2, y2, head=22, sw=4):
    import math
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    xe, ye = x2 - ux * head, y2 - uy * head
    el.append(f'<line x1="{x1}" y1="{y1}" x2="{xe:.1f}" y2="{ye:.1f}" '
              f'stroke="{ARROW}" stroke-width="{sw}" stroke-linecap="round"/>')
    px, py = -uy, ux
    w = head * 0.46
    pts = f"{x2:.1f},{y2:.1f} {xe + px * w:.1f},{ye + py * w:.1f} {xe - px * w:.1f},{ye - py * w:.1f}"
    el.append(f'<polygon points="{pts}" fill="{ARROW}"/>')


def line(x1, y1, x2, y2, sw=4):
    el.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ARROW}" '
              f'stroke-width="{sw}" stroke-linecap="round"/>')


AX = 900  # main vertical axis

# --- input -------------------------------------------------------------
rect(AX - 340, 55, 680, 150, GREY_F, GREY_S)
text(AX, 118, "Peptide sequence", 42, INK, "bold")
text(AX, 170, "e.g., H A I Y K F R … (single-letter code)", 29, SUB)
arrow(AX, 205, AX, 285)

# --- shared encoder ----------------------------------------------------
rect(300, 290, 1200, 440, CONT_F, CONT_S, rx=20, sw=3)
text(340, 348, "Shared encoder", 36, "#44525e", "bold", anchor="start")

rect(340, 375, 520, 195, BLUE_F, BLUE_S)
text(600, 432, "Frozen ESM-2", 38, INK, "bold")
text(600, 480, "esm2_t33_650M_UR50D", 28, SUB)
text(600, 528, "650M parameters · frozen", 28, BLUE_S)

rect(900, 375, 560, 195, GREEN_F, GREEN_S)
text(1180, 432, "Learnable embedding", 38, INK, "bold")
text(1180, 480, "33-token vocab → 1280-d", 28, SUB)
text(1180, 528, "trainable", 28, GREEN_T)

rect(340, 600, 1120, 100, "#f1f4f7", "#e1e7ec", rx=12, sw=2.5)
text(900, 662, "h = α · h(ESM) + (1 − α) · h(learnable),   α = 0.9", 32, INK)

# --- split to the two branches ----------------------------------------
line(AX, 730, AX, 800)
text(AX, 838, "per-residue embeddings (L × 1280)", 28, SUB)
arrow(AX, 862, 560, 928)
arrow(AX, 862, 1240, 928)

rect(240, 935, 640, 250, GREEN_F, GREEN_S)
text(560, 1000, "Transformer branch", 38, INK, "bold")
text(560, 1050, "Global context", 30, BLUE_S)
text(560, 1100, "4 layers · 8 heads · d = 1280", 28, SUB)
text(560, 1148, "long-range dependencies", 28, SUB)

rect(920, 935, 640, 250, GREEN_F, GREEN_S)
text(1240, 1000, "CNN branch", 38, INK, "bold")
text(1240, 1050, "Local motifs", 30, BLUE_S)
text(1240, 1100, "1D conv · kernel = 7", 28, SUB)
text(1240, 1148, "short sequence patterns", 28, SUB)

# --- fusion ------------------------------------------------------------
arrow(560, 1185, AX - 120, 1268)
arrow(1240, 1185, AX + 120, 1268)
rect(AX - 380, 1275, 760, 135, TEAL_F, TEAL_S)
text(AX, 1333, "Feature fusion (concatenate)", 38, INK, "bold")
text(AX, 1383, "L × 2560", 29, SUB)
arrow(AX, 1410, AX, 1478)

# --- task heads + TUM loss --------------------------------------------
rect(180, 1485, 1440, 275, CONT_F, CONT_S, rx=20, sw=3)
text(AX, 1548, "21 task-specific heads (one per bioactivity)", 38, INK, "bold")

heads = ["ACE", "DPP-IV", "Bitter", "Umami", None, None]
bw, gap, y0, bh = 180, 22, 1580, 130
x = 212
for lbl in ["ACE", "DPP-IV", "Bitter", "Umami"]:
    rect(x, y0, bw, bh, GREEN_F, GREEN_S, rx=12)
    text(x + bw / 2, y0 + bh / 2 + 10, lbl, 29)
    x += bw + gap
rect(x, y0, bw, bh, GREEN_F, GREEN_S, rx=12)
text(x + bw / 2, y0 + bh / 2 - 8, "Anti-", 29)
text(x + bw / 2, y0 + bh / 2 + 30, "microbial", 29)
x += bw + gap
text(x + 34, y0 + bh / 2 + 12, "…", 40, SUB)
x += 92
rect(x, y0, 268, bh, GREEN_F, GREEN_S, rx=12)
text(x + 134, y0 + bh / 2 - 8, "Signal", 29)
text(x + 134, y0 + bh / 2 + 30, "peptide", 29)

rect(1680, 1520, 300, 205, OR_F, OR_S, rx=14, sw=3, dash="14 10")
text(1830, 1585, "TUM loss", 34, INK, "bold")
text(1830, 1635, "task-balancing", 27, SUB)
text(1830, 1690, "training only", 27, OR_T)
arrow(1676, 1622, 1624, 1622)

# --- output ------------------------------------------------------------
arrow(AX, 1760, AX, 1828)
rect(AX - 420, 1835, 840, 150, GREY_F, GREY_S)
text(AX, 1898, "21 bioactivity probabilities", 42, INK, "bold")
text(AX, 1948, "computed for each peptide in a single forward pass", 29, SUB)

# --- legend ------------------------------------------------------------
ly = 2085
for cx, f, s, lbl, dash in [
    (398, BLUE_F, BLUE_S, "Frozen (pre-trained)", None),
    (854, GREEN_F, GREEN_S, "Trainable", None),
    (1145, OR_F, OR_S, "Training only", "12 8"),
]:
    rect(cx, ly - 34, 46, 46, f, s, rx=8, sw=3, dash=dash)
    text(cx + 66, ly + 2, lbl, 30, "#44525e", anchor="start")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#ffffff"/>'
       + "\n".join(el) + "</svg>")
(Path(__file__).resolve().parent / "Figure_1_workflow.svg").write_text(svg, encoding="utf-8")

import cairosvg
from PIL import Image

px = round(190 / 25.4 * 600)  # 190 mm at 600 dpi
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "Figure_1_workflow.png"),
                 output_width=px)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / "Figure_1_workflow.pdf"))

im = Image.open(OUT / "Figure_1_workflow.png").convert("RGB")
im.save(OUT / "Figure_1_workflow.png", dpi=(600, 600))
im.save(OUT / "Figure_1_workflow.tif", dpi=(600, 600), compression="tiff_lzw")
print("size px:", im.size, "-> mm:", round(im.size[0] / 600 * 25.4, 1), "x",
      round(im.size[1] / 600 * 25.4, 1))
