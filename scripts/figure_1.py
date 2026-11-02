"""Figure 1 (experimental timeline), drawn entirely with matplotlib.

All elements are drawn by the authors in code: the physiological and TET traces
are illustrative (synthetic, fixed random seed), and the icons are built from
geometric primitives. No generative AI or third-party artwork is used.

Usage:  python scripts/figure_1.py [output.png]   (default: results/figures/figure_1.png)
"""
import sys
from pathlib import Path as FsPath

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, PathPatch, Arc
from matplotlib.path import Path

OUT = sys.argv[1] if len(sys.argv) > 1 else (
    FsPath(__file__).resolve().parents[1] / "results" / "figures" / "figure_1.png")

plt.rcParams["font.family"] = "Arial"
BLUE, RED, ORANGE = "#2f78b7", "#d0312d", "#f08a24"
GREEN, LGREEN, LBLUE, PURPLE = "#3a9a3a", "#7cc47c", "#93c4e6", "#8a55b0"
PANEL_BG, CARD_BG = "#f0f0f0", "#ffffff"
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(16, 8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 80)
ax.axis("off")

PANELS = [(2, 37.5), (42.3, 36), (81.6, 37.5), (121.9, 36.1)]
Y0, H = 1.5, 76


def panel(x, w):
    ax.add_patch(FancyBboxPatch((x, Y0), w, H, boxstyle="round,pad=0,rounding_size=2",
                                fc=PANEL_BG, ec="#e2e2e2", lw=1))


def card(x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=CARD_BG, ec="none"))


def title(xc, stage, l1, l2):
    ax.text(xc, 72.5, f"Stage {stage}:", ha="center", va="center", fontsize=21, weight="bold")
    ax.text(xc, 68.3, l1, ha="center", va="center", fontsize=20)
    ax.text(xc, 64.2, l2, ha="center", va="center", fontsize=20)


def arrow(x):
    ax.annotate("", xy=(x + 2.6, 35.5), xytext=(x - 0.2, 35.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="k", mutation_scale=28))


# ----------------------------------------------------------------- icons
def heart(xc, yc, s=2.6):
    t = np.linspace(0, 2 * np.pi, 200)
    hx = 16 * np.sin(t) ** 3
    hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    ax.add_patch(Polygon(np.c_[xc + hx / 16 * s, yc + hy / 16 * s], fc=BLUE, ec="none"))
    ex = np.array([-0.85, -0.35, -0.2, 0.0, 0.15, 0.3, 0.85]) * s
    ey = np.array([0, 0, 0.35, -0.45, 0.55, 0, 0]) * s * 0.9
    ax.plot(xc + ex, yc + 0.1 * s + ey, color="white", lw=1.8, solid_capstyle="round")


def drop(xc, yc, s=2.6):
    t = np.linspace(-np.pi * 0.85, np.pi * 0.85, 120)
    bx, by = np.sin(t) * s * 0.75, -np.cos(t) * s * 0.75
    pts = np.r_[[[0, s * 1.35]], np.c_[bx, by][::-1]]
    ax.add_patch(Polygon(np.c_[xc + pts[:, 0], yc + pts[:, 1]], fc=ORANGE, ec="none"))
    ax.add_patch(Arc((xc, yc), s * 0.85, s * 0.85, theta1=210, theta2=290,
                     color="white", lw=1.6))


def lungs(xc, yc, s=2.6, c=GREEN):
    for sign in (-1, 1):
        verts = [(0.25, 0.9), (0.9, 0.9), (1.15, -0.2), (1.0, -1.0), (0.55, -1.05),
                 (0.25, -0.6), (0.25, 0.9)]
        verts = [(xc + sign * vx * s, yc + vy * s) for vx, vy in verts]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=c, lw=1.8))
    ax.plot([xc, xc], [yc + 1.3 * s, yc + 0.15 * s], color=c, lw=1.8)
    for sign in (-1, 1):
        ax.plot([xc, xc + sign * 0.45 * s], [yc + 0.15 * s, yc - 0.25 * s], color=c, lw=1.8)


def speaker(xc, yc, s=1.3):
    ax.add_patch(Polygon([(xc - 1.0 * s, yc - 0.45 * s), (xc - 0.45 * s, yc - 0.45 * s),
                          (xc + 0.25 * s, yc - 1.05 * s), (xc + 0.25 * s, yc + 1.05 * s),
                          (xc - 0.45 * s, yc + 0.45 * s), (xc - 1.0 * s, yc + 0.45 * s)],
                         fc="k", ec="none"))
    for r in (0.9, 1.5):
        ax.add_patch(Arc((xc + 0.25 * s, yc), r * s * 2 * 0.55, r * s * 2,
                         theta1=-55, theta2=55, color="k", lw=1.8))


def wind(xc, yc, s=1.0):
    for dy, L, r in ((1.2, 4.2, 0.75), (0.0, 5.0, 0.0), (-1.2, 3.6, 0.6)):
        ax.plot([xc - 2.5 * s, xc - 2.5 * s + L * s], [yc + dy * s] * 2, color="k", lw=2.4,
                solid_capstyle="round")
        if r:
            ax.add_patch(Arc((xc - 2.5 * s + L * s, yc + dy * s + np.sign(dy) * r * s),
                             2 * r * s, 2 * r * s,
                             theta1=-90 if dy > 0 else 0, theta2=180 if dy > 0 else 270,
                             color="k", lw=2.4))


# --------------------------------------------------------------- signals
def ecg(n_beats, n=1500, amp=1.0, noise=0.04):
    x = np.linspace(0, 1, n)
    y = noise * rng.standard_normal(n)
    for c in np.linspace(0.04, 0.96, n_beats):
        y += amp * np.exp(-((x - c) / 0.004) ** 2)
        y -= 0.35 * amp * np.exp(-((x - c - 0.009) / 0.004) ** 2)
        y += 0.18 * amp * np.exp(-((x - c - 0.035) / 0.012) ** 2)
    return x, y


def eda_flat(n=800):
    x = np.linspace(0, 1, n)
    return x, np.convolve(rng.standard_normal(n), np.ones(15) / 15, "same") * 0.25


def eda_peaks(n=1500, peaks=(0.06, 0.3, 0.52, 0.6, 0.83, 0.93), amps=(1, .55, 1, .6, .45, 1)):
    x = np.linspace(0, 1, n)
    y = 0.03 * rng.standard_normal(n)
    for c, a in zip(peaks, amps):
        d = np.clip(x - c, 0, None)
        y += a * (np.exp(-d / 0.03) - np.exp(-d / 0.006)) * (x > c) * 1.6
    return x, y


def resp(cycles, n=1500, amp=1.0):
    x = np.linspace(0, 1, n)
    return x, amp * np.sin(2 * np.pi * cycles * x - np.pi / 2) + 0.03 * rng.standard_normal(n)


def draw(xs, ys, x0, x1, yc, h, color, lw=1.6):
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12) - 0.5
    ax.plot(x0 + xs * (x1 - x0), yc + ys * h, color=color, lw=lw)


def physio_block(px, x0, x1, labels_x, active):
    rows = [(43, "Cardiac", heart), (29, "EDA", drop), (16, "Respiration", lungs)]
    for yc, lab, icon in rows:
        icon(labels_x, yc + 2.0)
        ax.text(labels_x, yc - 2.9, lab, ha="center", va="center", fontsize=14)
    ax.plot([x0, x0], [11, 49], ls=(0, (3, 3)), color="#555", lw=0.9)
    ax.plot([x1, x1], [11, 49], ls=(0, (3, 3)), color="#555", lw=0.9)
    if active:
        draw(*ecg(21, amp=1), x0, x1, 43.5, 6.0, BLUE)
        draw(*eda_peaks(), x0, x1, 29, 9.5, ORANGE)
        draw(*resp(17.5), x0, x1, 16, 8.0, GREEN, lw=1.8)
    else:
        xs, ys = ecg(7, amp=1, noise=0.04)
        draw(xs, ys, x0, x1, 43.5, 6.0, BLUE)
        xs, ys = eda_flat()
        draw(xs, ys, x0, x1, 28.5, 0.7, ORANGE)
        draw(*resp(3, amp=1), x0, x1, 16.5, 6.0, GREEN, lw=1.8)
    for sx in (x0, (x0 + x1) / 2, x1):
        speaker(sx, 8.6)
    ax.plot([x0, x1], [5.6, 5.6], color="k", lw=1.2)


def tet_card(x, w, curves):
    card(x, 6, w, 52)
    rows = np.linspace(53, 16, 6)
    labels = ["Pleasantness", "Unpleasantness", "Emotional Intensity",
              "Interoception", "Bliss", "Anxiety"]
    for (lab, (c, bump)), yr in zip(zip(labels, curves), rows):
        xs = np.linspace(0, 1, 400)
        if bump is None:
            ys = 0.15 * np.convolve(rng.standard_normal(400), np.ones(40) / 40, "same")
            yb = yr - 3.4
        else:
            mu, sd, a, decay = bump
            rise = np.exp(-((xs - mu) / sd) ** 2 / 2)
            fall = (1 - decay) * np.exp(-((xs - mu) / (sd * 2.2)) ** 2 / 2) + decay
            ys = a * np.where(xs < mu, rise, fall) * (1 / (1 + np.exp(-(xs - 0.22) / 0.035)))
            yb = yr - 4.6
        ax.plot(x + 1.5 + xs * (w - 3), yb + ys, color=c, lw=2.4)
        ax.text(x + w / 2, yr, lab, ha="center", va="center", fontsize=15.5)


# ---------------------------------------------------------------- layout
for x, w in PANELS:
    panel(x, w)

# Stage 1
xc = PANELS[0][0] + PANELS[0][1] / 2
title(xc, 1, "10-minute Eyes Closed", "(RS)")
physio_block(0, 17.5, 37, 9.0, active=False)
ax.text((17.5 + 37) / 2, 4.4, "10-minute", ha="center", va="top", fontsize=15)

# Stage 2
xc = PANELS[1][0] + PANELS[1][1] / 2
title(xc, 2, "Retrospective TET", "(Post-RS)")
tet_card(PANELS[1][0] + 2, PANELS[1][1] - 4,
         [(BLUE, None), (RED, None), (ORANGE, None), (LGREEN, None), (BLUE, None), (PURPLE, None)])

# Stage 3
xc = PANELS[2][0] + PANELS[2][1] / 2
title(xc, 3, "DMT Administration", "(20mg/40mg)")
wind(87.5, 57.2)
ax.text(87.8, 53.3, "Inhalation", ha="center", va="center", fontsize=14)
physio_block(0, 97.5, 117.2, 88.8, active=True)
ax.text(97.5, 4.4, "$t_0$", ha="center", va="top", fontsize=17)
ax.text((97.5 + 117.2) / 2 + 0.6, 4.4, "20 minutes", ha="center", va="top", fontsize=15)

# Stage 4
xc = PANELS[3][0] + PANELS[3][1] / 2
title(xc, 4, "Retrospective TET", "(Post-DMT)")
tet_card(PANELS[3][0] + 2, PANELS[3][1] - 4,
         [(BLUE, (0.40, 0.08, 3.2, 0.35)), (RED, (0.58, 0.12, 1.1, 0.05)),
          (ORANGE, (0.40, 0.07, 3.3, 0.05)), (LGREEN, (0.58, 0.09, 3.3, 0.05)),
          (LBLUE, (0.45, 0.08, 2.2, 0.05)), (PURPLE, (0.42, 0.08, 2.2, 0.05))])

for x in (39.8, 79.0, 119.6):
    arrow(x)

fig.savefig(OUT, dpi=250, facecolor="white")
print("saved", OUT)
