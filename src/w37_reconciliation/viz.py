"""Style graphique commun aux notebooks.

Palette catégorielle : les 3 premiers slots de la palette de référence (validés « toutes paires »,
y compris en daltonisme). Au-delà de 3 séries : petits multiples, pas de 4e couleur.
Séquentiel : un seul ton (bleu), clair -> foncé. Divergent : bleu <-> rouge, milieu gris.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "w37_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "w37_div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#a32a2a"])


def setup() -> None:
    """Thème matplotlib : axes et grille discrets, texte en encre neutre."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "lines.linewidth": 2, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
    })


def heatmap(ax: plt.Axes, M: np.ndarray, labels_x, labels_y=None, title: str = "",
            diverging: bool = False, fmt: str = "{:.2f}", vmax: float | None = None,
            colorbar: bool = True) -> None:
    """Matrice annotée. diverging=True : 0 au centre (gris), négatif bleu, positif rouge."""
    labels_y = labels_x if labels_y is None else labels_y
    if diverging:
        lim = vmax or float(np.abs(M).max()) or 1.0
        im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    else:
        im = ax.imshow(M, cmap=SEQUENTIAL, vmin=0, vmax=vmax)
    norm = im.norm
    for (i, j), v in np.ndenumerate(M):
        shade = abs(norm(v) - 0.5) * 2 if diverging else norm(v)
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                color="white" if shade > 0.6 else TEXT)
    ax.set_xticks(range(M.shape[1]), labels_x, rotation=35, ha="right")
    ax.set_yticks(range(M.shape[0]), labels_y)
    ax.grid(False)
    ax.set_title(title)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def short(labels) -> list[str]:
    """'Bretagne/GMS' -> 'Bret/GMS', 'PaysLoire' -> 'PdL' : tient sous un axe."""
    return [str(u).replace("Bretagne", "Bret").replace("PaysLoire", "PdL") for u in labels]
