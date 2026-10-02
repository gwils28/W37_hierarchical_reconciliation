"""Figures du bloc 5. Style commun : `viz.setup()`. Une couleur d'accent, le reste en encre neutre."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .. import viz

RECONCILED = ["BU", "OLS", "WLS struct", "WLS var", "MinT shrink"]


def key_figure(by_level: pd.DataFrame, title: str, highlight: str = "MinT shrink",
               show_origins: bool = True) -> plt.Figure:
    """MASE par niveau et par méthode : moyenne ± écart-type entre origines.

    Lignes de référence : le naïf saisonnier (« si on ne le bat pas, rien d'autre ne compte ») et le
    bottom-up (la vraie baseline de réconciliation). Les méthodes sont sur l'axe, pas en couleurs.
    show_origins : une ligne fine par origine. Elle montre si la dispersion vient du NIVEAU d'erreur de
    l'origine (lignes parallèles) ou du CLASSEMENT des méthodes (lignes qui se croisent).
    """
    order = ["base", *RECONCILED]
    stats = by_level.groupby(["level", "method"])["mase"].agg(["mean", "std"])
    n = by_level["origin"].nunique()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False, layout="constrained")
    for ax, level in zip(axes, ["France", "Régions"], strict=True):
        s = stats.loc[level]
        if show_origins:
            per = by_level[by_level["level"] == level].pivot(index="origin", columns="method", values="mase")
            for _, row in per[order].iterrows():
                ax.plot(range(len(order)), row.to_numpy(), color=viz.GRID, lw=1, marker=".", ms=4, zorder=0.5)
        for ref, style, label in [("SN", ":", "naïf saisonnier (168 h)"), ("BU", "--", "bottom-up")]:
            ax.axhline(s.loc[ref, "mean"], color=viz.TEXT_2, ls=style, lw=1.2, zorder=1)
            ax.annotate(label, (-0.45, s.loc[ref, "mean"]), xytext=(0, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color=viz.TEXT_2)
        for i, m in enumerate(order):
            color = viz.BLUE if m == highlight else viz.TEXT_2
            ax.errorbar(i, s.loc[m, "mean"], yerr=s.loc[m, "std"], fmt="o", ms=7, color=color,
                        ecolor=color, elinewidth=1.5, capsize=0, zorder=3)
            ax.annotate(f"{s.loc[m, 'mean']:.3f}", (i, s.loc[m, "mean"]), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=viz.TEXT, fontweight="semibold" if m == highlight else "normal")
        ax.set_xticks(range(len(order)), ["base MSTL" if m == "base" else m for m in order], rotation=20)
        ax.set_xlim(-0.5, len(order) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{level}" + (" (moyenne des 12)" if level == "Régions" else ""))
        ax.set_ylabel("MASE (plus bas = mieux)")
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{title} — moyenne ± écart-type sur {n} origines", x=0.01, ha="left", fontweight="semibold")
    return fig


def savefig(fig: plt.Figure, path) -> None:
    fig.savefig(path, bbox_inches="tight")
