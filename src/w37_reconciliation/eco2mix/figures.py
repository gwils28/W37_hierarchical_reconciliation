"""Figures du bloc 5. Style commun : `viz.setup()`. Une couleur d'accent, le reste en encre neutre."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

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


# --- benchmark : courbes (lineplots) --------------------------------------------------------------------
#
# Règle de couleur : trois teintes seulement, chacune avec un sens fixe dans tout le bloc.
#   bleu   = bottom-up (la méthode recommandée)      orange = MinT shrink (la méthode de référence du plan)
#   vert   = RTE J−1 (repère externe, avec météo)
# Tout le reste est en encre neutre, distingué par le tracé, et nommé par une étiquette directe en bout de
# courbe : l'identité ne repose jamais sur la couleur seule.

MID = "#9a9893"                                   # gris des autres réconciliations
STYLE = {
    "Naïf": dict(color=viz.TEXT_2, ls=(0, (1, 2)), lw=1.3),
    "SN 24 h": dict(color=viz.TEXT_2, ls="-.", lw=1.3),
    "SN": dict(color=viz.TEXT_2, ls="--", lw=1.3),
    "Moy. 4 sem.": dict(color=viz.TEXT_2, ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.3),
    "base": dict(color=viz.TEXT, ls="-", lw=1.6),
    "BU": dict(color=viz.BLUE, ls="-", lw=2.6),
    "OLS": dict(color=MID, ls="-", lw=1.3),
    "WLS struct": dict(color=MID, ls=(0, (4, 1.5)), lw=1.3),
    "WLS var": dict(color=MID, ls=(0, (1, 1)), lw=1.6),
    "MinT shrink": dict(color=viz.ORANGE, ls="-", lw=2.6),
    "MinT oos": dict(color=viz.ORANGE, ls="--", lw=2.0),
    "WLS var oos": dict(color=MID, ls="--", lw=1.6),
    "RTE J-1": dict(color=viz.AQUA, ls="--", lw=1.6),
    "RTE J-1 débiaisée": dict(color=viz.AQUA, ls="-", lw=2.2),
}
LABEL = {"SN": "SN 168 h", "base": "MSTL (base)", "RTE J-1": "RTE J−1",
         "RTE J-1 débiaisée": "RTE J−1 débiaisée", "MinT oos": "MinT, W hors éch.",
         "WLS var oos": "WLS var, W hors éch."}


def label(method: str, base_label: str = "MSTL (base)") -> str:
    return base_label if method == "base" else LABEL.get(method, method)


def _direct_labels(ax: plt.Axes, ends: dict[str, float], x: float, min_gap: float = 0.05,
                   fmt=None) -> None:
    """Étiquettes en bout de courbe, écartées verticalement pour ne pas se chevaucher (fraction d'axe)."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    items = sorted(ends.items(), key=lambda kv: kv[1])
    pos = [(v - lo) / (hi - lo) for _, v in items]
    for i in range(1, len(pos)):                                   # pousse vers le haut
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    over = pos[-1] - 0.98
    if over > 0:                                                   # puis redescend le tout si ça déborde
        pos = [p - over for p in pos]
    shift = ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans)
    trans = blended_transform_factory(ax.transData, ax.transAxes) + shift
    for (name, v), p in zip(items, pos, strict=True):
        ax.text(x, p, name if fmt is None else f"{name}  {fmt(v)}", transform=trans, fontsize=8.5,
                color=viz.TEXT, va="center", ha="left", clip_on=False)


def _plot_lines(ax: plt.Axes, x, curves: dict[str, np.ndarray], base_label: str = "MSTL (base)",
                markers: bool = False, fmt=None) -> None:
    for m, y in curves.items():
        st = STYLE.get(m, dict(color=viz.TEXT_2, lw=1.3))
        ax.plot(x, y, label=label(m, base_label), marker="o" if markers else None, ms=5,
                zorder=3 if st["color"] in (viz.BLUE, viz.ORANGE, viz.AQUA) else 2, **st)
    ax.margins(x=0.02)
    ax.figure.canvas.draw()                                        # ylim définitives avant les étiquettes
    _direct_labels(ax, {label(m, base_label): float(np.asarray(y)[-1]) for m, y in curves.items()},
                   x[-1] if not hasattr(x, "iloc") else x.iloc[-1], fmt=fmt)


def horizon_curves(h_table: pd.DataFrame, methods: list[str], title: str, base_label: str = "MSTL (base)",
                   levels=("France", "Régions"), ylim=None) -> plt.Figure:
    """MASE à chaque heure de la journée prévue (horizon 1 à 24 h), une courbe par solution."""
    fig, axes = plt.subplots(1, len(levels), figsize=(6.4 * len(levels), 4.4), layout="constrained",
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels, strict=True):
        w = h_table[h_table["level"] == level].pivot(index="horizon", columns="method", values="mase")
        curves = {m: w[m].to_numpy() for m in methods if m in w}
        _plot_lines(ax, w.index.to_numpy(), curves, base_label)
        ax.set_title(level + (" (moyenne des 12)" if level == "Régions" else ""))
        ax.set_xlabel("horizon (heures après l'origine, 00:00 UTC)")
        ax.set_xticks([1, 6, 12, 18, 24])
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("MASE (plus bas = mieux)")
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return fig


def gain_curves(df: pd.DataFrame, x: str, methods: list[str], reference: str, title: str,
                xlabel: str, levels=("France", "Régions"), base_label: str = "MSTL (base)",
                markers: bool = False, xticks=None, sharey: bool = True,
                end_values: bool = False) -> plt.Figure:
    """Gain (%) contre `reference` en fonction de `x` (horizon, mois, étude, métrique, taille de fenêtre…).

    df : colonnes level, method, `x`, gain. Au-dessus de 0 = mieux que la référence.
    end_values : écrit la valeur du dernier point à côté du nom (utile quand le dernier point est
    lui-même un résultat, ex. « toutes les origines » ; trompeur sur un axe d'horizon).
    """
    fig, axes = plt.subplots(1, len(levels), figsize=(6.4 * len(levels), 4.2), layout="constrained",
                             sharey=sharey)
    axes = np.atleast_1d(axes)
    for ax, level in zip(axes, levels, strict=True):
        w = df[df["level"] == level].pivot(index=x, columns="method", values="gain") * 100
        xs = np.arange(len(w)) if xticks == "categorical" else w.index.to_numpy()
        ax.axhline(0, color=viz.TEXT, lw=1, zorder=1)
        ax.annotate(f"= {label(reference, base_label)}", (xs[0], 0), xytext=(0, 4),
                    textcoords="offset points", fontsize=8.5, color=viz.TEXT_2, va="bottom")
        curves = {m: w[m].to_numpy() for m in methods if m in w and m != reference}
        _plot_lines(ax, xs, curves, base_label, markers=markers,
                    fmt=(lambda v: f"{v:+.1f} %") if end_values else None)
        if xticks == "categorical":
            ax.set_xticks(xs, list(w.index))
            ax.grid(axis="x", visible=False)
        ax.set_title(level + (" (moyenne des 12)" if level == "Régions" else ""))
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(f"gain vs {label(reference, base_label)} (%)\n← moins bon        meilleur →")
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return fig


def ladder(by_lvl: pd.DataFrame, steps: list[str], groups: dict[str, list[str]], title: str,
           base_label: str = "MSTL (base)") -> plt.Figure:
    """L'escalier du benchmark : MASE moyen de chaque solution, du plus simple au plus élaboré.

    Une courbe par niveau ; la pente entre deux marches dit d'où vient le gain. Les groupes (naïves,
    statistique, réconciliées, externe) sont des bandes de fond.
    """
    stats = by_lvl.groupby(["level", "method"])["mase"].mean()
    fig, ax = plt.subplots(figsize=(12, 4.6), layout="constrained")
    xs = np.arange(len(steps))
    for level, st in [("France", dict(color=viz.TEXT, ls="-", marker="o")),
                      ("Régions", dict(color=viz.TEXT_2, ls="--", marker="s"))]:
        y = np.array([stats.get((level, m), np.nan) for m in steps])
        ax.plot(xs, y, lw=1.8, ms=7, label=level + (" (moyenne des 12)" if level == "Régions" else ""), **st)
    # valeurs : la courbe du dessus est annotée au-dessus, celle du dessous en dessous
    fr = np.array([stats.get(("France", m), np.nan) for m in steps])
    rg = np.array([stats.get(("Régions", m), np.nan) for m in steps])
    for i in range(len(steps)):
        for v, other in ((fr[i], rg[i]), (rg[i], fr[i])):
            if np.isfinite(v):
                up = not np.isfinite(other) or v >= other
                ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 8 if up else -9), textcoords="offset points",
                            ha="center", va="bottom" if up else "top", fontsize=8, color=viz.TEXT)
    for m, color in [("BU", viz.BLUE), ("MinT shrink", viz.ORANGE), ("RTE J-1 débiaisée", viz.AQUA)]:
        if m in steps:
            ax.axvline(steps.index(m), color=color, lw=6, alpha=0.18, zorder=0)
    start = 0
    for g, members in groups.items():
        n = len(members)
        ax.annotate(g, ((start + start + n - 1) / 2, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9, color=viz.TEXT_2, fontweight="semibold")
        if start:
            ax.axvline(start - 0.5, color=viz.GRID, lw=1.2)
        start += n
    ax.set_xticks(xs, [label(m, base_label) for m in steps], rotation=25, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("MASE moyen (plus bas = mieux)")
    ax.legend(loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold", y=1.06)
    return fig


def convergence_plot(conv: dict[str, pd.DataFrame], title: str, ylabel: str) -> plt.Figure:
    """Moyenne cumulée du gain quand on ajoute des origines, une courbe par étude, bande 5-95 % au hasard."""
    fig, axes = plt.subplots(1, len(conv), figsize=(6.4 * len(conv), 4), layout="constrained", sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, c) in zip(axes, conv.items(), strict=True):
        ax.fill_between(c["k"], c["p05"] * 100, c["p95"] * 100, color=viz.GRID, alpha=0.9, lw=0,
                        label="ordre aléatoire : 90 % des tirages")
        ax.plot(c["k"], c["chrono"] * 100, color=viz.ORANGE, lw=2, label="ordre chronologique")
        ax.axhline(0, color=viz.TEXT, lw=1)
        ax.axvline(4, color=viz.TEXT_2, lw=1, ls=":")
        ax.annotate("4 origines", (4, ax.get_ylim()[1]), xytext=(3, -3), textcoords="offset points",
                    fontsize=8.5, color=viz.TEXT_2, va="top")
        final = c["chrono"].iat[-1] * 100
        ax.annotate(f"{final:+.1f} % après {len(c)}", (c["k"].iat[-1], final), xytext=(-4, 10),
                    textcoords="offset points", ha="right", fontsize=8.5, color=viz.TEXT)
        ax.set_title(name)
        ax.set_xlabel("nombre d'origines utilisées")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontweight="semibold")
    return fig
