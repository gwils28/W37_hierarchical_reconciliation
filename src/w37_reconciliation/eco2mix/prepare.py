"""Des CSV bruts à une hiérarchie horaire propre : contrôles qualité, changement d'heure, 30 min -> 1 h.

Trois règles, chacune testée :

1. **Tout en UTC.** En heure légale, un jour a 23 h en mars et 25 h en octobre ; en UTC, toujours 24.
2. **Le défaut d'heure de la source est corrigé, et chaque correction est comptée.** La source range
   48 demi-heures par jour LOCAL : en mars, l'heure 02:00 qui n'existe pas est remplie par un doublon ;
   en octobre, l'heure 02:00 qui se répète n'est gardée qu'une fois, d'où un trou d'une heure à 00:00 UTC.
   On retire les doublons et on bouche les trous courts par interpolation linéaire. Un trou plus long
   que `max_gap_hours` lève une erreur : on ne fabrique pas de données en silence.
3. **30 min -> 1 h par MOYENNE.** Une consommation en MW est une puissance : la moyenne de deux
   demi-heures est la puissance moyenne de l'heure. La somme donnerait des « MW × 2 », sans unité.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hierarchicalforecast.utils import aggregate

from ..data import Hierarchy
from .source import REGION_CODES


def read_regional(raw_dir: Path) -> pd.DataFrame:
    """Format long brut : region, ds (UTC, sans fuseau), y (MW). Aucune correction à ce stade."""
    frames = []
    for code, name in REGION_CODES.items():
        d = pd.read_csv(raw_dir / f"regional_{code}.csv", sep=";", encoding="utf-8-sig",
                        usecols=["date_heure", "consommation"])
        frames.append(pd.DataFrame({"region": name, "ds": _utc(d["date_heure"]), "y": d["consommation"]}))
    return pd.concat(frames, ignore_index=True)


def read_national(raw_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(raw_dir / "national.csv", sep=";", encoding="utf-8-sig")
    return pd.DataFrame({"ds": _utc(d["date_heure"]), "y": d["consommation"], "rte_j1": d["prevision_j1"]})


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


@dataclass
class SeriesQuality:
    """Ce que le nettoyage a trouvé et fait, pour UNE série. Rien n'est corrigé sans être compté ici."""

    rows: int
    duplicates_identical: int
    duplicates_conflicting: int
    missing_halfhours: int
    missing_values: int
    imputed_hours: int
    longest_gap_hours: int


def clean_half_hourly(y: pd.Series) -> tuple[pd.Series, dict]:
    """y indexé par ds (UTC, pas de 30 min, éventuellement dupliqué). Renvoie la série dédoublonnée sur
    une grille complète de 30 min (trous = NaN) et les comptes associés."""
    y = y.sort_index()
    dup = y.index.duplicated(keep=False)
    spread = y[dup].groupby(level=0).agg(lambda v: v.max() - v.min()) if dup.any() else pd.Series(dtype=float)
    y = y.groupby(level=0).mean()                     # doublon identique : sans effet ; conflictuel : moyenne
    grid = pd.date_range(y.index.min(), y.index.max(), freq="30min")
    counts = {"rows": int(len(dup)), "duplicates_identical": int((spread == 0).sum()),
              "duplicates_conflicting": int((spread > 0).sum()),
              "missing_halfhours": int(len(grid.difference(y.index))), "missing_values": int(y.isna().sum())}
    return y.reindex(grid), counts


def to_hourly(y30: pd.Series, max_gap_hours: int = 1) -> tuple[pd.Series, pd.Series, dict]:
    """Moyenne horaire, puis interpolation des trous d'au plus `max_gap_hours` heures.

    Une heure dont une seule demi-heure est connue prend cette valeur (moyenne sur ce qui existe).
    Renvoie (série horaire, masque des heures imputées, comptes).
    """
    yh = y30.resample("1h").mean()
    gaps = _gap_lengths(yh)
    longest = int(gaps.max()) if len(gaps) else 0
    if longest > max_gap_hours:
        start = gaps.idxmax()
        raise ValueError(f"trou de {longest} h à partir de {start} > max_gap_hours={max_gap_hours} : "
                         "à traiter explicitement, pas par interpolation")
    imputed = yh.isna()
    return yh.interpolate(limit_area="inside"), imputed, {"imputed_hours": int(imputed.sum()),
                                                         "longest_gap_hours": longest}


def _gap_lengths(y: pd.Series) -> pd.Series:
    """Pour chaque début de trou, sa longueur (en pas)."""
    isna = y.isna().to_numpy()
    out, i = {}, 0
    while i < len(isna):
        if isna[i]:
            j = i
            while j < len(isna) and isna[j]:
                j += 1
            out[y.index[i]] = j - i
            i = j
        else:
            i += 1
    return pd.Series(out, dtype=float)


def prepare_regions(raw: pd.DataFrame, start: str, end: str,
                    max_gap_hours: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Toutes les régions : (horaire long [region, ds, y], masque d'imputation large, tableau qualité)."""
    hourly, masks, quality = [], {}, {}
    for region, g in raw.groupby("region", sort=True):
        y30, c1 = clean_half_hourly(g.set_index("ds")["y"])
        y30 = y30[(y30.index >= start) & (y30.index < end)]
        yh, imputed, c2 = to_hourly(y30, max_gap_hours)
        quality[region] = SeriesQuality(**c1, **c2).__dict__
        masks[region] = imputed
        hourly.append(pd.DataFrame({"region": region, "ds": yh.index, "y": yh.to_numpy()}))
    return pd.concat(hourly, ignore_index=True), pd.DataFrame(masks), pd.DataFrame(quality).T


def hourly_national(nat: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Le national a changé de pas (30 min puis 15 min) : même traitement, moyenne horaire."""
    nat = nat.groupby("ds").mean().sort_index()
    nat = nat[(nat.index >= start) & (nat.index < end)]
    return nat.resample("1h").mean()


def additivity(hourly: pd.DataFrame, national: pd.DataFrame) -> pd.Series:
    """Écart national officiel − somme des 12 régions, heure par heure (MW)."""
    total = hourly.pivot(index="ds", columns="region", values="y").sum(axis=1, min_count=12)
    return (national["y"] - total).dropna()


def build_eco2mix_hierarchy(hourly: pd.DataFrame, total_name: str = "France") -> Hierarchy:
    """1 total + 12 régions. Le total est CONSTRUIT comme la somme : il est cohérent par définition."""
    df = hourly.assign(pays=total_name)
    Y_df, S_df, tags = aggregate(df=df, spec=[["pays"], ["pays", "region"]])
    return Hierarchy(Y_df=Y_df, S_df=S_df, tags=tags)


ABBREVIATIONS = {"Provence-Alpes-Côte d'Azur": "PACA", "Auvergne-Rhône-Alpes": "AURA",
                 "Bourgogne-Franche-Comté": "BFC", "Centre-Val de Loire": "Centre-VdL",
                 "Nouvelle-Aquitaine": "N.-Aquitaine", "Hauts-de-France": "HdF", "Île-de-France": "IdF",
                 "Pays de la Loire": "PdL"}


def short_name(unique_id: str) -> str:
    """'France/Provence-Alpes-Côte d'Azur' -> 'PACA' (pour les figures et les tableaux)."""
    name = unique_id.split("/")[-1]
    return ABBREVIATIONS.get(name, name)


def assert_no_missing(Y_df: pd.DataFrame) -> None:
    """Le contrôle du plan : un seul NaN dans une feuille casse la cohérence, sans message clair."""
    bad = Y_df[Y_df["y"].isna()]
    if len(bad):
        raise ValueError(f"{len(bad)} valeurs manquantes, ex. {bad.head(3).to_dict('records')}")
    counts = Y_df.groupby("unique_id")["ds"].agg(["min", "max", "count"])
    if counts.nunique().max() > 1:
        raise ValueError(f"séries de longueurs ou de bornes différentes :\n{counts}")
    if not np.isfinite(Y_df["y"].to_numpy()).all():
        raise ValueError("valeurs non finies")
