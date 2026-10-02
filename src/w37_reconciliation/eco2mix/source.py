"""Téléchargement des données éCO2mix (ODRÉ / RTE), avec un manifeste de provenance (`..download`).

Les fichiers bruts sont écrits tels que l'API les renvoie (CSV `;`) et ne sont plus jamais modifiés.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from ..download import fetch, load_manifest, save_manifest, verify  # noqa: F401  (verify : API publique)

REGION_CODES = {
    "11": "Île-de-France", "24": "Centre-Val de Loire", "27": "Bourgogne-Franche-Comté", "28": "Normandie",
    "32": "Hauts-de-France", "44": "Grand Est", "52": "Pays de la Loire", "53": "Bretagne",
    "75": "Nouvelle-Aquitaine", "76": "Occitanie", "84": "Auvergne-Rhône-Alpes",
    "93": "Provence-Alpes-Côte d'Azur",
}


def export_url(api: str, dataset: str, select: str, where: str) -> str:
    """URL d'export CSV de l'API Explore v2.1 d'Opendatasoft."""
    query = urlencode({"select": select, "where": where, "delimiter": ";"})
    return f"{api}/{dataset}/exports/csv?{query}"


def _where(nature: str, start: str, end: str, extra: str = "") -> str:
    clause = f'nature="{nature}" and date_heure >= date\'{start}\' and date_heure < date\'{end}\''
    return f"{extra} and {clause}" if extra else clause


def download_all(cfg: dict, raw_dir: Path, force: bool = False, log=print) -> dict:
    """Une requête par région (reprise possible) + le national. Retourne le manifeste."""
    src = cfg["source"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir, src["attribution"])

    jobs = {f"regional_{code}.csv": export_url(
        src["api"], src["regional"], "code_insee_region,libelle_region,date_heure,consommation",
        _where(src["nature"], src["start"], src["end"], f'code_insee_region="{code}"'))
        for code in REGION_CODES}
    jobs["national.csv"] = export_url(src["api"], src["national"], "date_heure,consommation,prevision_j1",
                                      _where(src["nature"], src["start"], src["end"]))

    for name, url in jobs.items():
        dest = raw_dir / name
        if dest.exists() and not force and manifest["files"].get(name, {}).get("url") == url:
            log(f"  déjà présent : {name}")
            continue
        log(f"  téléchargement : {name} …")
        manifest["files"][name] = fetch(url, dest, count_rows=True)
        save_manifest(raw_dir, manifest)
        meta = manifest["files"][name]
        log(f"    {meta['rows']:,} lignes, {meta['bytes'] / 1e6:.1f} Mo")
    return manifest
