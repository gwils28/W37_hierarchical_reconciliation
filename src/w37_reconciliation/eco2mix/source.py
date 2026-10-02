"""Téléchargement des données éCO2mix (ODRÉ / RTE), avec un manifeste de provenance.

Les fichiers bruts sont écrits tels que l'API les renvoie (CSV `;`) et ne sont plus jamais modifiés.
Le manifeste garde, pour chacun : l'URL exacte, la date de téléchargement, la taille, le nombre de lignes
et le sha256. C'est ce qui permet de dire, dans six mois, de quelles données vient un chiffre.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import requests

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


def _fetch(url: str, dest: Path, retries: int = 3, timeout: int = 600) -> dict:
    """Télécharge en flux vers un fichier temporaire, puis le renomme : jamais de fichier à moitié écrit."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            sha, size = hashlib.sha256(), 0
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        sha.update(chunk)
                        size += len(chunk)
            tmp.replace(dest)
            with dest.open("rb") as f:
                rows = sum(1 for _ in f) - 1                  # moins l'en-tête
            return {"url": url, "file": dest.name, "bytes": size, "rows": rows, "sha256": sha.hexdigest(),
                    "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("inaccessible")


def download_all(cfg: dict, raw_dir: Path, force: bool = False, log=print) -> dict:
    """Une requête par région (reprise possible) + le national. Retourne le manifeste."""
    src = cfg["source"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("attribution", src["attribution"])
    manifest.setdefault("files", {})

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
        manifest["files"][name] = _fetch(url, dest)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        meta = manifest["files"][name]
        log(f"    {meta['rows']:,} lignes, {meta['bytes'] / 1e6:.1f} Mo")
    return manifest


def verify(raw_dir: Path) -> list[str]:
    """Recalcule les sha256 : renvoie la liste des fichiers modifiés ou manquants depuis le téléchargement."""
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    bad = []
    for name, meta in manifest["files"].items():
        path = raw_dir / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(name)
    return bad
