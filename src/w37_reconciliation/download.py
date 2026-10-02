"""Téléchargement avec provenance, commun aux blocs 5 et 6.

Un fichier brut est écrit tel que la source le renvoie, puis n'est plus jamais modifié. Son entrée de
manifeste garde l'URL exacte, la date, la taille et le sha256 : c'est ce qui permet de dire, dans six mois,
de quelles données vient un chiffre, et de détecter qu'un fichier a été modifié depuis.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests


def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 600, count_rows: bool = False) -> dict:
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
            meta = {"url": url, "file": dest.name, "bytes": size, "sha256": sha.hexdigest(),
                    "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds")}
            if count_rows:
                with dest.open("rb") as f:
                    meta["rows"] = sum(1 for _ in f) - 1          # moins l'en-tête
            return meta
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("inaccessible")


def load_manifest(raw_dir: Path, attribution: str) -> dict:
    path = raw_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest.setdefault("attribution", attribution)
    manifest.setdefault("files", {})
    return manifest


def save_manifest(raw_dir: Path, manifest: dict) -> None:
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def verify(raw_dir: Path) -> list[str]:
    """Recalcule les sha256 : renvoie la liste des fichiers modifiés ou manquants depuis le téléchargement."""
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    bad = []
    for name, meta in manifest["files"].items():
        path = raw_dir / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(name)
    return bad
