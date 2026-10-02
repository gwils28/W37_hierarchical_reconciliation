"""Empreinte de la structure S : refuser un run si S a changé sans bump de version."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def s_hash(S_df: pd.DataFrame) -> str:
    """sha256 des valeurs ET des libellés : une feuille renommée change la hiérarchie."""
    S = S_df.set_index("unique_id")
    payload = json.dumps({"rows": S.index.tolist(), "cols": S.columns.tolist(),
                          "values": S.to_numpy().astype(int).tolist()})
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_check(S_df: pd.DataFrame, source: str, registry_path: str | Path) -> str:
    """Enregistre le hash d'une nouvelle version de source ; lève si une version connue a changé de S."""
    registry_path = Path(registry_path)
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    h = s_hash(S_df)
    if source in registry and registry[source] != h:
        raise RuntimeError(f"S a changé pour {source} sans bump de version "
                           f"({registry[source][:12]} -> {h[:12]}) : run refusé")
    registry[source] = h
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True))
    return h
