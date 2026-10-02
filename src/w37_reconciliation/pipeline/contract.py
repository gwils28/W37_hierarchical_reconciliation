"""Chargement et validation minimale du contrat `reconciliation.yaml`."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONTRACT = Path(__file__).resolve().parents[3] / "configs" / "reconciliation.yaml"
REQUIRED_SECTIONS = ("hierarchy", "reconciler", "quality_gate", "outputs")


def load_contract(path: str | Path | None = None) -> dict:
    """Lit le contrat ; échoue tôt si une section obligatoire manque."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONTRACT).read_text())
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"contrat incomplet, sections manquantes : {missing}")
    return cfg
