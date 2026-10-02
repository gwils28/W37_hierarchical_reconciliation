"""L'exemple M5 de BayesReconPy : un magasin (CA_1), 11 séries agrégées, 3 049 articles, prévision à 1 jour.

| niveau | séries | prévision de base fournie | information pour W |
|---|---|---|---|
| agrégats (magasin, 3 catégories, 7 rayons) | 11 | gaussienne (moyenne, écart-type) | 1 941 j de résidus |
| articles | 3 049 | pmf sur 0, 1, 2, … (modèle ADAM) | aucun résidu |
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..download import fetch, load_manifest, save_manifest, verify
from .pmf import Pmf

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "bayesrecon.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> tuple[dict, Path]:
    cfg = yaml.safe_load(Path(path).read_text())
    data_dir = Path(cfg["data_dir"])
    return cfg, (data_dir if data_dir.is_absolute() else ROOT / data_dir) / "raw"


def download(cfg: dict, raw_dir: Path, force: bool = False, log=print) -> dict:
    src = cfg["source"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir, src["attribution"].strip())
    for name in src["files"]:
        url = f"{src['base_url']}/{src['commit']}/data/{name}"
        dest = raw_dir / name
        if dest.exists() and not force and manifest["files"].get(name, {}).get("url") == url:
            log(f"  déjà présent : {name}")
            continue
        log(f"  téléchargement : {name} …")
        manifest["files"][name] = fetch(url, dest)
        save_manifest(raw_dir, manifest)
    bad = verify(raw_dir)
    if bad:
        raise RuntimeError(f"sha256 différent du manifeste : {bad}")
    return manifest


@dataclass(frozen=True)
class M5Example:
    A: np.ndarray                     # (11, 3049) : quels articles composent chaque agrégat
    upper_names: list[str]
    bottom_names: list[str]
    mu_u: np.ndarray                  # moyennes gaussiennes des 11 agrégats
    sd_u: np.ndarray
    residuals_u: np.ndarray           # (1941, 11)
    pmf_b: list[Pmf]                  # 3 049 pmf d'articles
    actual_u: np.ndarray
    actual_b: np.ndarray
    Q_u: np.ndarray                   # échelles du MASE (erreur in-sample du naïf), fournies avec l'exemple
    Q_b: np.ndarray

    @property
    def mean_b(self) -> np.ndarray:
        return np.array([p.mean for p in self.pmf_b])

    @property
    def var_b(self) -> np.ndarray:
        return np.array([p.var for p in self.pmf_b])


def load_m5(raw_dir: Path) -> M5Example:
    bad = verify(raw_dir)
    if bad:
        raise RuntimeError(f"fichiers bruts modifiés depuis le téléchargement : {bad}")
    d = pd.read_pickle(raw_dir / "M5_CA1_basefc.pkl")
    up, bo = d["upper"], d["bottom"]
    return M5Example(
        A=np.asarray(d["A"], dtype=float), upper_names=list(up), bottom_names=list(bo),
        mu_u=np.array([v["mu"] for v in up.values()]), sd_u=np.array([v["sigma"] for v in up.values()]),
        residuals_u=np.array([v["residuals"] for v in up.values()]).T,
        pmf_b=[Pmf.of(v["pmf"]) for v in bo.values()],
        actual_u=np.array([v["actual"] for v in up.values()]),
        actual_b=np.array([v["actual"] for v in bo.values()]),
        Q_u=np.asarray(d["Q_u"], dtype=float), Q_b=np.asarray(d["Q_b"], dtype=float))
