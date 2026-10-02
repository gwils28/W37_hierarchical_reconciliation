"""Orchestration : chaque étape lit les sorties de la précédente sur disque.

    data/eco2mix/raw/        CSV bruts + manifest.json (sha256)          <- download
    data/eco2mix/interim/    horaire propre, qualité, national           <- prepare
    data/eco2mix/results/    prévisions, échelles MASE, diagnostic de W  <- backtest (main | robustness)

Les notebooks relisent ces fichiers : ils expliquent, ils ne recalculent pas le backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..data import Hierarchy
from . import prepare, source
from .backtest import BacktestSpec, origins_from_config, run_backtest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "eco2mix.yaml"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def results(self) -> Path:
        return self.root / "results"


def load_config(path: Path | str = DEFAULT_CONFIG) -> tuple[dict, Paths]:
    cfg = yaml.safe_load(Path(path).read_text())
    data_dir = Path(cfg["data_dir"])
    return cfg, Paths(data_dir if data_dir.is_absolute() else ROOT / data_dir)


def download(cfg: dict, paths: Paths, force: bool = False, log=print) -> None:
    source.download_all(cfg, paths.raw, force=force, log=log)
    bad = source.verify(paths.raw)
    if bad:
        raise RuntimeError(f"sha256 différent du manifeste : {bad}")


def run_prepare(cfg: dict, paths: Paths, log=print) -> None:
    src, prep = cfg["source"], cfg["prepare"]
    bad = source.verify(paths.raw)
    if bad:
        raise RuntimeError(f"fichiers bruts modifiés depuis le téléchargement : {bad}")
    raw = prepare.read_regional(paths.raw)
    hourly, imputed, quality = prepare.prepare_regions(raw, src["start"], src["end"])
    national = prepare.hourly_national(prepare.read_national(paths.raw), src["start"], src["end"])
    gap = prepare.additivity(hourly, national)
    by_year = gap.abs().groupby(gap.index.year).max()
    over = by_year[by_year > prep["additivity_tol_mw"]]

    paths.interim.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(paths.interim / "hourly_regions.parquet", index=False)
    imputed.to_parquet(paths.interim / "imputed_mask.parquet")
    quality.to_csv(paths.interim / "quality.csv")
    national.to_parquet(paths.interim / "national_hourly.parquet")
    log(f"  {hourly['region'].nunique()} régions × {hourly['ds'].nunique():,} heures "
        f"({hourly['ds'].min():%Y-%m-%d} → {hourly['ds'].max():%Y-%m-%d %H:%M} UTC)")
    log(f"  heures imputées : {int(quality['imputed_hours'].sum())} ; "
        f"doublons retirés : {int(quality[['duplicates_identical', 'duplicates_conflicting']].sum().sum())}")
    gap.rename("national_moins_somme").to_frame().to_parquet(paths.interim / "additivity_gap.parquet")
    log(f"  additivité national − Σ régions : médiane {gap.median():+.2f} MW ; max |écart| par année ≤ "
        f"{prep['additivity_tol_mw']} MW sauf {over.round(0).to_dict() or 'aucune'}")


def load_hierarchy(cfg: dict, paths: Paths) -> Hierarchy:
    hourly = pd.read_parquet(paths.interim / "hourly_regions.parquet")
    hier = prepare.build_eco2mix_hierarchy(hourly, cfg["prepare"]["total_name"])
    prepare.assert_no_missing(hier.Y_df)
    return hier


def run_backtest_stage(cfg: dict, paths: Paths, which: str = "main", log=print) -> dict[str, pd.DataFrame]:
    hier = load_hierarchy(cfg, paths)
    origins = origins_from_config(cfg, which)
    log(f"  {which} : {len(origins)} origines, {len(hier.order)} séries")
    out = run_backtest(hier, origins, BacktestSpec.from_config(cfg), log=log)
    paths.results.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(paths.results / f"{which}_{name}.parquet", index=False)
    return out


def load_results(paths: Paths, which: str = "main") -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(paths.results / f"{which}_{name}.parquet")
            for name in ("forecasts", "scales", "diagnostics")}
