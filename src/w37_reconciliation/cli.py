"""Point d'entrée : `uv run w37 mint` (bloc 2) et `uv run w37 job` (bloc 3)."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def _cmd_mint(args: argparse.Namespace) -> int:
    from .mint.experiment import gap_decomposition, m_greater_than_T_demo, run_mint_experiment

    exp = run_mint_experiment(n=args.n, h=args.h, seed=args.seed)
    m = exp.metrics
    print(f"series (m) = {m['m']}, feuilles (nb) = {m['nb']}, residus (T) = {m['T']}")
    print(f"ordre      : {exp.hierarchy.order}")
    print(f"lambda SS  : {m['lambda']:.6f}")
    print(f"P^2 == P   : {m['idempotence']:.3e}")
    print(f"SGS == S   : {m['unbiasedness']:.3e}")
    print(f"incoherence: {m['incoherence']:.3e}")
    print(f"ecart rel. : {m['relative_gap']:.3e}   (abs. {m['absolute_gap']:.3e} "
          f"sur des valeurs ~{m['mean_level']:.0f})")
    checks = {"SGS == S (< 1e-12)": m["unbiasedness"] < 1e-12,
              "incoherence (< 1e-9)": m["incoherence"] < 1e-9,
              "ecart lib (< 1e-3)": m["relative_gap"] < 1e-3,
              "ecart lib non nul": m["relative_gap"] > 0}
    for name, ok in checks.items():
        print(f"  [{'OK' if ok else 'KO'}] {name}")
    print("\ndecomposition de l'ecart a la bibliotheque :")
    print(gap_decomposition(exp).to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    print("\npiege m > T :", json.dumps(m_greater_than_T_demo(exp.E), default=float))
    return 0 if all(checks.values()) else 1


def _cmd_job(args: argparse.Namespace) -> int:
    from .data import build_hierarchy, make_region_channel_data
    from .pipeline import load_contract
    from .pipeline.job import run_job

    cfg = load_contract(args.config)
    # La hiérarchie de démo (région × canal) n'a ni famille ni national : poids adaptés.
    cfg["quality_gate"]["level_weights"] = {"feuille": 0.7, "region": 0.3}
    res = run_job(build_hierarchy(make_region_channel_data()), cfg, args.output_dir, source=args.source,
                  levels={"region": "Region", "feuille": "Region/Canal"})
    print(json.dumps({"lineage": res.lineage, "gate": res.report.__dict__}, indent=2, default=float))
    if res.published_to:
        print(f"PUBLIE -> {res.published_to}")
        return 0
    print(f"BLOQUE ({cfg['quality_gate']['fail_action']}) : {res.report.failures}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="w37", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mint", help="bloc 2 : MinT(shrink) maison vs HierarchicalForecast")
    p.add_argument("--n", type=int, default=80, help="nombre de trimestres")
    p.add_argument("--h", type=int, default=8, help="horizon de prévision")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_mint)

    p = sub.add_parser("job", help="bloc 3 : un run du job de réconciliation avec quality gate")
    p.add_argument("--config", type=Path, default=None,
                   help="contrat YAML (défaut : configs/reconciliation.yaml)")
    p.add_argument("--output-dir", type=Path, default=Path("03_system_design/outputs"))
    p.add_argument("--source", default="demo_region_canal@v1", help="source versionnée de la hiérarchie")
    p.set_defaults(func=_cmd_job)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
