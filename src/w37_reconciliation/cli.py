"""Point d'entrée : `w37 mint` (bloc 2), `w37 job` (bloc 3), `w37 biais` et `w37 quantiles` (bloc 4),
`w37 eco2mix` (bloc 5), `w37 bayesrecon` (bloc 6)."""
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


def _cmd_biais(args: argparse.Namespace) -> int:
    from .fundamentals import bias_counterexample, bias_sweep

    print("Contre-exemple : biais de +15 sur le seul total, W = diag(1, 9, 9, 9)\n")
    print(bias_counterexample().round(3).to_string())
    print("\nMonte-Carlo : MAE par niveau selon le biais du total (total σ=1, feuilles σ=3)\n")
    sweep = bias_sweep([0, 2, 4, 6, 8]).pivot(index="biais", columns="méthode")
    print(sweep.round(3).to_string())
    return 0


def _cmd_quantiles(args: argparse.Namespace) -> int:
    from .fundamentals import bootstrap_coverage, coverage_experiment, rho_sweep

    print(f"Couverture empirique des quantiles 90 % (graine 37, rho = {args.rho})\n")
    print(coverage_experiment(rho=args.rho).round(3).to_string())
    print("\nBalayage en rho (niveau total)\n")
    print(rho_sweep().round(3).to_string(index=False))
    print("\nEn pratique : bootstrap des résidus in-sample (T = 200)\n")
    print(bootstrap_coverage(rho=args.rho).round(3).to_string())
    return 0


def _cmd_eco2mix(args: argparse.Namespace) -> int:
    from .eco2mix import metrics, run

    cfg, paths = run.load_config(args.config or run.DEFAULT_CONFIG)
    if args.action == "download":
        run.download(cfg, paths, force=args.force)
    elif args.action == "prepare":
        run.run_prepare(cfg, paths)
    elif args.action == "backtest":
        run.run_backtest_stage(cfg, paths, args.which)
    elif args.action == "report":
        hier = run.load_hierarchy(cfg, paths)
        res = run.load_results(paths, args.which)
        fc = res["forecasts"]
        reconciled = [c for c in metrics.methods_in(fc) if c not in ("base", "SN")]
        tol = float(cfg["inference"]["coherence_tol_abs"])
        inc = metrics.assert_coherent(fc, hier, reconciled, tol)
        by_level = metrics.mase_by_level(metrics.mase_long(fc, res["scales"], hier))
        order = ["SN", "base", *reconciled]
        print(f"MASE par niveau ({args.which}, {fc['origin'].nunique()} origines) : moyenne et écart-type\n")
        print(metrics.summary_table(by_level, order).round(3).to_string())
        print(f"\nIncohérence max (MW) après réconciliation : {inc.to_numpy().max():.2e}   [OK] < {tol:g}")
        base_gap = metrics.incoherence(fc, hier, "base").max()
        print(f"Incohérence max (MW) des prévisions de base MSTL : {base_gap:.1f}")
        k = metrics.key_figure(by_level)
        for level, v in k.items():
            print(f"MinT shrink vs BU, {level:7s} : {v['mean']:+.2%} ± {v['std']:.2%} "
                  f"(meilleur à {v['wins']}/{v['n_origins']} origines)")
    return 0


def _cmd_bayesrecon(args: argparse.Namespace) -> int:
    import pandas as pd

    from .bayesrecon import example, methods, scores

    cfg, raw = example.load_config(args.config or example.DEFAULT_CONFIG)
    example.download(cfg, raw, force=args.force)
    if args.action == "download":
        return 0
    ex = example.load_m5(raw)
    rc, alpha = cfg["reconciliation"], 1 - cfg["reconciliation"]["interval"]
    recs = methods.run_all(ex, rc["num_samples"], rc["seed"])
    table = scores.summary(recs, ex, alpha)
    pd.set_option("display.width", 200)
    print(f"M5, magasin CA_1 : {ex.A.shape[0]} agrégats, {ex.A.shape[1]} articles, prévision à 1 jour\n")
    print(table.T.round(4).to_string())
    print("\nSkill scores (%) par rapport à la base (positif = mieux) :\n")
    print(scores.skill_table(recs, ex, alpha).round(2).to_string())
    limit = float(cfg["gate"]["max_negative_share"])
    print(f"\nQuality gate du bloc 3 (max_negative_share = {limit:g}, sur les moyennes des articles) :")
    for name, share in table["articles · part moyennes < 0"].items():
        print(f"  {name:14s} {share:.4f}  {'[OK]' if share <= limit else '[BLOQUÉ]'}")
    return 0


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

    p = sub.add_parser("biais", help="bloc 4 : MinT propage le biais d'une seule prévision de base")
    p.set_defaults(func=_cmd_biais)

    p = sub.add_parser("quantiles", help="bloc 4 : MinT réconcilie des moyennes, pas des quantiles")
    p.add_argument("--rho", type=float, default=0.6, help="corrélation entre régions")
    p.set_defaults(func=_cmd_quantiles)

    p = sub.add_parser("eco2mix", help="bloc 5 : mini-projet éCO2mix (télécharger, préparer, backtester)")
    p.add_argument("action", choices=["download", "prepare", "backtest", "report"])
    p.add_argument("--which", choices=["main", "robustness"], default="main",
                   help="protocole principal (4 origines) ou étude de robustesse (2024 entière)")
    p.add_argument("--config", type=Path, default=None, help="défaut : configs/eco2mix.yaml")
    p.add_argument("--force", action="store_true", help="retélécharger même si le fichier est présent")
    p.set_defaults(func=_cmd_eco2mix)

    p = sub.add_parser("bayesrecon", help="bloc 6 : MinT face au conditionnement (BayesReconPy), exemple M5")
    p.add_argument("action", choices=["download", "run"])
    p.add_argument("--config", type=Path, default=None, help="défaut : configs/bayesrecon.yaml")
    p.add_argument("--force", action="store_true", help="retélécharger même si le fichier est présent")
    p.set_defaults(func=_cmd_bayesrecon)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
