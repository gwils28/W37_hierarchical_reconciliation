"""Bloc 6 — BayesReconPy : réconcilier des prévisions de comptage (ventes intermittentes)
sans valeurs négatives.

`example` (données M5 CA_1, manifeste) -> `methods` (MinT gaussien, bottom-up, TD-cond, MixCond, avec
copies défensives) -> `scores` (négatifs, MASE, intervalles, RPS) ; `toy` : un exemple exact, à la main,
de ce que fait le conditionnement.
"""
