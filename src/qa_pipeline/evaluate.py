"""Measure how well the detectors catch injected defects, and how reliable the judge is."""
from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def detection_report(df: pd.DataFrame, flag_col: str,
                     truth_col: str = "is_defective",
                     type_col: str = "injected_defect") -> dict:
    """Precision/recall/F1 overall, plus recall broken down by defect type."""
    y, f = df[truth_col].astype(bool), df[flag_col].astype(bool)
    tp, fp = int((y & f).sum()), int((~y & f).sum())
    fn, tn = int((y & ~f).sum()), int((~y & ~f).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_type = {}
    for kind, g in df[y].groupby(type_col):
        caught = int(g[flag_col].astype(bool).sum())
        lo, hi = wilson_interval(caught, len(g))
        per_type[kind] = {"n": len(g), "recall": caught / len(g), "ci95": [lo, hi]}

    return {"detector": flag_col, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "recall_by_defect": per_type}


def position_bias_report(j: pd.DataFrame) -> dict:
    """How often does swapping A/B change the judge's verdict?"""
    ok = j[~j["j_error"].astype(bool)]
    inconsistent = int((~ok["j_position_consistent"].astype(bool)).sum())
    lo, hi = wilson_interval(inconsistent, len(ok))
    return {"n_judged": len(ok), "n_errors": int(j["j_error"].sum()),
            "position_inconsistency_rate": inconsistent / len(ok) if len(ok) else 0.0,
            "ci95": [lo, hi]}


def agreement(a: pd.Series, b: pd.Series) -> dict:
    """Raw agreement and Cohen's kappa between two raters (judges or judge vs human)."""
    a, b = a.astype(str), b.astype(str)
    return {"n": len(a), "raw_agreement": float((a == b).mean()),
            "cohens_kappa": float(cohen_kappa_score(a, b))}


def batch_acceptance(flags: pd.Series, max_defect_rate: float = 0.05) -> dict:
    """Accept a batch only if the 95% upper bound on its flagged rate is under the limit."""
    k, n = int(flags.astype(bool).sum()), len(flags)
    lo, hi = wilson_interval(k, n)
    return {"n": n, "flagged": k, "flagged_rate": k / n if n else 0.0,
            "ci95": [lo, hi], "max_defect_rate": max_defect_rate,
            "accepted": hi < max_defect_rate}
