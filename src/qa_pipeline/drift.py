"""Simulate annotation campaigns over time and detect quality drift.

Each "week" is a batch drawn from the clean pool, corrupted according to that week's
scenario (e.g. a sloppy annotator joins in week 5). The detector flags rows; we then
track the flagged rate on a p-chart and test score distributions for shift.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .inject import inject_defects

DEFAULT_SCENARIO = [
    # weeks 1-4: healthy baseline
    *[{"week": w, "rate": 0.03, "mix": None, "note": "baseline"} for w in range(1, 5)],
    # weeks 5-7: a low-effort annotator joins
    *[{"week": w, "rate": 0.15, "mix": {"low_effort": 3, "truncated": 2},
       "note": "sloppy annotator"} for w in range(5, 8)],
    # week 8: fixed, back to normal
    {"week": 8, "rate": 0.03, "mix": None, "note": "annotator removed"},
    # weeks 9-12: guideline confusion causes label flips
    *[{"week": w, "rate": 0.10, "mix": {"label_flip": 1}, "note": "guideline confusion"}
      for w in range(9, 13)],
]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def simulate_campaign(clean: pd.DataFrame,
                      detector: Callable[[pd.DataFrame], pd.DataFrame],
                      scenario: list[dict] | None = None,
                      batch_size: int = 200, baseline_weeks: int = 4,
                      score_col: str = "h_chosen_words", seed: int = 0) -> pd.DataFrame:
    """Run the detector on each weekly batch and compute control-chart statistics.

    detector: takes a batch and returns it with a boolean `flag` column and `score_col`.
    """
    scenario = scenario or DEFAULT_SCENARIO
    rng = np.random.default_rng(seed)
    batches, stats = [], []

    for spec in scenario:
        idx = rng.choice(len(clean), size=min(batch_size, len(clean)), replace=False)
        batch = inject_defects(clean.iloc[idx].reset_index(drop=True),
                               rate=spec["rate"], mix=spec["mix"], seed=seed + spec["week"])
        batch = detector(batch)
        batch["week"] = spec["week"]
        batches.append(batch)
        stats.append({"week": spec["week"], "note": spec["note"], "n": len(batch),
                      "true_defect_rate": batch["is_defective"].mean(),
                      "flagged_rate": batch["flag"].mean()})

    s = pd.DataFrame(stats)
    p_bar = s.head(baseline_weeks)["flagged_rate"].mean()
    s["center"] = p_bar
    sigma = np.sqrt(p_bar * (1 - p_bar) / s["n"])
    s["ucl"] = p_bar + 3 * sigma
    s["lcl"] = (p_bar - 3 * sigma).clip(lower=0)
    s["out_of_control"] = (s["flagged_rate"] > s["ucl"]) | (s["flagged_rate"] < s["lcl"])

    base_scores = pd.concat(batches[:baseline_weeks])[score_col].to_numpy(dtype=float)
    s["psi"] = [psi(base_scores, b[score_col].to_numpy(dtype=float)) for b in batches]
    s["ks_pvalue"] = [ks_2samp(base_scores, b[score_col]).pvalue for b in batches]
    return s


def plot_campaign(s: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(s["week"], s["flagged_rate"], "o-", label="Flagged rate (detector)")
    ax1.plot(s["week"], s["true_defect_rate"], "s--", alpha=0.6, label="True defect rate")
    ax1.plot(s["week"], s["center"], "k-", lw=1, label="Center line (baseline)")
    ax1.plot(s["week"], s["ucl"], "r:", label="Control limits (±3σ)")
    ax1.plot(s["week"], s["lcl"], "r:")
    ooc = s[s["out_of_control"]]
    ax1.scatter(ooc["week"], ooc["flagged_rate"], s=160, facecolors="none",
                edgecolors="red", lw=2, label="Out of control", zorder=5)
    ax1.set_ylabel("Proportion of batch")
    ax1.set_title("p-chart: flagged-pair rate per weekly batch")
    ax1.legend(fontsize=8)

    ax2.bar(s["week"], s["psi"], color=["tab:red" if v > 0.25 else "tab:orange" if v > 0.1
                                        else "tab:green" for v in s["psi"]])
    ax2.axhline(0.1, color="orange", ls=":", lw=1)
    ax2.axhline(0.25, color="red", ls=":", lw=1)
    ax2.set_ylabel("PSI vs baseline")
    ax2.set_xlabel("Week")
    ax2.set_xticks(s["week"])
    ax2.set_title("Distribution shift of chosen-response score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
