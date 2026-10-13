"""Command-line entry point.

    python -m qa_pipeline.cli prepare --source hh --n 3000
    python -m qa_pipeline.cli run --judge anthropic --limit 500
    python -m qa_pipeline.cli drift --judge none
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .evaluate import batch_acceptance, detection_report, position_bias_report
from .heuristics import run_heuristics
from .inject import inject_defects
from .judge import AnthropicBackend, Cache, MockBackend, PairJudge, apply_acceptance_criteria
from .load import load_hh_rlhf, make_synthetic


def _make_judge(kind: str, cache_path: str) -> PairJudge | None:
    if kind == "none":
        return None
    backend = MockBackend() if kind == "mock" else AnthropicBackend()
    return PairJudge(backend=backend, cache=Cache(cache_path))


def score(df: pd.DataFrame, judge: PairJudge | None, workers: int = 8) -> pd.DataFrame:
    """Run heuristics (+ judge if given) and produce a combined `flag` column."""
    out = pd.concat([df, run_heuristics(df)], axis=1)
    if judge is None:
        out["flag"] = out["h_flag"]
        return out
    j = apply_acceptance_criteria(judge.judge_frame(df, max_workers=workers))
    out = out.merge(j, on="pair_id", how="left")
    out["flag"] = out["h_flag"] | out["j_flag"].fillna(False).astype(bool)
    return out


def cmd_prepare(a: argparse.Namespace) -> None:
    df = load_hh_rlhf(a.n, subset=a.subset, seed=a.seed) if a.source == "hh" \
        else make_synthetic(a.n, seed=a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_json(a.out, orient="records", lines=True)
    print(f"Wrote {len(df)} pairs to {a.out}")


def cmd_run(a: argparse.Namespace) -> None:
    clean = pd.read_json(a.input, lines=True)
    if a.limit:
        clean = clean.sample(n=min(a.limit, len(clean)), random_state=a.seed)
    data = inject_defects(clean.reset_index(drop=True), rate=a.rate, seed=a.seed)
    scored = score(data, _make_judge(a.judge, a.cache), a.workers)

    detectors = ["h_flag"] + (["j_flag", "flag"] if a.judge != "none" else [])
    report = {"judge": a.judge, "n": len(scored), "injection_rate": a.rate,
              "detectors": {d: detection_report(scored.fillna({d: False}), d)
                            for d in detectors},
              "batch_acceptance": batch_acceptance(scored["flag"], a.max_defect_rate)}
    if a.judge != "none":
        report["position_bias"] = position_bias_report(scored)

    os.makedirs(a.outdir, exist_ok=True)
    scored.to_json(f"{a.outdir}/scored.jsonl", orient="records", lines=True)
    with open(f"{a.outdir}/report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n=== Detection results ({a.judge} judge, n={len(scored)}) ===")
    for d, r in report["detectors"].items():
        print(f"\n[{d}] precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"f1={r['f1']:.3f} fpr={r['false_positive_rate']:.3f}")
        for kind, s in r["recall_by_defect"].items():
            print(f"    {kind:<12} recall={s['recall']:.2f}  (n={s['n']}, "
                  f"95% CI {s['ci95'][0]:.2f}-{s['ci95'][1]:.2f})")
    if "position_bias" in report:
        pb = report["position_bias"]
        print(f"\nPosition inconsistency: {pb['position_inconsistency_rate']:.1%} "
              f"of {pb['n_judged']} pairs (judge errors: {pb['n_errors']})")
    ba = report["batch_acceptance"]
    print(f"\nBatch flagged rate {ba['flagged_rate']:.1%} (95% CI upper "
          f"{ba['ci95'][1]:.1%}) -> {'ACCEPT' if ba['accepted'] else 'REJECT'}")
    print(f"\nSaved {a.outdir}/scored.jsonl and {a.outdir}/report.json")


def cmd_drift(a: argparse.Namespace) -> None:
    from .drift import plot_campaign, simulate_campaign

    clean = pd.read_json(a.input, lines=True)
    judge = _make_judge(a.judge, a.cache)
    stats = simulate_campaign(clean, lambda b: score(b, judge, a.workers),
                              batch_size=a.batch_size, seed=a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    stats.to_csv(f"{a.outdir}/drift.csv", index=False)
    plot_campaign(stats, f"{a.outdir}/drift.png")
    cols = ["week", "note", "true_defect_rate", "flagged_rate", "ucl",
            "out_of_control", "psi", "ks_pvalue"]
    print(stats[cols].round(3).to_string(index=False))
    print(f"\nSaved {a.outdir}/drift.csv and {a.outdir}/drift.png")


def main() -> None:
    p = argparse.ArgumentParser(prog="qa_pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="download/sample data")
    pp.add_argument("--source", choices=["hh", "synthetic"], default="hh")
    pp.add_argument("--n", type=int, default=3000)
    pp.add_argument("--subset", default=None, help="e.g. helpful-base, harmless-base")
    pp.add_argument("--out", default="data/pairs.jsonl")
    pp.set_defaults(func=cmd_prepare)

    for name, func in [("run", cmd_run), ("drift", cmd_drift)]:
        sp = sub.add_parser(name)
        sp.add_argument("--input", default="data/pairs.jsonl")
        sp.add_argument("--judge", choices=["none", "mock", "anthropic"], default="none")
        sp.add_argument("--cache", default="results/cache.sqlite")
        sp.add_argument("--workers", type=int, default=8)
        sp.add_argument("--outdir", default="results")
        sp.set_defaults(func=func)
        if name == "run":
            sp.add_argument("--rate", type=float, default=0.15)
            sp.add_argument("--limit", type=int, default=None)
            sp.add_argument("--max-defect-rate", type=float, default=0.05)
        else:
            sp.add_argument("--batch-size", type=int, default=200)

    for sp in sub.choices.values():
        sp.add_argument("--seed", type=int, default=0)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
