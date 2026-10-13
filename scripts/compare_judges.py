"""Compare two judges on the same pairs: agreement, Cohen's kappa, and simple ensembles.

Uses saved scores only (no new LLM calls). Both runs must use the same input, --limit,
--rate and --seed so the pairs and injected defects match.

    python scripts/compare_judges.py
    python scripts/compare_judges.py --a results/scored.jsonl --b results/llama/scored.jsonl
"""
import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

QUALITY_T = 1.0  # same threshold as config D


def judge_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Recompute rules from raw scores so both judges are compared under identical rules."""
    ok = ~d["j_error"].astype(bool)
    consistent = d["j_winner_fwd"] == d["j_winner_rev"]
    return pd.DataFrame({
        "pair_id": d["pair_id"],
        "verdict": d["j_winner_fwd"].where(consistent, "inconsistent"),
        "flip": ok & consistent & (d["j_winner_fwd"] == "rejected"),
        "low_q": ok & ((d["j_chosen_relevance"] <= QUALITY_T) | (d["j_chosen_effort"] <= QUALITY_T)),
        "h_flag": d["h_flag"].astype(bool),
    })


def rates(flag: pd.Series, y: pd.Series, k: pd.Series) -> str:
    return (f"clean flagged {flag[~y].mean():>5.1%} | injected flips caught "
            f"{flag[k == 'label_flip'].mean():>4.0%} | all defects caught {flag[y].mean():>4.0%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="results/scored.jsonl", help="first judge's scored file")
    p.add_argument("--b", default="results/llama/scored.jsonl", help="second judge's scored file")
    p.add_argument("--names", default="qwen,llama")
    args = p.parse_args()
    na, nb = args.names.split(",")

    da, db = pd.read_json(args.a, lines=True), pd.read_json(args.b, lines=True)
    truth = da[["pair_id", "is_defective", "injected_defect"]]
    m = truth.merge(judge_columns(da), on="pair_id").merge(
        judge_columns(db).drop(columns="h_flag"), on="pair_id", suffixes=(f"_{na}", f"_{nb}"))

    check = da.set_index("pair_id")["injected_defect"].fillna("-")
    other = db.set_index("pair_id")["injected_defect"].fillna("-")
    shared = check.index.intersection(other.index)
    if len(shared) < len(check) or not (check[shared] == other[shared]).all():
        print("WARNING: the two runs don't share identical pairs/defects; comparison is unreliable.\n")

    y, k = m["is_defective"].astype(bool), m["injected_defect"]
    print(f"{len(m)} pairs judged by both {na} and {nb}\n")

    print("Agreement between judges")
    for col in ["verdict", "flip", "low_q"]:
        a, b = m[f"{col}_{na}"].astype(str), m[f"{col}_{nb}"].astype(str)
        print(f"  {col:<8} raw agreement {(a == b).mean():.1%}   Cohen's kappa {cohen_kappa_score(a, b):.2f}")

    clean = m[~y]
    both_consistent = (clean[f"verdict_{na}"] != "inconsistent") & (clean[f"verdict_{nb}"] != "inconsistent")
    agree = clean[f"verdict_{na}"] == clean[f"verdict_{nb}"]
    print(f"\nOn clean pairs where both judges were order-consistent ({both_consistent.sum()} pairs),")
    print(f"they picked the same winner {agree[both_consistent].mean():.1%} of the time.")

    fa, fb = m[f"flip_{na}"], m[f"flip_{nb}"]
    print("\nLabel-flip detection, single judge vs ensemble")
    print(f"  {na:<16}" + rates(fa, y, k))
    print(f"  {nb:<16}" + rates(fb, y, k))
    print(f"  {'either (OR)':<16}" + rates(fa | fb, y, k))
    print(f"  {'both (AND)':<16}" + rates(fa & fb, y, k))

    qa, qb = m[f"low_q_{na}"], m[f"low_q_{nb}"]
    print("\nFull auto-reject rule (heuristics + judge low-quality), single vs ensemble")
    for name, lq in [(na, qa), (nb, qb), ("either (OR)", qa | qb), ("both (AND)", qa & qb)]:
        f = m["h_flag"] | lq
        tp, fp = (f & y).sum(), (f & ~y).sum()
        print(f"  {name:<16}FPR {fp / (~y).sum():>5.1%} | precision {tp / max(tp + fp, 1):.2f} | "
              f"recall {f[y].mean():>5.1%}")


if __name__ == "__main__":
    main()
