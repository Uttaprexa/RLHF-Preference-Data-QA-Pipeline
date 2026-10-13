"""Compare full acceptance-rule configurations using saved scores (no new LLM calls)."""
import pandas as pd

d = pd.read_json("results/scored.jsonl", lines=True)
y, k = d["is_defective"].astype(bool), d["injected_defect"]
h, flip, ident = (d[c].astype(bool) for c in ["h_flag", "j_label_flip", "j_identical"])

configs = [
    ("A: current rules",                  True,  1.5, "reject"),
    ("B: drop judge 'identical'",         False, 1.5, "reject"),
    ("C: B + quality threshold 1.0",      False, 1.0, "reject"),
    ("D: C + flips go to human review",   False, 1.0, "review"),
]
print(f"{'config':<34}| {'FPR':>6} | {'prec':>5} | {'recall':>6} | {'trunc':>5} | {'flips':>5} | {'review queue':>12}")
for name, use_ident, t, flips in configs:
    lq = (d["j_chosen_relevance"] <= t) | (d["j_chosen_effort"] <= t)
    j = lq | (ident if use_ident else False) | (flip if flips == "reject" else False)
    f = h | j
    review = (flip & ~f) if flips == "review" else pd.Series(False, index=d.index)
    tp, fp = (f & y).sum(), (f & ~y).sum()
    caught = f | review  # a defect counts as caught if rejected OR sent to review
    print(f"{name:<34}| {fp / (~y).sum():>6.1%} | {tp / max(tp + fp, 1):>5.2f} | {f[y].mean():>6.1%} | "
          f"{caught[k == 'truncated'].mean():>5.0%} | {caught[k == 'label_flip'].mean():>5.0%} | "
          f"{review.mean():>12.1%}")
