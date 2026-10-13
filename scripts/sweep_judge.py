"""Re-apply judge acceptance rules at different thresholds using saved scores (no new LLM calls)."""
import pandas as pd

d = pd.read_json("results/scored.jsonl", lines=True)
y, k = d["is_defective"].astype(bool), d["injected_defect"]
clean = ~y

print("Clean pairs flagged by each judge rule (current settings):")
print(d.loc[clean, ["j_label_flip", "j_low_quality_chosen", "j_identical"]].mean().round(3).to_string())

print("\nLow-quality threshold sweep (flag if chosen relevance or effort <= t):")
print(f"{'t':<5}| {'clean flagged':>13} | {'all recall':>10} | {'truncated':>9} | {'low_effort':>10} | {'label_flip':>10}")
for t in [1.0, 1.25, 1.5, 2.0]:
    lq = (d["j_chosen_relevance"] <= t) | (d["j_chosen_effort"] <= t)
    f = lq | d["j_label_flip"].astype(bool) | d["j_identical"].astype(bool)
    r = lambda kind: f[k == kind].mean()
    print(f"{t:<5}| {f[clean].mean():>13.1%} | {f[y].mean():>10.1%} | {r('truncated'):>9.1%} | "
          f"{r('low_effort'):>10.1%} | {r('label_flip'):>10.1%}")

print("\nLabel-flip rule, requiring a minimum judge confidence:")
for cf in [0, 3, 4, 4.5]:
    fl = d["j_label_flip"].astype(bool) & (d["j_confidence"] >= cf)
    print(f"  confidence >= {cf}: clean flagged {fl[clean].mean():.1%}, "
          f"injected flips caught {fl[k == 'label_flip'].mean():.1%}")
