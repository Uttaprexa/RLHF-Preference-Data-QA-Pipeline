"""Inject known-bad examples so detection can be measured against ground truth."""
from __future__ import annotations

import random

import pandas as pd

DEFECT_TYPES = ["label_flip", "low_effort", "off_topic", "truncated", "duplicate"]

LOW_EFFORT_RESPONSES = [
    "I don't know.", "Sure.", "Yes.", "Maybe.", "Ok.", "Not sure, sorry.",
    "That's a good question.", "Can you ask something else?",
]


def _corrupt(row: dict, kind: str, pool: list[str], rng: random.Random) -> dict:
    row = dict(row)
    if kind == "label_flip":
        row["chosen"], row["rejected"] = row["rejected"], row["chosen"]
    elif kind == "low_effort":
        row["chosen"] = rng.choice(LOW_EFFORT_RESPONSES)
    elif kind == "off_topic":
        # A real, fluent response that was written for a different prompt.
        candidates = [r for r in pool if r != row["chosen"]]
        row["chosen"] = rng.choice(candidates)
    elif kind == "truncated":
        words = row["chosen"].split()
        row["chosen"] = " ".join(words[: max(1, len(words) // 4)])
    elif kind == "duplicate":
        row["rejected"] = row["chosen"]
    else:
        raise ValueError(f"Unknown defect type: {kind}")
    return row


def inject_defects(df: pd.DataFrame, rate: float = 0.15,
                   mix: dict[str, float] | None = None, seed: int = 0) -> pd.DataFrame:
    """Corrupt a `rate` fraction of rows. `mix` weights the defect types (default uniform).

    Adds `injected_defect` (None for clean rows) and `is_defective` columns.
    """
    rng = random.Random(seed)
    mix = mix or {k: 1.0 for k in DEFECT_TYPES}
    unknown = set(mix) - set(DEFECT_TYPES)
    if unknown:
        raise ValueError(f"Unknown defect types in mix: {unknown}")
    kinds, weights = zip(*mix.items())
    pool = df["chosen"].tolist()

    out = []
    for row in df.to_dict("records"):
        row["injected_defect"] = None
        if rng.random() < rate:
            kind = rng.choices(kinds, weights=weights, k=1)[0]
            row = _corrupt(row, kind, pool, rng)
            row["injected_defect"] = kind
        out.append(row)
    res = pd.DataFrame(out)
    res["is_defective"] = res["injected_defect"].notna()
    return res
