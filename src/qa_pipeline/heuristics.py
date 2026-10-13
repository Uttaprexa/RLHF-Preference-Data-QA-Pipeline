"""Cheap, deterministic checks that run before (or alongside) the LLM judge."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

GENERIC_PATTERNS = re.compile(
    r"^(i don'?t know|sure|yes|no|maybe|ok(ay)?|not sure|that'?s a good question)\b",
    re.IGNORECASE,
)


def last_human_turn(prompt: str) -> str:
    return prompt.split("Human:")[-1].strip()


def relevance_scores(prompts: list[str], responses: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity between the last user turn and the response.

    Crude but free. Swap in sentence-transformers embeddings for a stronger signal.
    """
    queries = [last_human_turn(p) for p in prompts]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vec.fit(queries + responses)
    q, r = vec.transform(queries), vec.transform(responses)
    return np.asarray(q.multiply(r).sum(axis=1)).ravel()


def run_heuristics(df: pd.DataFrame, min_words: int = 6,
                   min_relevance: float = 0.02) -> pd.DataFrame:
    """Return heuristic features and flags aligned to df.index."""
    chosen_words = df["chosen"].str.split().str.len().fillna(0)
    rejected_words = df["rejected"].str.split().str.len().fillna(0)

    out = pd.DataFrame(index=df.index)
    out["h_chosen_words"] = chosen_words
    out["h_low_effort"] = (chosen_words < min_words) | (
        df["chosen"].str.match(GENERIC_PATTERNS) & (chosen_words < 12)
    )
    out["h_duplicate"] = df["chosen"].str.strip() == df["rejected"].str.strip()
    out["h_relevance"] = relevance_scores(df["prompt"].tolist(), df["chosen"].tolist())
    out["h_off_topic"] = (out["h_relevance"] < min_relevance) & (chosen_words >= min_words)
    out["h_length_ratio"] = (chosen_words + 1) / (rejected_words + 1)
    out["h_flag"] = out[["h_low_effort", "h_duplicate", "h_off_topic"]].any(axis=1)
    return out
