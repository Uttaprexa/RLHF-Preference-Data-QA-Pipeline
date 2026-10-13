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
EMBED_MODEL = "all-MiniLM-L6-v2"
_model = None


def last_human_turn(prompt: str) -> str:
    return prompt.split("Human:")[-1].strip()


def context_window(prompt: str, max_chars: int = 600) -> str:
    """End of the conversation, so short follow-ups like 'ok thanks' keep their topic."""
    return prompt[-max_chars:]


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def tfidf_relevance(prompts: list[str], responses: list[str]) -> np.ndarray:
    """Word-overlap relevance. Kept as a baseline: it fails on conversational data
    because ~35% of good HH-RLHF replies share no keywords with the user turn."""
    queries = [last_human_turn(p) for p in prompts]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vec.fit(queries + responses)
    q, r = vec.transform(queries), vec.transform(responses)
    return np.asarray(q.multiply(r).sum(axis=1)).ravel()


def embedding_relevance(prompts: list[str], responses: list[str]) -> np.ndarray:
    """Cosine similarity of sentence embeddings: compares meaning, not words."""
    model = _get_model()
    kw = dict(batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    q = model.encode([context_window(p) for p in prompts], **kw)
    r = model.encode(list(responses), **kw)
    return (q * r).sum(axis=1)


def relevance_scores(prompts: list[str], responses: list[str],
                     method: str = "embedding") -> np.ndarray:
    if method == "embedding":
        try:
            return embedding_relevance(prompts, responses)
        except ImportError:
            print("sentence-transformers not installed; falling back to TF-IDF")
    return tfidf_relevance(prompts, responses)


def run_heuristics(df: pd.DataFrame, min_words: int = 6,
                   min_relevance: float = 0.10,
                   method: str = "embedding") -> pd.DataFrame:
    """Return heuristic features and flags aligned to df.index.

    min_relevance must be calibrated per method (see README); 0.10 is a placeholder.
    """
    chosen_words = df["chosen"].str.split().str.len().fillna(0)
    rejected_words = df["rejected"].str.split().str.len().fillna(0)

    out = pd.DataFrame(index=df.index)
    out["h_chosen_words"] = chosen_words
    out["h_low_effort"] = (chosen_words < min_words) | (
        df["chosen"].str.match(GENERIC_PATTERNS) & (chosen_words < 12)
    )
    out["h_duplicate"] = df["chosen"].str.strip() == df["rejected"].str.strip()
    out["h_relevance"] = relevance_scores(df["prompt"].tolist(), df["chosen"].tolist(), method)
    out["h_relevance_tfidf"] = tfidf_relevance(df["prompt"].tolist(), df["chosen"].tolist())
    out["h_off_topic"] = (out["h_relevance"] < min_relevance) & (chosen_words >= min_words)
    out["h_length_ratio"] = (chosen_words + 1) / (rejected_words + 1)
    out["h_flag"] = out[["h_low_effort", "h_duplicate", "h_off_topic"]].any(axis=1)
    return out
