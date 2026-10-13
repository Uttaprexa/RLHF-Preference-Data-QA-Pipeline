import json

import pandas as pd
import pytest

from qa_pipeline.evaluate import batch_acceptance, detection_report, wilson_interval
from qa_pipeline.heuristics import run_heuristics
from qa_pipeline.inject import DEFECT_TYPES, inject_defects
from qa_pipeline.judge import JudgeParseError, MockBackend, PairJudge, apply_acceptance_criteria, parse_verdict
from qa_pipeline.load import make_synthetic, normalize


@pytest.fixture
def clean():
    return make_synthetic(400, seed=1)


def test_normalize_splits_last_assistant_turn():
    c = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: how are you?\n\nAssistant: great"
    r = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: how are you?\n\nAssistant: bad"
    row = normalize(c, r)
    assert row["chosen"] == "great" and row["rejected"] == "bad"
    assert row["prompt"].endswith("how are you?") and not row["prompt_mismatch"]


def test_injection_rate_and_ground_truth(clean):
    out = inject_defects(clean, rate=0.3, seed=0)
    assert 0.2 < out["is_defective"].mean() < 0.4
    assert set(out["injected_defect"].dropna()) <= set(DEFECT_TYPES)
    flips = out[out["injected_defect"] == "label_flip"]
    merged = flips.merge(clean, on="pair_id", suffixes=("", "_orig"))
    assert (merged["chosen"] == merged["rejected_orig"]).all()


def test_injection_is_deterministic(clean):
    a = inject_defects(clean, rate=0.2, seed=42)
    b = inject_defects(clean, rate=0.2, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_heuristics_catch_obvious_defects(clean):
    data = inject_defects(clean, rate=0.5, mix={"low_effort": 1, "duplicate": 1}, seed=0)
    h = run_heuristics(data)
    defective = data["is_defective"]
    assert h.loc[defective, "h_flag"].mean() > 0.95


def test_parse_verdict_validates():
    good = {"A": dict(helpfulness=4, relevance=5, effort=4, harmlessness=5),
            "B": dict(helpfulness=2, relevance=3, effort=2, harmlessness=5),
            "preference": "A", "confidence": 4, "issues": ["none", "bogus"], "rationale": "x"}
    v = parse_verdict("```json\n" + json.dumps(good) + "\n```")
    assert v["preference"] == "A" and v["issues"] == ["none"]
    bad = dict(good, preference="C")
    with pytest.raises(JudgeParseError):
        parse_verdict(json.dumps(bad))


def test_mock_judge_end_to_end(clean):
    data = inject_defects(clean.head(100), rate=0.3, seed=0)
    j = apply_acceptance_criteria(PairJudge(MockBackend()).judge_frame(data, max_workers=2))
    assert len(j) == len(data) and not j["j_error"].any()
    merged = data.merge(j, on="pair_id")
    flips = merged[merged["injected_defect"] == "label_flip"]
    assert flips["j_label_flip"].mean() > 0.8


def test_metrics():
    lo, hi = wilson_interval(5, 100)
    assert lo < 0.05 < hi
    df = pd.DataFrame({"is_defective": [True, True, False, False],
                       "injected_defect": ["a", "b", None, None],
                       "flag": [True, False, True, False]})
    r = detection_report(df, "flag")
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert batch_acceptance(pd.Series([False] * 1000), 0.05)["accepted"]
    assert not batch_acceptance(pd.Series([True] * 100 + [False] * 900), 0.05)["accepted"]
