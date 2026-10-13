"""LLM-as-judge for preference pairs.

Each pair is judged twice with the response order swapped, so position bias can be
measured and inconsistent verdicts can be routed to human review.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pandas as pd

DIMENSIONS = ["helpfulness", "relevance", "effort", "harmlessness"]
ISSUE_TAGS = ["off_topic", "low_effort", "truncated", "identical_responses",
              "harmful", "factually_wrong", "none"]

JUDGE_SYSTEM = f"""You are a meticulous data-quality auditor for RLHF preference data.
You will see a conversation and two candidate final assistant responses, A and B.

Score EACH response from 1 to 5 on these dimensions:
- helpfulness: does it actually help the user with what they asked?
- relevance: does it address this specific conversation? (1 = about something else entirely)
- effort: is it substantive and complete? (1 = empty, generic filler, or cut off mid-thought)
- harmlessness: is it safe and appropriate? (5 = no concerns)

Then state which response is better overall: "A", "B", or "tie" (use "tie" only if they
are essentially identical in quality). Give your confidence from 1 to 5.
List any issues you see in either response, using only these tags: {", ".join(ISSUE_TAGS)}.

Judge content only. Do not favor a response because of its position or its length alone.

Respond with ONLY a JSON object, no markdown, in exactly this shape:
{{"A": {{"helpfulness": int, "relevance": int, "effort": int, "harmlessness": int}},
 "B": {{"helpfulness": int, "relevance": int, "effort": int, "harmlessness": int}},
 "preference": "A" | "B" | "tie", "confidence": int, "issues": [str], "rationale": str}}"""

USER_TEMPLATE = """<conversation>
{prompt}
</conversation>

<response_A>
{a}
</response_A>

<response_B>
{b}
</response_B>"""


# ---------------------------------------------------------------- parsing / validation

class JudgeParseError(ValueError):
    pass


def parse_verdict(text: str) -> dict:
    """Extract and validate the judge's JSON. Raises JudgeParseError if malformed."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise JudgeParseError(f"No JSON object found in: {text[:200]!r}")
    try:
        v = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(str(e)) from e

    for side in ("A", "B"):
        if side not in v:
            raise JudgeParseError(f"Missing scores for {side}")
        for d in DIMENSIONS:
            score = v[side].get(d)
            if not isinstance(score, (int, float)) or not 1 <= score <= 5:
                raise JudgeParseError(f"Bad score {side}.{d}={score!r}")
    if v.get("preference") not in ("A", "B", "tie"):
        raise JudgeParseError(f"Bad preference {v.get('preference')!r}")
    v.setdefault("confidence", 3)
    v["issues"] = [i for i in v.get("issues", []) if i in ISSUE_TAGS]
    v.setdefault("rationale", "")
    return v


# ---------------------------------------------------------------- backends

class AnthropicBackend:
    def __init__(self, model: str | None = None, max_tokens: int = 600):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("JUDGE_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class OllamaBackend:
    """Free, local judge via Ollama (https://ollama.com). No API key, no cost."""

    def __init__(self, model: str | None = None, host: str | None = None,
                 max_tokens: int = 500):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = (host if host.startswith("http") else f"http://{host}").rstrip("/")
        self.max_tokens = max_tokens

    def _post(self, path: str, payload: dict | None = None, timeout: int = 300) -> dict:
        import urllib.request
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(f"{self.host}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def check(self) -> None:
        """Fail fast with a clear message if Ollama isn't running or the model is missing."""
        try:
            tags = self._post("/api/tags", timeout=5)
        except Exception as e:
            raise SystemExit(f"Can't reach Ollama at {self.host}. Is the Ollama app running? ({e})")
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise SystemExit(f"Model '{self.model}' not found. Run: ollama pull {self.model}\n"
                             f"Installed: {sorted(names) or 'none'}")

    def complete(self, system: str, user: str) -> str:
        out = self._post("/api/chat", {
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_predict": self.max_tokens},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        })
        return out["message"]["content"]


class MockBackend:
    """Offline stand-in that imitates a judge using simple rules.

    ONLY for smoke-testing the pipeline without an API key. Its numbers are not
    results and must never be reported as LLM-judge performance.
    """
    model = "mock"

    @staticmethod
    def _section(user: str, tag: str) -> str:
        m = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", user, re.DOTALL)
        return m.group(1) if m else ""

    def _score(self, prompt: str, resp: str) -> dict:
        words = len(resp.split())
        q = set(re.findall(r"[a-z]{4,}", prompt.split("Human:")[-1].lower()))
        r = set(re.findall(r"[a-z]{4,}", resp.lower()))
        overlap = len(q & r) / (len(q) or 1)
        effort = 1 if words < 6 else 2 if words < 15 else 4
        relevance = 1 if overlap == 0 and words >= 6 else 4
        return {"helpfulness": min(effort, relevance), "relevance": relevance,
                "effort": effort, "harmlessness": 5}

    def complete(self, system: str, user: str) -> str:
        prompt = self._section(user, "conversation")
        a, b = self._section(user, "response_A"), self._section(user, "response_B")
        sa, sb = self._score(prompt, a), self._score(prompt, b)
        ta, tb = sum(sa.values()), sum(sb.values())
        if a.strip() == b.strip():
            pref = "tie"
        elif ta == tb:
            pref = "A"  # deliberately position-biased, like many real judges
        else:
            pref = "A" if ta > tb else "B"
        issues = ["identical_responses"] if pref == "tie" else []
        return json.dumps({"A": sa, "B": sb, "preference": pref, "confidence": 3,
                           "issues": issues, "rationale": "mock"})


# ---------------------------------------------------------------- cache

class Cache:
    def __init__(self, path: str = "results/cache.sqlite"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, k: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.conn.commit()


# ---------------------------------------------------------------- judge

@dataclass
class PairJudge:
    backend: object
    cache: Cache | None = None
    retries: int = 3

    def _call(self, prompt: str, a: str, b: str) -> dict | None:
        user = USER_TEMPLATE.format(prompt=prompt, a=a, b=b)
        k = Cache.key(getattr(self.backend, "model", "?"), JUDGE_SYSTEM, user)
        if self.cache and (hit := self.cache.get(k)):
            return json.loads(hit)
        for attempt in range(self.retries):
            try:
                verdict = parse_verdict(self.backend.complete(JUDGE_SYSTEM, user))
                if self.cache:
                    self.cache.set(k, json.dumps(verdict))
                return verdict
            except JudgeParseError:
                continue  # model produced bad JSON; retry
            except Exception:  # network / rate limit
                time.sleep(2 ** attempt)
        return None

    def judge_pair(self, row: dict) -> dict:
        """Judge in both orders and map results back onto chosen/rejected."""
        fwd = self._call(row["prompt"], row["chosen"], row["rejected"])   # A = chosen
        rev = self._call(row["prompt"], row["rejected"], row["chosen"])   # A = rejected
        if fwd is None or rev is None:
            return {"pair_id": row["pair_id"], "j_error": True}

        def winner(v: dict, chosen_side: str) -> str:
            if v["preference"] == "tie":
                return "tie"
            return "chosen" if v["preference"] == chosen_side else "rejected"

        w_fwd, w_rev = winner(fwd, "A"), winner(rev, "B")
        out = {"pair_id": row["pair_id"], "j_error": False,
               "j_winner_fwd": w_fwd, "j_winner_rev": w_rev,
               "j_position_consistent": w_fwd == w_rev,
               "j_confidence": (fwd["confidence"] + rev["confidence"]) / 2,
               "j_issues": sorted(set(fwd["issues"]) | set(rev["issues"])),
               "j_rationale": fwd["rationale"]}
        for d in DIMENSIONS:
            out[f"j_chosen_{d}"] = (fwd["A"][d] + rev["B"][d]) / 2
            out[f"j_rejected_{d}"] = (fwd["B"][d] + rev["A"][d]) / 2
        return out

    def judge_frame(self, df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
        rows = df.to_dict("records")
        done, start, lock = [0], time.time(), threading.Lock()

        def run(row: dict) -> dict:
            res = self.judge_pair(row)
            with lock:
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(rows):
                    rate = (time.time() - start) / done[0]
                    left = rate * (len(rows) - done[0]) / 60
                    print(f"  judged {done[0]}/{len(rows)} pairs, ~{left:.0f} min left",
                          flush=True)
            return res

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(run, rows))
        return pd.DataFrame(results)


def apply_acceptance_criteria(j: pd.DataFrame, min_quality: float = 1.0) -> pd.DataFrame:
    """Turn raw judge scores into flags, following rubric.md.

    Calibrated on 300 HH-RLHF pairs with qwen2.5:7b (scripts/compare_configs.py, config D):
    - Auto-reject only when the chosen response scores <= 1.0 on relevance or effort.
    - Judge 'identical_responses' tag is diagnostic only: it fired on 11.5% of clean
      pairs, and the exact-match heuristic already catches 100% of duplicates.
    - Label flips go to human review, not auto-reject: the judge disagreed with the
      annotator on 23.5% of clean pairs, so it can't be trusted to overrule labels alone.
    """
    j = j.copy()
    ok = ~j["j_error"].astype(bool)
    j["j_label_flip"] = ok & (j["j_winner_fwd"] == "rejected") & (j["j_winner_rev"] == "rejected")
    j["j_low_quality_chosen"] = ok & (
        (j["j_chosen_relevance"] <= min_quality) | (j["j_chosen_effort"] <= min_quality)
    )
    j["j_identical"] = ok & j["j_issues"].apply(
        lambda x: "identical_responses" in x if isinstance(x, list) else False)
    j["j_position_inconsistent"] = ok & ~j["j_position_consistent"].fillna(True).astype(bool)
    j["j_flag"] = j["j_low_quality_chosen"]
    j["j_needs_review"] = j["j_label_flip"] & ~j["j_flag"]
    return j
