"""Load HH-RLHF and normalize it to (prompt, chosen, rejected) rows."""
from __future__ import annotations

import pandas as pd

ASSISTANT_TAG = "\n\nAssistant:"


def split_transcript(text: str) -> tuple[str, str]:
    """HH-RLHF stores full transcripts. The final 'Assistant:' turn is the response;
    everything before it is the prompt (conversation context)."""
    idx = text.rfind(ASSISTANT_TAG)
    if idx == -1:
        return text.strip(), ""
    return text[:idx].strip(), text[idx + len(ASSISTANT_TAG):].strip()


def normalize(chosen_raw: str, rejected_raw: str) -> dict:
    prompt, chosen = split_transcript(chosen_raw)
    prompt_r, rejected = split_transcript(rejected_raw)
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "prompt_mismatch": prompt != prompt_r,  # a native defect in the raw data
    }


def load_hh_rlhf(n: int = 3000, split: str = "train", subset: str | None = None,
                 seed: int = 0) -> pd.DataFrame:
    """Download and sample HH-RLHF from Hugging Face.

    subset: e.g. "helpful-base" or "harmless-base" (passed as data_dir).
    """
    from datasets import load_dataset  # lazy import so tests don't need it

    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    df = pd.DataFrame([normalize(r["chosen"], r["rejected"]) for r in ds])

    n_raw = len(df)
    df = df[(df["chosen"] != "") & (df["rejected"] != "") & (~df["prompt_mismatch"])]
    print(f"Loaded {n_raw} pairs, kept {len(df)} after dropping empty/mismatched rows.")

    df = df.drop(columns="prompt_mismatch").reset_index(drop=True)
    df.insert(0, "pair_id", [f"hh-{i:06d}" for i in range(len(df))])
    return df


# ---------------------------------------------------------------- offline demo data

_TOPICS = [
    ("How do I keep basil plants alive indoors?",
     "Basil needs at least six hours of bright light a day, so place it in a south-facing window or under a grow light. Water when the top inch of soil feels dry, and pinch off flower buds so the plant keeps producing leaves.",
     "Just water it sometimes and it should be fine."),
    ("What is a good way to start learning the guitar?",
     "Start with a few open chords like G, C, D and E minor, and practice switching between them slowly for ten minutes a day. Learning a simple song you enjoy keeps practice motivating, and a clip-on tuner helps you stay in tune.",
     "Guitars are instruments with strings."),
    ("Can you explain how compound interest works?",
     "Compound interest means you earn interest on your original deposit and also on the interest already added. For example, 1000 dollars at 5 percent becomes 1050 after one year, then 1102.50 after two, because the second year's interest is calculated on 1050.",
     "Interest is money the bank gives you."),
    ("How should I prepare for a job interview?",
     "Research the company's products and recent news, then prepare short stories about your past work using the situation, task, action, result format. Practice answering common questions out loud and prepare two or three thoughtful questions to ask the interviewer.",
     "Wear clothes and show up on time."),
    ("Why is the sky blue?",
     "Sunlight contains all colors, but shorter blue wavelengths scatter off air molecules much more than longer red wavelengths. This effect, called Rayleigh scattering, sends blue light across the whole sky toward your eyes.",
     "Because it reflects the ocean."),
    ("How can I improve my sleep schedule?",
     "To fix your sleep schedule, go to bed and wake up at the same time every day, including weekends. Avoid screens and caffeine in the hours before bed, keep your bedroom cool and dark, and get morning sunlight to help reset your internal clock.",
     "Sleep more hours."),
    ("What's an easy recipe for dinner tonight?",
     "For an easy dinner, try a one-pan garlic chicken: brown chicken thighs in olive oil, add sliced garlic, cherry tomatoes and spinach, then simmer for ten minutes. Serve it over rice or pasta for a complete meal in about thirty minutes.",
     "Order a pizza."),
    ("How do I write a strong cover letter?",
     "A strong cover letter opens with the specific role and why the company interests you, then use one or two paragraphs to connect your concrete achievements to the job requirements. Close by restating your enthusiasm and inviting a conversation.",
     "Write about yourself a lot."),
    ("What should I pack for a weekend hiking trip?",
     "Bring layered clothing, a rain jacket, sturdy boots, a map, a headlamp, a first aid kit and enough water and food for the whole trip plus extra. A water filter and a small repair kit are also worth packing for longer trails.",
     "A backpack."),
    ("How do vaccines train the immune system?",
     "Vaccines expose your immune system to a harmless piece or weakened form of a pathogen. Your body produces antibodies and memory cells against it, so if you meet the real pathogen later your immune system recognizes it and responds quickly.",
     "They make you immune somehow."),
]


def make_synthetic(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Small synthetic HH-RLHF-shaped dataset for running the pipeline offline.

    Use real HH-RLHF (load_hh_rlhf) for anything you report.
    """
    import random

    rng = random.Random(seed)
    openers = ["", "Hi! ", "Quick question: ", "Hey, ", "I was wondering, "]
    rows = []
    for i in range(n):
        q, good, bad = rng.choice(_TOPICS)
        rows.append({"pair_id": f"syn-{i:06d}",
                     "prompt": f"Human: {rng.choice(openers)}{q}",
                     "chosen": good, "rejected": bad})
    return pd.DataFrame(rows)
