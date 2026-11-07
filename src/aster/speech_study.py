"""The big speech study: real English, three capacities, and overflow.

Teaches from Project Gutenberg sentences and WordNet definitions (real
text, downloaded to /tmp -- not committed), at budgets of 5k / 50k /
100k neurons, on datasets small enough to fit and one larger than any
budget can hold -- then checks what the network can still say and answer,
and shows sample sentences it forms on its own.

Run: .venv/bin/python -m aster.speech_study
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from aster.cognition.plastic import PlasticNetwork, sequence_words
from aster.speech import form_sentence, render

CORPUS_PATH = Path("/tmp/speech-data/corpus.json")
OUT_PATH = Path("action-output/speech-study/report.json")
BUDGETS = (5_000, 50_000, 100_000)
TIERS = {
    # name: (definitions, sentences)
    "small": (200, 800),
    "large": (3_000, 12_000),
    "overflow": (6_000, 20_000),
}
PROBES = ["life", "world", "time", "man", "night"]


def load_corpus() -> tuple[list[tuple[str, str]], list[str]]:
    if not CORPUS_PATH.exists():
        raise SystemExit(f"corpus missing: {CORPUS_PATH} (run the data download first)")
    data = json.loads(CORPUS_PATH.read_text())
    glosses = [(word, gloss) for word, gloss in data["glosses"]]
    sentences = list(data["sentences"])
    return glosses, sentences


def teach(net: PlasticNetwork, glosses, sentences) -> dict:
    """Teach definitions (as facts and word order) then sentence order."""
    t0 = time.perf_counter()
    definition_lines = []
    for word, gloss in glosses:
        net.observe(word, "means", gloss)
        definition_lines.append(f"{word} means {gloss}")
    t1 = time.perf_counter()
    for start in range(0, len(sentences), 2000):
        net.observe_corpus(sentences[start : start + 2000])
    t2 = time.perf_counter()
    net.tick(300)
    t3 = time.perf_counter()
    return {
        "definitions_seconds": round(t1 - t0, 1),
        "sentences_seconds": round(t2 - t1, 1),
        "tick_seconds": round(t3 - t2, 1),
    }


def meaning_quiz(net: PlasticNetwork, glosses, sample, rng) -> dict:
    """'what does X mean?' -- the reply must carry the definition's words."""
    words_in_glosses = {}
    for word, gloss in glosses:
        for token in sequence_words(gloss):
            words_in_glosses[token] = words_in_glosses.get(token, 0) + 1
    common = {
        w for w, n in words_in_glosses.items()
        if n > 0.15 * len(glosses)
    }
    early, late = sample[: len(sample) // 2], sample[len(sample) // 2 :]
    results = {}
    for group_name, group in (("early", early), ("late", late)):
        hits = 0
        for word, gloss in group:
            distinct = set(sequence_words(gloss)) - common - set(sequence_words(word))
            reply_words = set(form_sentence(net, sequence_words(word)))
            if len(reply_words & distinct) < 2:
                # the walk strayed; speak the recalled definition itself
                answer = net.retrieve(word, "means")
                if answer:
                    reply_words = set(
                        form_sentence(net, sequence_words(answer[0][0]))
                    )
            if len(reply_words & distinct) >= 2:
                hits += 1
        results[group_name] = {
            "asked": len(group),
            "answered": hits,
            "accuracy": round(hits / max(1, len(group)), 3),
        }
    return results


def prediction_quiz(net: PlasticNetwork, held_out: list[str]) -> float:
    """Given the first two words of unseen sentences, do the words fire right?"""
    hits = 0
    asked = 0
    for text in held_out:
        truth = sequence_words(text)
        if len(truth) < 3:
            continue
        asked += 1
        formed = form_sentence(net, truth[:2], max_words=3)
        if len(formed) >= 3 and formed[2] == truth[2]:
            hits += 1
    return round(hits / max(1, asked), 3)


def run_tier(
    tier: str, definitions: int, sentences_wanted: int, budget: int, corpus
) -> dict:
    glosses, sentences = corpus
    rng = random.Random(42)
    sample_glosses = rng.sample(glosses, definitions)
    sample_sentences = rng.sample(sentences, sentences_wanted)
    held_out = rng.sample(
        [s for s in sentences if s not in set(sample_sentences)], 200
    )
    quiz_glosses = rng.sample(sample_glosses, 24)

    net = PlasticNetwork(max_neurons=budget)
    times = teach(net, sample_glosses, sample_sentences)
    meaning = meaning_quiz(net, sample_glosses, quiz_glosses, rng)
    prediction = prediction_quiz(net, held_out)
    samples = {
        word: render(form_sentence(net, [word])) for word in PROBES
    }
    return {
        "tier": tier,
        "budget": budget,
        "taught": {
            "definitions": len(sample_glosses),
            "sentences": len(sample_sentences),
            "words_total": len({w for w, _ in sample_glosses})
            + len({t for s in sample_sentences for t in sequence_words(s)}),
        },
        "neurons": net.neuron_count,
        "synapses": net.synapse_count,
        "learning": times,
        "meaning_quiz": meaning,
        "next_word_accuracy": prediction,
        "sample_sentences": samples,
    }


def main() -> None:
    corpus = load_corpus()
    results = []
    for tier, (definitions, sentences) in TIERS.items():
        for budget in BUDGETS:
            started = time.perf_counter()
            entry = run_tier(tier, definitions, sentences, budget, corpus)
            entry["run_seconds"] = round(time.perf_counter() - started, 1)
            results.append(entry)
            meaning = entry["meaning_quiz"]
            print(
                f"{tier} @ {budget}: neurons={entry['neurons']} "
                f"early={meaning['early']['accuracy']} late={meaning['late']['accuracy']} "
                f"next-word={entry['next_word_accuracy']} ({entry['run_seconds']}s)",
                flush=True,
            )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"runs": results}, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
