"""Train a 1,000,000-neuron brain on a large real corpus; measure how it talks.

The corpus (Project Gutenberg books + WordNet definitions) is downloaded
and parsed here if not already present. Everything is variable length --
no padding, no truncation, no context window -- because the network has
no fixed input size. Training is the same continual pathway learning as
everywhere else: telling nudges synapses up, use nudges them up further,
idle time nudges them down.

Run on GitHub Actions (4 cores / 16 GB) via `scripts/run-talk-study`.
Report: action-output/talk-study/report.json.
"""

from __future__ import annotations

import json
import random
import re
import resource
import time
import urllib.request
import zipfile
from pathlib import Path

from aster.cognition.plastic import PlasticNetwork, sequence_words
from aster.speech import Speaker, form_sentence, render

DATA_DIR = Path("/tmp/speech-data")
CORPUS_PATH = DATA_DIR / "corpus.json"
OUT_PATH = Path("action-output/talk-study/report.json")
DOWNLOADS = {
    "gutenberg.zip": (
        "https://github.com/nltk/nltk_data/raw/gh-pages/"
        "packages/corpora/gutenberg.zip"
    ),
    "wordnet.zip": (
        "https://github.com/nltk/nltk_data/raw/gh-pages/"
        "packages/corpora/wordnet.zip"
    ),
}


def rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def ensure_corpus() -> tuple[list[tuple[str, str]], list[str]]:
    """Download the real corpora if needed and parse them into items."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        target = DATA_DIR / name
        if not target.exists():
            print(f"downloading {url}", flush=True)
            urllib.request.urlretrieve(url, target)
    if not CORPUS_PATH.exists():
        word_re = re.compile(r"[a-z']+")
        sentences: list[str] = []
        g = zipfile.ZipFile(DATA_DIR / "gutenberg.zip")
        for name in g.namelist():
            if not name.endswith(".txt"):
                continue
            text = g.read(name).decode("utf-8", "ignore").casefold()
            for raw in re.split(r"[.!?:;()\n]", text):
                words = [w for w in word_re.findall(raw) if 1 <= len(w) <= 15]
                if 4 <= len(words) <= 22:
                    sentences.append(" ".join(words))
        seen: set[str] = set()
        sentences = [s for s in sentences if not (s in seen or seen.add(s))]
        glosses: list[tuple[str, str]] = []
        w = zipfile.ZipFile(DATA_DIR / "wordnet.zip")
        for name in ("wordnet/data.noun", "wordnet/data.verb", "wordnet/data.adj"):
            for line in w.read(name).decode("utf-8", "ignore").splitlines():
                if " | " not in line:
                    continue
                head, gloss = line.split(" | ", 1)
                lemmas = [
                    f.split("#")[0].replace("_", " ") for f in head.split()[4:]
                ]
                words = [
                    t for t in word_re.findall(gloss.casefold()) if 1 < len(t) <= 15
                ][:16]
                if lemmas and len(words) >= 3:
                    glosses.append((lemmas[0], " ".join(words)))
        seen_g: set[tuple[str, str]] = set()
        glosses = [g for g in glosses if not (g in seen_g or seen_g.add(g))]
        CORPUS_PATH.write_text(
            json.dumps({"glosses": glosses, "sentences": sentences})
        )
        print(f"parsed: {len(sentences)} sentences, {len(glosses)} definitions")
    data = json.loads(CORPUS_PATH.read_text())
    return data["glosses"], data["sentences"]


def teach_all(net: PlasticNetwork, glosses, sentences) -> dict:
    t0 = time.perf_counter()
    for word, gloss in glosses:
        net.observe(word, "means", gloss)
    t1 = time.perf_counter()
    net.observe_corpus([f"{w} means {g}" for w, g in glosses])
    t2 = time.perf_counter()
    net.observe_corpus(sentences)  # variable-length sentences, one stream
    t3 = time.perf_counter()
    net.tick(300)
    return {
        "definition_facts_seconds": round(t1 - t0, 1),
        "definition_order_seconds": round(t2 - t1, 1),
        "sentences_seconds": round(t3 - t2, 1),
        "items_taught": len(glosses) + len(sentences),
    }


def verbatim_probe(net, taught_texts: set[str], rng) -> dict:
    """How much of what it says is replayed storage vs recombined words."""
    probes = rng.sample(sorted(taught_texts), 100)
    taught_bigrams = {
        f"{a} {b}"
        for text in taught_texts
        for a, b in zip(sequence_words(text), sequence_words(text)[1:])
    }
    verbatim = 0
    novel_bigrams = 0
    total_bigrams = 0
    samples = []
    for text in probes:
        words = sequence_words(text)
        seed = words[:1]
        formed = form_sentence(net, seed)
        sentence = " ".join(formed)
        if sentence in taught_texts:
            verbatim += 1
        for a, b in zip(formed, formed[1:]):
            total_bigrams += 1
            if f"{a} {b}" not in taught_bigrams:
                novel_bigrams += 1
        if len(samples) < 6:
            samples.append(render(formed))
    return {
        "probes": len(probes),
        "verbatim_replay_rate": round(verbatim / max(1, len(probes)), 3),
        "novel_bigram_rate": round(novel_bigrams / max(1, total_bigrams), 3),
        "samples": samples,
    }


def question_sensitivity(speaker: Speaker, glosses, rng) -> dict:
    """Different questions about the same word must get different answers."""
    subjects = [word for word, _ in rng.sample(glosses, 30)]
    different = 0
    pairs = 0
    for subject in subjects:
        first = speaker.respond(f"what does {subject} mean")
        second = speaker.respond(f"what is {subject}")
        pairs += 1
        if first != second:
            different += 1
    return {
        "pairs": pairs,
        "different_replies": different,
        "rate": round(different / max(1, pairs), 3),
    }


def repetition_probe(glosses, sentences, rng) -> dict:
    """Is repetition 'learning' (nudging weights) or just more storage?"""
    sample = rng.sample(sentences, 4000)
    held_out = rng.sample(
        [s for s in sentences if s not in set(sample)], 200
    )

    def mean_edge_weight(net: PlasticNetwork) -> float:
        if not net._synapses:
            return 0.0
        return round(
            sum(net._synapses.values()) / len(net._synapses), 3
        )

    def next_word_accuracy(net: PlasticNetwork) -> float:
        hits = asked = 0
        for text in held_out:
            truth = sequence_words(text)
            if len(truth) < 3:
                continue
            asked += 1
            formed = form_sentence(net, truth[:2], max_words=3)
            if len(formed) >= 3 and formed[2] == truth[2]:
                hits += 1
        return round(hits / max(1, asked), 3)

    once = PlasticNetwork()
    once.observe_corpus(sample)
    twice = PlasticNetwork()
    twice.observe_corpus(sample)
    twice.observe_corpus(sample)  # same words again: nudge, not add
    doubled = PlasticNetwork()
    doubled.observe_corpus(rng.sample(sentences, 8000))
    return {
        "same_data_once": {
            "mean_edge_weight": mean_edge_weight(once),
            "next_word_accuracy": next_word_accuracy(once),
            "neurons": once.neuron_count,
        },
        "same_data_twice": {
            "mean_edge_weight": mean_edge_weight(twice),
            "next_word_accuracy": next_word_accuracy(twice),
            "neurons": twice.neuron_count,
        },
        "double_data_once": {
            "mean_edge_weight": mean_edge_weight(doubled),
            "next_word_accuracy": next_word_accuracy(doubled),
            "neurons": doubled.neuron_count,
        },
    }


def main() -> None:
    glosses, sentences = ensure_corpus()
    rng = random.Random(42)
    net = PlasticNetwork(max_neurons=1_000_000)
    print(
        f"training on {len(glosses)} definitions + {len(sentences)} sentences "
        f"(variable length, no padding); RSS {rss_mb()}MB",
        flush=True,
    )
    times = teach_all(net, glosses, sentences)
    print(
        f"taught: neurons={net.neuron_count} synapses={net.synapse_count} "
        f"RSS={rss_mb()}MB", flush=True,
    )

    speaker = Speaker(net=net, warm=False)
    quiz = rng.sample(glosses, 24)
    distinct_common = {
        "a", "an", "the", "of", "to", "in", "or", "and", "for", "with",
        "one", "two", "used", "form", "type", "kind", "part", "act",
        "state", "quality", "person", "thing", "make", "made",
    }
    meaning_hits = {"early": 0, "late": 0}
    meaning_asked = {"early": 0, "late": 0}
    for index, (word, gloss) in enumerate(quiz):
        group = "early" if index < 12 else "late"
        meaning_asked[group] += 1
        reply = set(sequence_words(speaker.respond(f"what does {word} mean")))
        distinct = (
            set(sequence_words(gloss)) - distinct_common - set(sequence_words(word))
        )
        if len(reply & distinct) >= 2:
            meaning_hits[group] += 1

    held_out = rng.sample(
        [s for s in sentences[:50000] if s not in set(sentences[::2])], 300
    )
    hits = asked = 0
    for text in held_out:
        truth = sequence_words(text)
        if len(truth) < 3:
            continue
        asked += 1
        formed = form_sentence(net, truth[:2], max_words=3)
        if len(formed) >= 3 and formed[2] == truth[2]:
            hits += 1

    taught_texts = set(sentences) | {f"{w} means {g}" for w, g in glosses}
    report = {
        "budget_neurons": 1_000_000,
        "corpus": {
            "definitions": len(glosses),
            "sentences": len(sentences),
            "note": "all items variable length; no padding, no truncation",
        },
        "learning": times,
        "neurons": net.neuron_count,
        "synapses": net.synapse_count,
        "peak_rss_mb": rss_mb(),
        "meaning_quiz": {
            group: {
                "asked": meaning_asked[group],
                "answered": meaning_hits[group],
                "accuracy": round(
                    meaning_hits[group] / max(1, meaning_asked[group]), 3
                ),
            }
            for group in ("early", "late")
        },
        "held_out_next_word_accuracy": round(hits / max(1, asked), 3),
        "verbatim_probe": verbatim_probe(net, taught_texts, rng),
        "question_sensitivity": question_sensitivity(
            Speaker(net=net, warm=False), glosses, rng
        ),
        "repetition_probe": repetition_probe(glosses, sentences, rng),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "learning"}, indent=2)[:2000])
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
