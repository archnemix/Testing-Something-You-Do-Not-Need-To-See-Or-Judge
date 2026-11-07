"""The brain configuration: train by predicting, at 300k neurons, on real text.

Every piece maps to a known brain mechanism (audited):
- predict every next word, learn from being wrong -- predictive processing
- surprise (1 - confidence) gates all learning -- the neuromodulator
- unexpected = fast (flashbulb), expected = slow (habituation)
- a confidently wrong guess is pushed down -- prediction-error unlearning
- speaking/asking strengthen what was used -- use-dependent plasticity
- idle pathways decay and the weakest lose neurons -- forgetting, capacity
- sentences are variable-length streams -- no padding, no truncation

The report keeps RAW OUTPUT: the surprise trace, the sentences it forms,
the answers it gives, prediction accuracy before/after. Read it and see
what the brain was able to do.

Run: python -m aster.brain_run   (also on GitHub Actions via talk-study)
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from aster.cognition.plastic import PlasticNetwork, sequence_words
from aster.speech import Speaker, form_sentence, render
from aster.talk_study import rss_mb

OUT_PATH = Path(os.environ.get("BRAIN_OUT", "action-output/brain-run/report.json"))
BUDGET = int(os.environ.get("BRAIN_BUDGET", "300000"))
DEFS = int(os.environ.get("BRAIN_DEFS", "25000"))
SENTENCES = int(os.environ.get("BRAIN_SENTENCES", "60000"))


def stream_corpus(net: PlasticNetwork, lines: list[str]) -> dict:
    """Train by prediction over a stream of variable-length lines."""
    t0 = time.perf_counter()
    surprises = []
    for line in lines:
        surprises.extend(net.stream_learn(line))
    elapsed = time.perf_counter() - t0
    half = max(1, len(surprises) // 2)
    return {
        "seconds": round(elapsed, 1),
        "mean_surprise_first_half": round(sum(surprises[:half]) / half, 4),
        "mean_surprise_second_half": round(
            sum(surprises[half:]) / max(1, len(surprises) - half), 4
        ),
    }


def main() -> None:
    from aster.talk_study import ensure_corpus

    glosses, sentences = ensure_corpus()
    rng = random.Random(42)
    train_defs = rng.sample(glosses, DEFS)
    train_sentences = rng.sample(sentences, SENTENCES)
    held_out = rng.sample(
        [s for s in sentences if s not in set(train_sentences)], 300
    )

    net = PlasticNetwork(max_neurons=BUDGET)
    lines = [f"{w} means {g}" for w, g in train_defs] + train_sentences
    print(
        f"brain run: {len(lines)} variable-length lines at {BUDGET} neurons "
        f"(predictive training); RSS {rss_mb()}MB",
        flush=True,
    )
    first_pass = stream_corpus(net, lines)

    # the same material again: surprise should collapse -- it learned to predict
    replay = rng.sample(lines, 2000)
    before = sum(sum(net.stream_learn(line)) / max(1, len(sequence_words(line)))
                 for line in replay) / len(replay)
    net2 = PlasticNetwork()
    fresh = sum(sum(net2.stream_learn(line)) / max(1, len(sequence_words(line)))
                for line in replay) / len(replay)

    # held-out prediction skill
    hits = asked = 0
    for text in held_out:
        truth = sequence_words(text)
        if len(truth) < 3:
            continue
        asked += 1
        formed = form_sentence(net, truth[:2], max_words=3)
        if len(formed) >= 3 and formed[2] == truth[2]:
            hits += 1

    # the oddball probe on fresh text (the brain's mismatch response)
    odd_standard = "the jar is on the shelf"
    odd_deviant = "the jar is on the roof"
    odd_trace = []
    for trial in range(12):
        text = odd_deviant if trial == 6 else odd_standard
        s = net.stream_learn(text)
        odd_trace.append(round(sum(s) / len(s), 3))

    # what can it say and answer? raw output
    speaker = Speaker(net=net, warm=False)
    questions = [
        f"what does {train_defs[i][0]} mean" for i in (0, 1, 2)
    ] + ["what is " + train_defs[3][0], "what does " + train_defs[4][0]]
    raw_answers = {q: speaker.respond(q) for q in questions}
    raw_speech = {
        word: render(form_sentence(net, [word]))
        for word in ("life", "world", "time", "man", "night")
    }

    report = {
        "budget_neurons": BUDGET,
        "trained_on": {
            "definitions": DEFS,
            "sentences": SENTENCES,
            "note": "all variable length; no padding, no truncation",
        },
        "neurons": net.neuron_count,
        "synapses": net.synapse_count,
        "peak_rss_mb": rss_mb(),
        "first_pass_learning": first_pass,
        "learn_to_predict": {
            "mean_surprise_fresh_network": round(fresh, 4),
            "mean_surprise_after_training": round(before, 4),
        },
        "held_out_next_word_accuracy": round(hits / max(1, asked), 3),
        "oddball_mismatch_trace": odd_trace,
        "raw_answers": raw_answers,
        "raw_speech": raw_speech,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
