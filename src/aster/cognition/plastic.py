"""Plastic neuron-and-synapse substrate: knowledge lives in pathway strengths.

There is no fact dictionary here. Telling the network things strengthens
pathways between neurons (Hebbian: co-active neurons wire together), answers
form by activation spreading along those pathways, used pathways strengthen
further, and unused pathways decay and are pruned. New concepts recruit new
neurons (structural growth) and long disuse removes them again.

Input is a stream of variable-length phrases: no padding, no context window,
no fixed-size buffers. The whole network state is the memory, and every
interaction -- telling or asking -- changes it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KNOWN_BELOW = 0.5  # a pathway at or above this strength counts as "already known"
ANSWER_AT = 0.2  # ensemble activation needed to fire as an answer
RECRUIT_AT = 0.75  # identity overlap below which a new ensemble is born
CUE_AT = 0.4  # weaker overlap still cues an existing concept (associative recall)
PREFIX_COVER = 0.75  # a cue's word-beginnings must mostly match the memory
FALLOFF = 0.8  # activation multiplier per spreading hop
USE_BOOST = 0.5  # retrieval re-strengthening factor (use-dependent plasticity)
IMPRINT_BOOST = 0.2
CONSOLIDATION = 0.8  # strong pathways are protected from decay (selective forgetting)
SAVINGS = 0.6  # relearning boost from the dormant engram peak (relearn faster)
DETERMINERS = {"the", "a", "an", "my", "your", "this", "that"}


@dataclass(frozen=True, slots=True)
class Ensemble:
    """A small pool of neurons that fires together for one concept."""

    identifier: int
    neurons: tuple[int, ...]
    label: str
    born: int


def sequence_words(text: str) -> list[str]:
    """The words of a sentence in order, punctuation stripped (speech)."""
    return re.findall(r"[a-z0-9']+", text.casefold())


def phrase_features(phrase: str) -> set[str]:
    """Perception: a phrase lights up spelling-feature neurons (case folded).

    Determiners carry no identity ("a ball" and "the ball" are the same
    object), so perception keeps content words -- the brain remembers the
    detail that matters, not the exact string.
    """
    cleaned = re.sub(r"[^a-z0-9]+", " ", phrase.casefold()).strip()
    words = [word for word in cleaned.split() if word]
    words = [word for word in words if word not in DETERMINERS] or words
    features: set[str] = set()
    if phrase.casefold().startswith(("'", '"')) and words:
        # a quoted literal is its own word ('hello' is not hello): give it
        # a distinct word-beginning so it is a distinct memory
        features.add("quote-start")
        features.add(f"pre:'{words[0][:3]}")
    if not words and phrase.strip():
        # symbols are words too (code: ( ) = + ...) -- their own word class
        features.add(f"pre:|{phrase.strip()[:3]}")
    for word in words:
        features.add(f"len:{min(len(word), 12)}")
        for character in word:
            features.add(f"ch:{character}")
        for index in range(len(word) - 1):
            features.add(f"bg:{word[index:index + 2]}")
        features.add(f"pre:{word[:3]}")
        features.add(f"suf:{word[-3:]}")
    features.add(f"words:{min(len(words), 8)}")
    return features


class PlasticNetwork:
    """Use-dependent Hebbian network of neuron ensembles and gated synapses."""

    def __init__(
        self,
        learning_rate: float = 0.6,
        decay_rate: float = 0.002,
        prune_below: float = 0.02,
        spread_hops: int = 4,
        ensemble_size: int = 3,
        max_neurons: int | None = None,
    ) -> None:
        if learning_rate <= 0.0 or not 0.0 < decay_rate < 1.0:
            raise ValueError("learning_rate must be positive and decay_rate in (0, 1)")
        if spread_hops < 1 or ensemble_size < 1:
            raise ValueError("spread_hops and ensemble_size must be at least 1")
        if max_neurons is not None and max_neurons < ensemble_size:
            raise ValueError("max_neurons must fit at least one ensemble")
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.prune_below = prune_below
        self.spread_hops = spread_hops
        self.ensemble_size = ensemble_size
        self.max_neurons = max_neurons
        self.clock = 0
        self._next_neuron = 0
        self._next_ensemble_id = 0
        self._protected: set[int] = set()
        self._ensembles: dict[int, Ensemble] = {}
        self._ensemble_features: dict[int, set[str]] = {}
        self._feature_ensembles: dict[str, set[int]] = {}
        self._imprints: dict[int, dict[str, float]] = {}
        self._synapses: dict[tuple[int, int, int], float] = {}
        self._outgoing: dict[int, set[tuple[int, int]]] = {}
        self._incoming: dict[int, set[tuple[int, int]]] = {}
        self._peaks: dict[tuple[int, int, int], float] = {}
        self._neuron_owner: dict[int, int] = {}
        self._strength: dict[int, float] = {}
        self._context_gates: dict[tuple[str, ...], int] = {}
        self._next_context_gate = -1

    # -- telling -----------------------------------------------------------

    def observe(self, subject: str, relation: str, object_phrase: str) -> bool:
        """Tell the network a fact; strengthen pathways.

        Returns True when the fact is new (the network grew, or the pathway
        was weak), False when the same fact is merely being repeated.
        """
        self.clock += 1
        self._protected: set[int] = set()
        try:
            subject_ens = self._recognize(subject, recruit=True)
            assert subject_ens
            self._protected.add(subject_ens.identifier)
            relation_ens = self._recognize(relation, recruit=True)
            assert relation_ens
            self._protected.add(relation_ens.identifier)
            object_ens = self._recognize(object_phrase, recruit=True)
            assert object_ens
            self._protected.add(object_ens.identifier)
        finally:
            self._protected = set()
        strength = self._path_strength(subject_ens, object_ens, relation_ens)
        for pre in subject_ens.neurons:
            for post in object_ens.neurons:
                self._strengthen(pre, post, relation_ens.identifier)
        self._strengthen_imprints(subject, subject_ens)
        self._strengthen_imprints(relation, relation_ens)
        self._strengthen_imprints(object_phrase, object_ens)
        return strength < KNOWN_BELOW

    def practice_sentence(self, text: str) -> list[str]:
        """Use-dependent plasticity: saying a sentence strengthens its pathways.

        Nothing new is added -- no new neurons, no new memories, no new
        gates. Speaking is practice: the links that just fired get nudged
        up, the loop that makes a skill better the more it is used. (Idle
        pathways still decay in `tick` -- use it or lose it.)
        """
        words = sequence_words(text)
        ensembles = [self._recognize(word, recruit=False) for word in words]
        relation_ens = self._recognize("comes after", recruit=False)
        if relation_ens is None:
            return words
        for index, (first, second) in enumerate(zip(ensembles, ensembles[1:])):
            if first is None or second is None:
                continue
            self._strengthen(first.neurons[0], second.neurons[0], relation_ens.identifier)
            if index >= 1:
                gate = self._context_gates.get((words[index - 1], words[index]))
                if gate is not None:
                    self._strengthen(first.neurons[0], second.neurons[0], gate)
            if index >= 2:
                gate = self._context_gates.get(
                    (words[index - 2], words[index - 1], words[index])
                )
                if gate is not None:
                    self._strengthen(first.neurons[0], second.neurons[0], gate)
        return words

    def stream_learn(self, text: str, words: list[str] | None = None) -> list[float]:
        """Learn the way a brain learns: predict, then be surprised.

        Before each word, the network predicts it from what came before.
        The surprise (1 minus confidence in the truth) is the neuromodulator:
        the unexpected is learned hard and fast (flashbulb), the expected
        barely moves (habituation), and a confidently wrong guess is nudged
        down (prediction error). Returns the surprise felt at each step --
        the same signal an EEG reads as an event-related response. Pass
        `words` to stream pre-tokenized units (e.g. code tokens).
        """
        words = (list(words) if words is not None else sequence_words(text)) + ["."]
        relation_ens = self._recognize("comes after", recruit=True)
        assert relation_ens
        surprises: list[float] = []
        ensembles: list[Ensemble] = []
        known: dict[str, Ensemble] = {}
        for index, word in enumerate(words):
            ensemble = known.get(word)
            if ensemble is None or ensemble.identifier not in self._ensembles:
                ensemble = self._recognize(word, recruit=True)
                assert ensemble
                known[word] = ensemble
            ensembles.append(ensemble)
            if index == 0:
                continue
            context = tuple(words[max(0, index - 3) : index - 1])
            predicted = self.successors(
                words[index - 1], context, ensemble=ensembles[index - 1]
            )
            confidence = 0.0
            wrong_label = None
            if predicted:
                if predicted[0][0] == word:
                    confidence = predicted[0][1]
                else:
                    wrong_label = predicted[0][0]
            surprise = 1.0 - confidence
            surprises.append(surprise)
            # the true path: the unexpected is learned hard, the expected creeps
            previous = ensembles[index - 1]
            gates = [relation_ens.identifier]
            if index >= 2:
                gates.append(
                    self._context_gate((words[index - 2], words[index - 1]))
                )
            if index >= 3:
                gates.append(
                    self._context_gate(
                        (words[index - 3], words[index - 2], words[index - 1])
                    )
                )
            for synapse_gate in gates:
                key = (previous.neurons[0], ensemble.neurons[0], synapse_gate)
                weight = self._synapses.get(key, 0.0)
                step = (
                    self.learning_rate
                    * (1.0 - weight)
                    * (0.3 + 1.7 * surprise)
                )
                new_weight = min(1.0, weight + step)
                self._peaks[key] = max(self._peaks.get(key, 0.0), new_weight)
                self._apply(key, new_weight, new_weight - weight)
            # prediction error: the confident wrong guess is pushed down
            if wrong_label is not None and surprise > 0.2:
                wrong_ens = self._recognize(wrong_label, recruit=False)
                if wrong_ens is not None:
                    weaken_keys = [
                        (previous.neurons[0], wrong_ens.neurons[0], synapse_gate)
                        for synapse_gate in gates
                    ]
                    self._weaken(weaken_keys, 0.3 * surprise)
            self._strengthen_imprints(word, ensemble)
        return surprises

    def _weaken(self, keys, amount: float) -> None:
        """Prediction error: nudge wrong pathways down (targeted, not global)."""
        for key in keys:
            weight = self._synapses.get(key, 0.0)
            if weight <= 0.0:
                continue
            new_weight = weight * (1.0 - amount)
            if new_weight < self.prune_below:
                self._drop_synapses([key])
            else:
                self._apply(key, new_weight, new_weight - weight)

    def observe_sequence(
        self, text: str, words: list[str] | None = None
    ) -> list[str]:
        """Teach the word order of a sentence: w1 then w2 then w3 ...

        Word order lives in the same synapses as everything else. Each step
        is wired under three gates: the plain 'comes after' relation (any
        context), the previous word pair, and the previous word triple --
        so the same word can follow different words differently, exactly
        like spoken language, while unused order still decays. Every
        sentence also learns where it ENDS (a final '.' word), so speech
        stops at the sentence boundary instead of running on. Pass `words`
        to teach pre-tokenized units (e.g. code tokens).
        """
        words = (list(words) if words is not None else sequence_words(text)) + ["."]
        self.clock += 1
        ensembles: list[Ensemble] = []
        for word in words:
            ensemble = self._recognize(word, recruit=True)
            assert ensemble
            self._strengthen_imprints(word, ensemble)
            ensembles.append(ensemble)
        relation_ens = self._recognize("comes after", recruit=True)
        assert relation_ens
        self._strengthen_imprints("comes after", relation_ens)
        for index, (first, second) in enumerate(zip(ensembles, ensembles[1:])):
            self._strengthen(first.neurons[0], second.neurons[0], relation_ens.identifier)
            if index >= 1:
                gate = self._context_gate((words[index - 1], words[index]))
                self._strengthen(first.neurons[0], second.neurons[0], gate)
            if index >= 2:
                gate = self._context_gate(
                    (words[index - 2], words[index - 1], words[index])
                )
                self._strengthen(first.neurons[0], second.neurons[0], gate)
        return words

    def _context_gate(self, words: tuple[str, ...]) -> int:
        """A gate code for one word context (negative: never an ensemble id)."""
        code = self._context_gates.get(words)
        if code is None:
            code = self._next_context_gate
            self._next_context_gate -= 1
            self._context_gates[words] = code
        return code

    def observe_corpus(self, sentences: list[str]) -> int:
        """Teach many sentences at once (words first, then the order links)."""
        self.clock += 1
        parsed = [sequence_words(text) + ["."] for text in sentences]
        self._recognize("comes after", recruit=True)
        known: dict[str, Ensemble] = {}
        for words in parsed:
            for word in words:
                if word not in known:
                    ensemble = self._recognize(word, recruit=True)
                    assert ensemble
                    known[word] = ensemble
        relation_ens = self._recognize("comes after", recruit=True)
        assert relation_ens
        wired = 0
        for words in parsed:
            for index, (first, second) in enumerate(zip(words, words[1:])):
                first_ens = known.get(first)
                second_ens = known.get(second)
                if first_ens is None or second_ens is None:
                    continue
                if first_ens.identifier not in self._ensembles:
                    continue  # forgotten under capacity pressure
                if second_ens.identifier not in self._ensembles:
                    continue
                self._strengthen(
                    first_ens.neurons[0], second_ens.neurons[0], relation_ens.identifier
                )
                if index >= 1:
                    gate = self._context_gate((words[index - 1], words[index]))
                    self._strengthen(first_ens.neurons[0], second_ens.neurons[0], gate)
                if index >= 2:
                    gate = self._context_gate(
                        (words[index - 2], words[index - 1], words[index])
                    )
                    self._strengthen(first_ens.neurons[0], second_ens.neurons[0], gate)
                wired += 1
        return wired

    def successors(
        self,
        phrase: str,
        context: tuple[str, ...] = (),
        ensemble: Ensemble | None = None,
    ) -> list[tuple[str, float]]:
        """Which learned words can follow this one, and how strongly.

        With a context of earlier words the answer uses the richest learned
        gate first (this word after those words), backing off to plain
        word order when the context itself was never heard.
        """
        if ensemble is None:
            ensemble = self._recognize(phrase, recruit=False)
        if ensemble is None:
            return []
        relation_ens = self._recognize("comes after", recruit=False)
        if relation_ens is None:
            return []
        gates = []
        if len(context) >= 2:
            gates.append(self._context_gates.get((context[-2], context[-1], phrase)))
        if context:
            gates.append(self._context_gates.get((context[-1], phrase)))
        gates.append(relation_ens.identifier)
        owner = self._neuron_owner
        for gate in gates:
            if gate is None:
                continue
            scored: dict[int, float] = {}
            for pre in ensemble.neurons:
                for post, post_gate in self._outgoing.get(pre, ()):
                    if post_gate != gate:
                        continue
                    identifier = owner.get(post)
                    if identifier is None or identifier == ensemble.identifier:
                        continue
                    weight = self._synapses.get((pre, post, post_gate), 0.0)
                    if weight > scored.get(identifier, 0.0):
                        scored[identifier] = weight
            if scored:
                return sorted(
                    (
                        (self._ensembles[identifier].label, weight)
                        for identifier, weight in scored.items()
                        if weight >= self.prune_below
                    ),
                    key=lambda item: -item[1],
                )
        return []

    # -- answering ---------------------------------------------------------

    def retrieve(self, subject: str, relation: str) -> list[tuple[str, int]]:
        """Answer by letting activation spread along gated pathways.

        Returns (label, hop) pairs: hop 1 is directly linked, higher hops
        arrived by following chains. Retrieval also re-strengthens the paths
        it used -- the more a thing is used, the stronger it gets.
        """
        self.clock += 1
        subject_ens = self._recognize(subject, recruit=False)
        relation_ens = self._recognize(relation, recruit=False)
        if subject_ens is None or relation_ens is None:
            return []
        gate = relation_ens.identifier
        activation: dict[int, float] = {}
        for neuron in subject_ens.neurons:
            activation[neuron] = 1.0
        cue_ids = {subject_ens.identifier, relation_ens.identifier}
        answered: list[tuple[str, int]] = []
        answered_ids: set[int] = set()
        for hop in range(1, self.spread_hops + 1):
            firing = self._spread(activation, gate)
            winners = self._winners(firing, cue_ids | answered_ids)
            for ensemble in winners:
                answered.append((ensemble.label, hop))
                answered_ids.add(ensemble.identifier)
            if not firing:
                break
            activation = {neuron: value * FALLOFF for neuron, value in firing.items()}
        self._consolidate_use(subject_ens, relation_ens, answered_ids)
        return answered

    # -- growth and forgetting --------------------------------------------

    def tick(self, times: int = 1) -> None:
        """Let idle time pass: unused pathways decay, dead cells are pruned.

        Telling and asking consolidate (strengthen); only idle time decays.
        Decay is selective: well-consolidated (strong) pathways fade far
        more slowly than weak one-off traces, so used skills survive while
        trivia quietly disappears.
        """
        if times < 1:
            raise ValueError("times must be at least 1")
        self.clock += times
        self._synapses = {
            key: self._decayed(weight, times)
            for key, weight in self._synapses.items()
            if self._decayed(weight, times) >= self.prune_below
        }
        imprint_floor = (1.0 - self.decay_rate * 0.1) ** times
        self._imprints = {
            neuron: {
                feature: weight * imprint_floor
                for feature, weight in weights.items()
                if weight * imprint_floor >= self.prune_below
            }
            for neuron, weights in self._imprints.items()
        }
        self._prune_dead_neurons()
        self._reindex()

    def _decayed(self, weight: float, times: int) -> float:
        rate = self.decay_rate * (1.0 - CONSOLIDATION * weight)
        return weight * (1.0 - rate) ** times

    def _prune_dead_neurons(self) -> None:
        owned = {
            neuron
            for ensemble in self._ensembles.values()
            for neuron in ensemble.neurons
        }
        self._synapses = {
            key: weight
            for key, weight in self._synapses.items()
            if key[0] in owned and key[1] in owned
        }
        connected: set[int] = set()
        for pre, post, _ in self._synapses:
            connected.add(pre)
            connected.add(post)
        for neuron, weights in self._imprints.items():
            if weights:
                connected.add(neuron)
        live = {
            identifier: Ensemble(
                ensemble.identifier,
                tuple(n for n in ensemble.neurons if n in connected),
                ensemble.label,
                ensemble.born,
            )
            for identifier, ensemble in self._ensembles.items()
        }
        self._ensembles = {
            identifier: ensemble
            for identifier, ensemble in live.items()
            if ensemble.neurons
        }
        self._ensemble_features = {
            identifier: self._ensemble_features.get(identifier, set())
            for identifier in self._ensembles
        }

    # -- inspection --------------------------------------------------------

    @property
    def neuron_count(self) -> int:
        return sum(len(ensemble.neurons) for ensemble in self._ensembles.values())

    @property
    def synapse_count(self) -> int:
        return len(self._synapses)

    def known_fact_count(self) -> int:
        """How many subject-object pathways count as learned (a projection)."""
        owner: dict[int, int] = {}
        for identifier, ensemble in self._ensembles.items():
            for neuron in ensemble.neurons:
                owner[neuron] = identifier
        grouped: dict[tuple[int, int, int], list[float]] = {}
        for (pre, post, gate), weight in self._synapses.items():
            key = (owner.get(pre, -1), owner.get(post, -1), gate)
            grouped.setdefault(key, []).append(weight)
        return sum(
            1
            for weights in grouped.values()
            if sum(weights) / len(weights) >= KNOWN_BELOW
        )

    def path_strength(self, subject: str, relation: str, object_phrase: str) -> float:
        """Current mean strength of one fact pathway (0.0 to 1.0)."""
        subject_ens = self._recognize(subject, recruit=False)
        relation_ens = self._recognize(relation, recruit=False)
        object_ens = self._recognize(object_phrase, recruit=False)
        if subject_ens is None or relation_ens is None or object_ens is None:
            return 0.0
        return self._path_strength(subject_ens, object_ens, relation_ens)

    def snapshot(self) -> dict[str, object]:
        """JSON-serializable state: this IS the long-term memory."""
        return {
            "clock": self.clock,
            "next_neuron": self._next_neuron,
            "next_ensemble_id": self._next_ensemble_id,
            "ensembles": [
                {
                    "identifier": e.identifier,
                    "neurons": list(e.neurons),
                    "label": e.label,
                    "born": e.born,
                }
                for e in sorted(self._ensembles.values(), key=lambda e: e.identifier)
            ],
            "imprints": [
                {"feature": f, "neuron": n, "weight": round(w, 6)}
                for n, weights in sorted(self._imprints.items())
                for f, w in sorted(weights.items())
            ],
            "synapses": [
                {"pre": a, "post": b, "gate": g, "weight": round(w, 6)}
                for (a, b, g), w in sorted(self._synapses.items())
            ],
            "peaks": [
                {"pre": a, "post": b, "gate": g, "peak": round(p, 6)}
                for (a, b, g), p in sorted(self._peaks.items())
            ],
            "context_gates": [
                {"words": list(words), "code": code}
                for words, code in sorted(self._context_gates.items())
            ],
        }

    def restore(self, payload: dict[str, object]) -> None:
        self.clock = int(payload.get("clock", 0))
        self._next_neuron = int(payload.get("next_neuron", 0))
        self._next_ensemble_id = int(payload.get("next_ensemble_id", 0))
        self._ensembles = {
            int(item["identifier"]): Ensemble(
                int(item["identifier"]),
                tuple(int(n) for n in item["neurons"]),
                str(item["label"]),
                int(item["born"]),
            )
            for item in payload.get("ensembles", [])  # type: ignore[union-attr]
        }
        self._imprints = {}
        for item in payload.get("imprints", []):  # type: ignore[union-attr]
            neuron = int(item["neuron"])
            self._imprints.setdefault(neuron, {})[str(item["feature"])] = float(
                item["weight"]
            )
        self._synapses = {
            (int(item["pre"]), int(item["post"]), int(item["gate"])): float(
                item["weight"]
            )
            for item in payload.get("synapses", [])  # type: ignore[union-attr]
        }
        self._peaks = {
            (int(item["pre"]), int(item["post"]), int(item["gate"])): float(
                item["peak"]
            )
            for item in payload.get("peaks", [])  # type: ignore[union-attr]
        }
        self._ensemble_features = {}
        for identifier, ensemble in self._ensembles.items():
            features: set[str] = set()
            for neuron in ensemble.neurons:
                features.update(self._imprints.get(neuron, {}))
            self._ensemble_features[identifier] = features
        self._next_ensemble_id = max(
            self._next_ensemble_id, max(self._ensembles, default=-1) + 1
        )
        self._context_gates = {}
        codes = [0]
        for item in payload.get("context_gates", []):  # type: ignore[union-attr]
            words = tuple(str(word) for word in item["words"])
            code = int(item["code"])
            self._context_gates[words] = code
            codes.append(code)
        self._next_context_gate = min(codes) - 1
        self._reindex()

    # -- internals ---------------------------------------------------------

    def _recognize(self, phrase: str, recruit: bool) -> Ensemble | None:
        features = phrase_features(phrase)
        if not features:
            return None
        cue_prefixes = {f for f in features if f.startswith("pre:")}
        # candidates must share enough word-beginnings with the cue; for
        # short phrases that means all of them, so intersect the prefix sets
        prefix_sets = [self._feature_ensembles.get(f, set()) for f in cue_prefixes]
        needed = PREFIX_COVER * len(cue_prefixes)
        if needed > len(cue_prefixes) - 1 and prefix_sets:
            candidate_ids: set[int] = set(prefix_sets[0])
            for holders in prefix_sets[1:]:
                candidate_ids &= holders
        else:
            candidate_ids = set()
            for holders in prefix_sets:
                candidate_ids |= holders
        scored: list[tuple[float, Ensemble]] = []
        for identifier in candidate_ids:
            ensemble = self._ensembles.get(identifier)
            if ensemble is None:
                continue
            stored = self._ensemble_features.get(identifier, set())
            shared = features & stored
            if not shared:
                continue
            # pattern separation: a cue reaches one memory by its word
            # beginnings, not by sharing stray letters ("vel" is not "tivel")
            shared_prefixes = {f for f in shared if f.startswith("pre:")}
            if len(shared_prefixes) < PREFIX_COVER * len(cue_prefixes):
                continue
            first_weights = self._imprints.get(ensemble.neurons[0], {})
            weight = sum(first_weights.get(feature, 0.0) for feature in shared)
            scored.append((weight / len(features | stored), ensemble))
        scored.sort(key=lambda item: item[0], reverse=True)
        threshold = RECRUIT_AT if recruit else CUE_AT
        if scored and scored[0][0] >= threshold:
            if len(scored) == 1 or scored[0][0] > scored[1][0]:
                return scored[0][1]  # one clear memory, not a look-alike tie
        if not recruit:
            return None
        return self._recruit(phrase, features)

    def _recruit(self, phrase: str, features: set[str]) -> Ensemble:
        if self.max_neurons is not None:
            while self.neuron_count + self.ensemble_size > self.max_neurons:
                if not self._evict_weakest():
                    break  # everything left is in use right now
        neurons = tuple(range(self._next_neuron, self._next_neuron + self.ensemble_size))
        self._next_neuron += self.ensemble_size
        ensemble = Ensemble(self._next_ensemble_id, neurons, phrase, self.clock)
        self._next_ensemble_id += 1
        self._ensembles[ensemble.identifier] = ensemble
        self._ensemble_features[ensemble.identifier] = set(features)
        for feature in features:
            if feature.startswith("pre:"):  # candidates match by word beginnings
                self._feature_ensembles.setdefault(feature, set()).add(
                    ensemble.identifier
                )
        for neuron in neurons:
            self._imprints[neuron] = {feature: 1.0 for feature in features}
            self._neuron_owner[neuron] = ensemble.identifier
        return ensemble

    def _evict_weakest(self) -> bool:
        """Capacity competition: the weakest idle ensemble gives up its neurons."""
        candidates = [
            ensemble
            for ensemble in self._ensembles.values()
            if ensemble.identifier not in self._protected
        ]
        if not candidates:
            return False

        # importance = strength of what it knows (its connections), kept
        # running; perception is excluded so short names are not weaker
        victim = min(
            candidates,
            key=lambda e: (self._strength.get(e.identifier, 0.0), -e.born),
        )
        victim_neurons = set(victim.neurons)
        del self._ensembles[victim.identifier]
        self._strength.pop(victim.identifier, None)
        features = self._ensemble_features.pop(victim.identifier, set())
        for feature in features:
            holders = self._feature_ensembles.get(feature)
            if holders is not None:
                holders.discard(victim.identifier)
                if not holders:
                    self._feature_ensembles.pop(feature, None)
        for neuron in victim_neurons:
            self._imprints.pop(neuron, None)
            self._neuron_owner.pop(neuron, None)
        dropped: set[tuple[int, int, int]] = set()
        for neuron in victim_neurons:
            for post, gate in self._outgoing.get(neuron, ()):
                dropped.add((neuron, post, gate))
            for pre, gate in self._incoming.get(neuron, ()):
                dropped.add((pre, neuron, gate))
        self._drop_synapses(dropped)
        return True

    def _drop_synapses(self, keys) -> None:
        for key in keys:
            weight = self._synapses.pop(key, 0.0)
            self._peaks.pop(key, None)
            pre, post, gate = key
            out = self._outgoing.get(pre)
            if out is not None:
                out.discard((post, gate))
                if not out:
                    self._outgoing.pop(pre, None)
            incoming = self._incoming.get(post)
            if incoming is not None:
                incoming.discard((pre, gate))
                if not incoming:
                    self._incoming.pop(post, None)
            if weight:
                touching = {
                    self._neuron_owner.get(pre), self._neuron_owner.get(post)
                } - {None}
                for identifier in touching:
                    self._strength[identifier] = (
                        self._strength.get(identifier, 0.0) - weight
                    )
                if gate in self._strength:
                    self._strength[gate] -= weight

    def _apply(self, key: tuple[int, int, int], weight: float, delta: float) -> None:
        pre, post, gate = key
        self._synapses[key] = weight
        self._outgoing.setdefault(pre, set()).add((post, gate))
        self._incoming.setdefault(post, set()).add((pre, gate))
        if delta:
            touching = {
                self._neuron_owner.get(pre), self._neuron_owner.get(post)
            } - {None}
            for identifier in touching:
                self._strength[identifier] = (
                    self._strength.get(identifier, 0.0) + delta
                )
            if gate in self._strength:
                self._strength[gate] += delta

    def _reindex(self) -> None:
        """Rebuild the lookup indexes from the synapses and ensembles."""
        self._outgoing = {}
        self._incoming = {}
        for pre, post, gate in self._synapses:
            self._outgoing.setdefault(pre, set()).add((post, gate))
            self._incoming.setdefault(post, set()).add((pre, gate))
        self._feature_ensembles = {}
        for identifier, features in self._ensemble_features.items():
            if identifier not in self._ensembles:
                continue
            for feature in features:
                if feature.startswith("pre:"):
                    self._feature_ensembles.setdefault(feature, set()).add(identifier)
        self._neuron_owner = {
            neuron: ensemble.identifier
            for ensemble in self._ensembles.values()
            for neuron in ensemble.neurons
        }
        self._strength = {}
        for (pre, post, gate), weight in self._synapses.items():
            touching = {
                self._neuron_owner.get(pre), self._neuron_owner.get(post)
            } - {None}
            for identifier in touching:
                self._strength[identifier] = (
                    self._strength.get(identifier, 0.0) + weight
                )
            if gate in self._ensembles:
                self._strength[gate] = self._strength.get(gate, 0.0) + weight

    def _strengthen(self, pre: int, post: int, gate: int) -> None:
        key = (pre, post, gate)
        weight = self._synapses.get(key, 0.0)
        peak = self._peaks.get(key, 0.0)
        rate = min(0.95, self.learning_rate * (1.0 + SAVINGS * peak))
        new_weight = weight + rate * (1.0 - weight)
        self._peaks[key] = max(peak, new_weight)
        self._apply(key, new_weight, new_weight - weight)

    def _strengthen_imprints(self, phrase: str, ensemble: Ensemble) -> None:
        """Repetition re-strengthens the perception of this memory itself."""
        features = phrase_features(phrase)
        for neuron in ensemble.neurons:
            weights = self._imprints.get(neuron)
            if not weights:
                continue
            for feature in features:
                weight = weights.get(feature)
                if weight is not None:
                    weights[feature] = weight + (
                        self.learning_rate * IMPRINT_BOOST * (1.0 - weight)
                    )

    def _path_strength(
        self, subject_ens: Ensemble, object_ens: Ensemble, relation_ens: Ensemble
    ) -> float:
        gate = relation_ens.identifier
        weights = [
            self._synapses.get((a, b, gate), 0.0)
            for a in subject_ens.neurons
            for b in object_ens.neurons
        ]
        return sum(weights) / len(weights)

    def _spread(self, activation: dict[int, float], gate: int) -> dict[int, float]:
        firing = dict(activation)
        for pre, value in activation.items():
            for post, synapse_gate in self._outgoing.get(pre, ()):
                if synapse_gate != gate:
                    continue
                weight = self._synapses.get((pre, post, synapse_gate), 0.0)
                if weight < self.prune_below:
                    continue
                firing[post] = firing.get(post, 0.0) + weight * value
        return firing

    def _winners(
        self, firing: dict[int, float], excluded: set[int]
    ) -> list[Ensemble]:
        candidates: list[tuple[float, int, Ensemble]] = []
        for ensemble in self._ensembles.values():
            if ensemble.identifier in excluded:
                continue
            activation = max((firing.get(n, 0.0) for n in ensemble.neurons), default=0.0)
            if activation >= ANSWER_AT:
                candidates.append((-activation, ensemble.born, ensemble))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [ensemble for _, _, ensemble in candidates]

    def _consolidate_use(
        self, subject_ens: Ensemble, relation_ens: Ensemble, answered_ids: set[int]
    ) -> None:
        """Use-dependent plasticity: whatever just fired together wires together."""
        gate = relation_ens.identifier
        for identifier in answered_ids:
            answer_ens = self._ensembles[identifier]
            for pre in subject_ens.neurons:
                for post in answer_ens.neurons:
                    key = (pre, post, gate)
                    weight = self._synapses.get(key, 0.0)
                    new_weight = weight + (
                        USE_BOOST * self.learning_rate * (1.0 - weight)
                    )
                    self._peaks[key] = max(
                        self._peaks.get(key, 0.0), new_weight
                    )
                    self._apply(key, new_weight, new_weight - weight)
