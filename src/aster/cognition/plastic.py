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
FALLOFF = 0.8  # activation multiplier per spreading hop
USE_BOOST = 0.5  # retrieval re-strengthening factor (use-dependent plasticity)
IMPRINT_BOOST = 0.2


@dataclass(frozen=True, slots=True)
class Ensemble:
    """A small pool of neurons that fires together for one concept."""

    identifier: int
    neurons: tuple[int, ...]
    label: str
    born: int


def phrase_features(phrase: str) -> set[str]:
    """Perception: a phrase lights up spelling-feature neurons (case folded)."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", phrase.casefold()).strip()
    words = [word for word in cleaned.split() if word]
    features: set[str] = set()
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
    ) -> None:
        if learning_rate <= 0.0 or not 0.0 < decay_rate < 1.0:
            raise ValueError("learning_rate must be positive and decay_rate in (0, 1)")
        if spread_hops < 1 or ensemble_size < 1:
            raise ValueError("spread_hops and ensemble_size must be at least 1")
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.prune_below = prune_below
        self.spread_hops = spread_hops
        self.ensemble_size = ensemble_size
        self.clock = 0
        self._next_neuron = 0
        self._ensembles: dict[int, Ensemble] = {}
        self._ensemble_features: dict[int, set[str]] = {}
        self._imprints: dict[tuple[str, int], float] = {}
        self._synapses: dict[tuple[int, int, int], float] = {}

    # -- telling -----------------------------------------------------------

    def observe(self, subject: str, relation: str, object_phrase: str) -> bool:
        """Tell the network a fact; strengthen pathways.

        Returns True when the fact is new (the network grew, or the pathway
        was weak), False when the same fact is merely being repeated.
        """
        self.clock += 1
        subject_ens = self._recognize(subject, recruit=True)
        relation_ens = self._recognize(relation, recruit=True)
        object_ens = self._recognize(object_phrase, recruit=True)
        assert subject_ens and relation_ens and object_ens
        strength = self._path_strength(subject_ens, object_ens, relation_ens)
        for pre in subject_ens.neurons:
            for post in object_ens.neurons:
                self._strengthen(pre, post, relation_ens.identifier)
        self._strengthen_imprints(subject)
        self._strengthen_imprints(relation)
        self._strengthen_imprints(object_phrase)
        return strength < KNOWN_BELOW

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
        """
        if times < 1:
            raise ValueError("times must be at least 1")
        self.clock += times
        factor = (1.0 - self.decay_rate) ** times
        self._synapses = {
            key: weight * factor
            for key, weight in self._synapses.items()
            if weight * factor >= self.prune_below
        }
        self._imprints = {
            key: weight * factor
            for key, weight in self._imprints.items()
            if weight * factor >= self.prune_below
        }
        self._prune_dead_neurons()

    def _prune_dead_neurons(self) -> None:
        connected: set[int] = set()
        for pre, post, _ in self._synapses:
            connected.add(pre)
            connected.add(post)
        for _, neuron in self._imprints:
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
                for (f, n), w in sorted(self._imprints.items())
            ],
            "synapses": [
                {"pre": a, "post": b, "gate": g, "weight": round(w, 6)}
                for (a, b, g), w in sorted(self._synapses.items())
            ],
        }

    def restore(self, payload: dict[str, object]) -> None:
        self.clock = int(payload.get("clock", 0))
        self._next_neuron = int(payload.get("next_neuron", 0))
        self._ensembles = {
            int(item["identifier"]): Ensemble(
                int(item["identifier"]),
                tuple(int(n) for n in item["neurons"]),
                str(item["label"]),
                int(item["born"]),
            )
            for item in payload.get("ensembles", [])  # type: ignore[union-attr]
        }
        self._imprints = {
            (str(item["feature"]), int(item["neuron"])): float(item["weight"])
            for item in payload.get("imprints", [])  # type: ignore[union-attr]
        }
        self._synapses = {
            (int(item["pre"]), int(item["post"]), int(item["gate"])): float(
                item["weight"]
            )
            for item in payload.get("synapses", [])  # type: ignore[union-attr]
        }
        self._ensemble_features = {}
        for feature, neuron in self._imprints:
            for identifier, ensemble in self._ensembles.items():
                if neuron in ensemble.neurons:
                    self._ensemble_features.setdefault(identifier, set()).add(feature)

    # -- internals ---------------------------------------------------------

    def _recognize(self, phrase: str, recruit: bool) -> Ensemble | None:
        features = phrase_features(phrase)
        if not features:
            return None
        best: Ensemble | None = None
        best_activation = 0.0
        for ensemble in self._ensembles.values():
            stored = self._ensemble_features.get(ensemble.identifier, set())
            shared = features & stored
            if not shared:
                continue
            weight = sum(
                self._imprints.get((feature, ensemble.neurons[0]), 0.0)
                for feature in shared
            )
            activation = weight / len(features | stored)
            if activation > best_activation:
                best_activation = activation
                best = ensemble
        if best is not None and best_activation >= (RECRUIT_AT if recruit else CUE_AT):
            return best
        if not recruit:
            return None
        return self._recruit(phrase, features)

    def _recruit(self, phrase: str, features: set[str]) -> Ensemble:
        neurons = tuple(range(self._next_neuron, self._next_neuron + self.ensemble_size))
        self._next_neuron += self.ensemble_size
        ensemble = Ensemble(len(self._ensembles), neurons, phrase, self.clock)
        self._ensembles[ensemble.identifier] = ensemble
        self._ensemble_features[ensemble.identifier] = set(features)
        for feature in features:
            for neuron in neurons:
                self._imprints[(feature, neuron)] = 1.0
        return ensemble

    def _strengthen(self, pre: int, post: int, gate: int) -> None:
        key = (pre, post, gate)
        weight = self._synapses.get(key, 0.0)
        self._synapses[key] = weight + self.learning_rate * (1.0 - weight)

    def _strengthen_imprints(self, phrase: str) -> None:
        features = phrase_features(phrase)
        for feature in features:
            for ensemble in self._ensembles.values():
                for neuron in ensemble.neurons:
                    key = (feature, neuron)
                    weight = self._imprints.get(key, 0.0)
                    if weight > 0.0:
                        self._imprints[key] = weight + (
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
        for (pre, post, synapse_gate), weight in self._synapses.items():
            if synapse_gate != gate or weight < self.prune_below:
                continue
            if pre in activation:
                firing[post] = firing.get(post, 0.0) + weight * activation[pre]
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
                    self._synapses[key] = weight + (
                        USE_BOOST * self.learning_rate * (1.0 - weight)
                    )
