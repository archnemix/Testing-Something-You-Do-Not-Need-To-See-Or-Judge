"""Deterministic collection and leakage-safe splitting of Workshop experience."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from aster.environment import (
    Action,
    GateKind,
    StepResult,
    Workshop,
    WorkshopLayout,
    collect_expert_episode,
    encode_observation,
    make_workshop_layout,
)

ENVIRONMENT_VERSION = 1
DATA_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class TransitionExample:
    layout_id: str
    state: tuple[float, ...]
    action: int
    next_state: tuple[float, ...]
    reward: float
    done: bool
    invalid: bool
    seed: int
    gate_kind: str
    episode_kind: str
    step: int


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: tuple[TransitionExample, ...]
    validation: tuple[TransitionExample, ...]
    test: tuple[TransitionExample, ...]
    train_layouts: tuple[tuple[int, str], ...]
    validation_layouts: tuple[tuple[int, str], ...]
    test_layouts: tuple[tuple[int, str], ...]
    train_layout_ids: tuple[str, ...]
    validation_layout_ids: tuple[str, ...]
    test_layout_ids: tuple[str, ...]


def collect_examples(
    seeds: Iterable[int],
    random_episodes_per_layout: int = 2,
    max_steps: int = 64,
) -> list[TransitionExample]:
    """Collect both rule-aware coverage and random experience for each layout."""
    if random_episodes_per_layout < 0:
        raise ValueError("random_episodes_per_layout cannot be negative")

    examples: list[TransitionExample] = []
    for seed in sorted(set(seeds)):
        for gate_kind in GateKind:
            layout = make_workshop_layout(gate_kind, seed=seed, max_steps=max_steps)
            layout_id = _layout_id(layout)
            expert_results = collect_expert_episode(layout)
            examples.extend(
                _to_examples(expert_results, seed, gate_kind, layout_id, "expert")
            )
            covered_states = {
                encode_observation(result.previous_observation): result.previous_observation
                for result in expert_results
            }
            for observation in covered_states.values():
                for action in Action:
                    counterfactual = Workshop.from_observation(layout, observation).step(action)
                    examples.extend(
                        _to_examples(
                            [counterfactual],
                            seed,
                            gate_kind,
                            layout_id,
                            "coverage",
                        )
                    )

            for episode_number in range(random_episodes_per_layout):
                episode_seed = seed * 10_007 + episode_number
                randomizer = random.Random(episode_seed)
                environment = Workshop(layout)
                results: list[StepResult] = []
                for _ in range(max_steps):
                    result = environment.step(randomizer.choice(tuple(Action)))
                    results.append(result)
                    if result.done:
                        break
                examples.extend(
                    _to_examples(
                        results,
                        seed,
                        gate_kind,
                        layout_id,
                        f"random-{episode_number}",
                    )
                )
    return examples


def split_by_layout(
    examples: Iterable[TransitionExample],
    seed: int = 0,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> DatasetSplits:
    """Split canonical layouts, keeping duplicate-seed layouts in one partition."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave room for test data")

    rows = list(examples)
    layout_ids = sorted({row.layout_id for row in rows})
    if len(layout_ids) < 3:
        raise ValueError("at least three distinct layouts are required for three data splits")

    shuffled = layout_ids.copy()
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * train_fraction)
    train_end = min(max(train_end, 1), len(shuffled) - 2)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_end = min(train_end + validation_count, len(shuffled) - 1)
    train_ids = tuple(sorted(shuffled[:train_end]))
    validation_ids = tuple(sorted(shuffled[train_end:validation_end]))
    test_ids = tuple(sorted(shuffled[validation_end:]))

    return DatasetSplits(
        train=_rows_for_layouts(rows, train_ids),
        validation=_rows_for_layouts(rows, validation_ids),
        test=_rows_for_layouts(rows, test_ids),
        train_layouts=_cases_for_layouts(rows, train_ids),
        validation_layouts=_cases_for_layouts(rows, validation_ids),
        test_layouts=_cases_for_layouts(rows, test_ids),
        train_layout_ids=train_ids,
        validation_layout_ids=validation_ids,
        test_layout_ids=test_ids,
    )

def write_splits(splits: DatasetSplits, output_dir: Path) -> None:
    """Write compact JSONL splits and a manifest for a reproducible training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        _write_jsonl(output_dir / f"{name}.jsonl", getattr(splits, name))

    split_hashes = {
        name: hashlib.sha256((output_dir / f"{name}.jsonl").read_bytes()).hexdigest()
        for name in ("train", "validation", "test")
    }
    manifest = {
        "environment_version": ENVIRONMENT_VERSION,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "split_sha256": split_hashes,
        "feature_size": len(splits.train[0].state) if splits.train else 0,
        "action_names": [action.value for action in Action],
        "train_layout_ids": list(splits.train_layout_ids),
        "validation_layout_ids": list(splits.validation_layout_ids),
        "test_layout_ids": list(splits.test_layout_ids),
        "train_layouts": [list(case) for case in splits.train_layouts],
        "validation_layouts": [list(case) for case in splits.validation_layouts],
        "test_layouts": [list(case) for case in splits.test_layouts],
        "counts": {
            name: len(getattr(splits, name))
            for name in ("train", "validation", "test")
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def _layout_id(layout: WorkshopLayout) -> str:
    payload = {
        "width": layout.width,
        "height": layout.height,
        "walls": sorted((position.x, position.y) for position in layout.walls),
        "agent_start": (layout.agent_start.x, layout.agent_start.y),
        "target": (layout.target.x, layout.target.y),
        "gate_kind": layout.gate_kind.value,
        "gate": (layout.gate.x, layout.gate.y),
        "key": (layout.key.x, layout.key.y) if layout.key else None,
        "switch": (layout.switch.x, layout.switch.y) if layout.switch else None,
        "max_steps": layout.max_steps,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _to_examples(
    results: Iterable[StepResult],
    seed: int,
    gate_kind: GateKind,
    layout_id: str,
    episode_kind: str,
) -> list[TransitionExample]:
    return [
        TransitionExample(
            layout_id=layout_id,
            state=encode_observation(result.previous_observation),
            action=tuple(Action).index(result.action),
            next_state=encode_observation(result.observation),
            reward=result.reward,
            done=result.done,
            invalid=result.invalid,
            seed=seed,
            gate_kind=gate_kind.value,
            episode_kind=episode_kind,
            step=result.previous_observation.steps,
        )
        for result in results
    ]


def _rows_for_layouts(
    rows: list[TransitionExample],
    layout_ids: tuple[str, ...],
) -> tuple[TransitionExample, ...]:
    included = set(layout_ids)
    return tuple(row for row in rows if row.layout_id in included)


def _cases_for_layouts(
    rows: list[TransitionExample],
    layout_ids: tuple[str, ...],
) -> tuple[tuple[int, str], ...]:
    included = set(layout_ids)
    representative: dict[str, tuple[int, str]] = {}
    for row in rows:
        if row.layout_id not in included:
            continue
        case = (row.seed, row.gate_kind)
        previous = representative.get(row.layout_id)
        if previous is None or case < previous:
            representative[row.layout_id] = case
    return tuple(sorted(representative.values()))


def _write_jsonl(path: Path, rows: tuple[TransitionExample, ...]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(asdict(row), separators=(",", ":")) + "\n")
