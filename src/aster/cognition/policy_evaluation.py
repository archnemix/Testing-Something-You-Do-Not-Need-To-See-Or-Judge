"""Closed-loop evaluation for learned, reactive, random, and oracle policies."""

import random
from collections.abc import Callable

from aster.cognition.policy import ActionPolicy
from aster.environment import (
    Action,
    GateKind,
    Workshop,
    available_actions,
    choose_reactive_action,
    collect_expert_episode,
    make_workshop_layout,
)

LayoutCase = tuple[int, str]
PolicyChooser = Callable[[Workshop, random.Random], Action]


def evaluate_action_policy(
    policy: ActionPolicy,
    seeds: tuple[int, ...],
    max_steps: int = 64,
) -> dict[str, float]:
    """Convenience wrapper evaluating both gate families for each seed."""
    cases = tuple((seed, gate.value) for seed in seeds for gate in GateKind)
    return evaluate_action_policy_layouts(policy, cases, max_steps=max_steps)


def evaluate_action_policy_layouts(
    policy: ActionPolicy,
    cases: tuple[LayoutCase, ...],
    max_steps: int = 64,
) -> dict[str, float]:
    """Evaluate a learned action policy without a goal-distance heuristic."""
    return _evaluate(
        cases,
        lambda environment, _rng: policy.choose(environment.observe()),
        max_steps=max_steps,
    )


def evaluate_random_legal_policy(
    seeds: tuple[int, ...],
    seed: int = 0,
    max_steps: int = 64,
) -> dict[str, float]:
    """Convenience wrapper for a random legal-action baseline."""
    cases = tuple((item, gate.value) for item in seeds for gate in GateKind)
    return evaluate_random_legal_policy_layouts(cases, seed, max_steps=max_steps)


def evaluate_random_legal_policy_layouts(
    cases: tuple[LayoutCase, ...],
    seed: int = 0,
    max_steps: int = 64,
) -> dict[str, float]:
    """Measure random legal choices on exactly the supplied layouts."""
    randomizer = random.Random(seed)
    return _evaluate(
        cases,
        lambda environment, rng: rng.choice(available_actions(environment.observe())),
        randomizer,
        max_steps=max_steps,
    )


def evaluate_reactive_policy_layouts(
    cases: tuple[LayoutCase, ...],
    max_steps: int = 64,
) -> dict[str, float]:
    """Evaluate a goal-seeking baseline that ignores state-changing objects."""
    return _evaluate(
        cases,
        lambda environment, _rng: choose_reactive_action(environment.observe()),
        max_steps=max_steps,
    )


def evaluate_oracle_policy_layouts(
    cases: tuple[LayoutCase, ...],
    max_steps: int = 64,
) -> dict[str, float]:
    """Replay the privileged rule-aware solver to estimate task solvability.

    This is not a deployable agent; it bounds achievable success and
    measures the gap between learned behavior and the oracle.
    """
    if not cases:
        raise ValueError("at least one evaluation layout is required")
    successes = 0
    invalid_actions = 0
    successful_steps = 0
    by_gate = {
        gate.value: {"tasks": 0, "successes": 0}
        for gate in GateKind
    }
    for seed, gate_name in cases:
        gate = GateKind(gate_name)
        layout = make_workshop_layout(gate, seed=seed, max_steps=max_steps)
        results = collect_expert_episode(layout)
        invalid_actions += sum(int(result.invalid) for result in results)
        gate_metrics = by_gate[gate.value]
        gate_metrics["tasks"] += 1
        if results[-1].observation.agent == layout.target:
            successes += 1
            successful_steps += len(results)
            gate_metrics["successes"] += 1

    task_count = len(cases)
    metrics = {
        "tasks": float(task_count),
        "successes": float(successes),
        "success_rate": successes / task_count,
        "invalid_actions": float(invalid_actions),
        "mean_steps_on_success": successful_steps / successes if successes else 0.0,
    }
    for gate_name, gate_metrics in by_gate.items():
        metrics[f"{gate_name}_success_rate"] = (
            gate_metrics["successes"] / gate_metrics["tasks"]
            if gate_metrics["tasks"]
            else 0.0
        )
    return metrics


def _evaluate(
    cases: tuple[LayoutCase, ...],
    choose: PolicyChooser,
    randomizer: random.Random | None = None,
    max_steps: int = 64,
) -> dict[str, float]:
    if not cases:
        raise ValueError("at least one evaluation layout is required")
    rng = randomizer or random.Random(0)
    task_count = 0
    successes = 0
    invalid_actions = 0
    successful_steps = 0
    total_steps = 0
    by_gate = {
        gate.value: {"tasks": 0, "successes": 0}
        for gate in GateKind
    }

    for seed, gate_name in cases:
        gate = GateKind(gate_name)
        environment = Workshop(make_workshop_layout(gate, seed=seed, max_steps=max_steps))
        while not environment.done:
            action = choose(environment, rng)
            result = environment.step(action)
            invalid_actions += int(result.invalid)
            total_steps += 1
        task_count += 1
        gate_metrics = by_gate[gate.value]
        gate_metrics["tasks"] += 1
        if environment.agent == environment.layout.target:
            successes += 1
            successful_steps += environment.steps
            gate_metrics["successes"] += 1

    metrics = {
        "tasks": float(task_count),
        "successes": float(successes),
        "success_rate": successes / task_count,
        "invalid_actions": float(invalid_actions),
        "mean_steps_on_success": successful_steps / successes if successes else 0.0,
        "total_steps": float(total_steps),
    }
    for gate_name, gate_metrics in by_gate.items():
        metrics[f"{gate_name}_success_rate"] = (
            gate_metrics["successes"] / gate_metrics["tasks"]
            if gate_metrics["tasks"]
            else 0.0
        )
    return metrics
