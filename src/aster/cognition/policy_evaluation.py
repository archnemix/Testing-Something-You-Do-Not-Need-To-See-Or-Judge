"""Closed-loop evaluation for learned and random legal-action policies."""

import random
from collections.abc import Callable

from aster.cognition.policy import ActionPolicy
from aster.environment import Action, GateKind, Workshop, available_actions, make_workshop_layout

LayoutCase = tuple[int, str]
PolicyChooser = Callable[[Workshop, random.Random], Action]


def evaluate_action_policy(
    policy: ActionPolicy,
    seeds: tuple[int, ...],
) -> dict[str, float]:
    """Convenience wrapper evaluating both gate families for each seed."""
    cases = tuple((seed, gate.value) for seed in seeds for gate in GateKind)
    return evaluate_action_policy_layouts(policy, cases)


def evaluate_action_policy_layouts(
    policy: ActionPolicy,
    cases: tuple[LayoutCase, ...],
) -> dict[str, float]:
    """Evaluate a learned action policy without a goal-distance heuristic."""
    return _evaluate(
        cases,
        lambda environment, _rng: policy.choose(environment.observe()),
    )


def evaluate_random_legal_policy(
    seeds: tuple[int, ...],
    seed: int = 0,
) -> dict[str, float]:
    """Convenience wrapper for a random legal-action baseline."""
    cases = tuple((item, gate.value) for item in seeds for gate in GateKind)
    return evaluate_random_legal_policy_layouts(cases, seed)


def evaluate_random_legal_policy_layouts(
    cases: tuple[LayoutCase, ...],
    seed: int = 0,
) -> dict[str, float]:
    """Measure random legal choices on exactly the supplied layouts."""
    randomizer = random.Random(seed)
    return _evaluate(
        cases,
        lambda environment, rng: rng.choice(available_actions(environment.observe())),
        randomizer,
    )


def _evaluate(
    cases: tuple[LayoutCase, ...],
    choose: PolicyChooser,
    randomizer: random.Random | None = None,
) -> dict[str, float]:
    if not cases:
        raise ValueError("at least one evaluation layout is required")
    rng = randomizer or random.Random(0)
    task_count = 0
    successes = 0
    invalid_actions = 0
    successful_steps = 0
    by_gate = {
        gate.value: {"tasks": 0, "successes": 0}
        for gate in GateKind
    }

    for seed, gate_name in cases:
        gate = GateKind(gate_name)
        environment = Workshop(make_workshop_layout(gate, seed=seed))
        while not environment.done:
            action = choose(environment, rng)
            result = environment.step(action)
            invalid_actions += int(result.invalid)
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
    }
    for gate_name, gate_metrics in by_gate.items():
        metrics[f"{gate_name}_success_rate"] = (
            gate_metrics["successes"] / gate_metrics["tasks"]
            if gate_metrics["tasks"]
            else 0.0
        )
    return metrics
