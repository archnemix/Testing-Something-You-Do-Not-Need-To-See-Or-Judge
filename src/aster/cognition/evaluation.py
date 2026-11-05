"""Closed-loop evaluation against held-out Workshop layouts."""

from aster.cognition.controller import choose_action
from aster.cognition.world_model import TransitionModel
from aster.environment import Action, GateKind, Workshop, make_workshop_layout

LayoutCase = tuple[int, str]


def evaluate_controller(model: TransitionModel, seeds: tuple[int, ...]) -> dict[str, float]:
    """Convenience wrapper evaluating both gate families for each seed."""
    cases = tuple((seed, gate.value) for seed in seeds for gate in GateKind)
    return evaluate_controller_layouts(model, cases)


def evaluate_controller_layouts(
    model: TransitionModel,
    cases: tuple[LayoutCase, ...],
) -> dict[str, float]:
    """Evaluate exactly the supplied layouts, without adding paired gate cases."""
    if not cases:
        raise ValueError("at least one evaluation layout is required")

    successes = 0
    invalid_actions = 0
    stopped_early = 0
    successful_steps = 0
    by_gate = {
        gate.value: {"tasks": 0, "successes": 0, "invalid_actions": 0}
        for gate in GateKind
    }

    for seed, gate_name in cases:
        gate_kind = GateKind(gate_name)
        environment = Workshop(make_workshop_layout(gate_kind, seed=seed))
        excluded_by_state: dict[tuple[object, ...], set[Action]] = {}

        while not environment.done:
            observation = environment.observe()
            state_key = (
                observation.agent,
                observation.key,
                observation.has_key,
                observation.gate_open,
            )
            excluded = frozenset(excluded_by_state.get(state_key, set()))
            choice = choose_action(model, observation, excluded)
            if choice is None:
                stopped_early += 1
                break

            result = environment.step(choice.action)
            if result.invalid:
                invalid_actions += 1
                by_gate[gate_kind.value]["invalid_actions"] += 1
                excluded_by_state.setdefault(state_key, set()).add(choice.action)

        gate_metrics = by_gate[gate_kind.value]
        gate_metrics["tasks"] += 1
        if environment.agent == environment.layout.target:
            successes += 1
            gate_metrics["successes"] += 1
            successful_steps += environment.steps

    task_count = len(cases)
    result = {
        "tasks": float(task_count),
        "successes": float(successes),
        "success_rate": successes / task_count,
        "invalid_actions": float(invalid_actions),
        "stopped_early": float(stopped_early),
        "mean_steps_on_success": successful_steps / successes if successes else 0.0,
    }
    for gate_name, metrics in by_gate.items():
        result[f"{gate_name}_success_rate"] = (
            metrics["successes"] / metrics["tasks"]
            if metrics["tasks"]
            else 0.0
        )
        result[f"{gate_name}_invalid_actions"] = float(metrics["invalid_actions"])
    return result
