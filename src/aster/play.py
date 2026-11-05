"""Run a trained Workshop model in a closed feedback loop."""

import argparse
from pathlib import Path

from aster.cognition.controller import choose_action
from aster.cognition.world_model import load_world_model
from aster.environment import GateKind, Workshop, make_workshop_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Aster checkpoint in Workshop.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate", choices=[kind.value for kind in GateKind], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    model = load_world_model(arguments.checkpoint, device=arguments.device)
    gate_kind = GateKind(arguments.gate)
    environment = Workshop(make_workshop_layout(gate_kind, seed=arguments.seed))
    print(f"Task: cross the {gate_kind.value} and reach {environment.layout.target}.")

    while not environment.done:
        choice = choose_action(model, environment.observe())
        if choice is None:
            print("Stopped: no action passed the current progress checks.")
            break

        result = environment.step(choice.action)
        print(
            f"{environment.steps:02d}. {choice.action.value}: {result.event}; "
            f"agent at {result.observation.agent}"
        )
        if result.invalid:
            print("Stopped after an invalid action; the task state was not advanced.")
            break

    succeeded = environment.agent == environment.layout.target
    if succeeded:
        print(f"Verified success in {environment.steps} actions.")
    else:
        print(f"Not completed. Final location: {environment.agent}.")


if __name__ == "__main__":
    main()
