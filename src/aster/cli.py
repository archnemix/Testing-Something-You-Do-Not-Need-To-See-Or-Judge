"""Interactive command-line interface."""

import os
from pathlib import Path

from aster.agent import Aster


def main() -> None:
    memory_path = Path(os.environ.get("ASTER_MEMORY_PATH", Path.home() / ".aster" / "memory.json"))
    agent = Aster(memory_path)
    print("Aster — a small symbolic cognition experiment. Type 'help' for examples.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not text:
            continue
        response = agent.respond(text)
        print(response)
        if text.casefold() in {"quit", "exit", "q"}:
            return
