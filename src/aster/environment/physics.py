"""A tiny deterministic physics world to learn from by watching.

Gravity, walls, and support here are ground truth. Nothing about them is
written into any network: a learner runs action trials and binds each
action to what it perceives happening. Causal knowledge ("letting go
causes falling") forms from repetition, and a name like "gravity" can be
bound later to the concept experience already built -- discovery first,
naming second, the way it works for people.
"""

from __future__ import annotations

from aster.cognition.plastic import PlasticNetwork

GROUND = 0


class PhysicsPlayground:
    """One object in a yard: hold it, drop it, push it against walls."""

    def __init__(self, yard_width: int = 3, height: int = 3, start_x: int | None = None):
        if yard_width < 2 or height < 1:
            raise ValueError("yard_width must be at least 2 and height at least 1")
        self.yard_width = yard_width
        self.height = height
        self.start_x = start_x if start_x is not None else yard_width // 2
        self.x = self.start_x
        self.y = height
        self.held = True

    def _reset(self, resting: bool) -> None:
        self.x = self.start_x
        self.held = not resting
        self.y = GROUND if resting else self.height

    def _release(self) -> list[str]:
        if self.held:
            self.held = False
        if self.y > GROUND:
            self.y = GROUND
            return ["falling", "landing"]
        return []

    def _push(self, direction: int) -> list[str]:
        target = min(max(self.x + direction, 0), self.yard_width - 1)
        if target == self.x:
            return ["staying still"]
        self.x = target
        return ["moving right"] if direction > 0 else ["moving left"]

    def trial(self, action: str) -> tuple[str, ...]:
        """Run one action trial; return the perceived outcome phrases.

        The action label is what the learner believes it did; the outcomes
        are what the world actually did. Their pairing is the causal event.
        """
        if action == "letting go":
            self._reset(resting=False)
            outcomes = self._release()
        elif action in ("pushing right", "pushing left"):
            self._reset(resting=True)
            direction = 1 if action == "pushing right" else -1
            outcomes = self._push(direction)
        elif action == "lifting":
            self._reset(resting=True)
            self.held = True
            self.y = self.height
            outcomes = ["staying in the air"]
        else:
            raise ValueError(f"unknown action: {action}")
        return tuple(outcomes)


def learn_from_trials(
    network: PlasticNetwork,
    playground: PhysicsPlayground,
    action: str,
    repetitions: int = 1,
) -> tuple[str, ...]:
    """Watch action-outcome pairings until causal pathways form."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    outcomes: tuple[str, ...] = ()
    for _ in range(repetitions):
        outcomes = playground.trial(action)
        for outcome in outcomes:
            network.observe(action, "causes", outcome)
    return outcomes
