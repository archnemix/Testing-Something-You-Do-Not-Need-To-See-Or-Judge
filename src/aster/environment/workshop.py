"""Deterministic action and outcome rules for the Workshop benchmark."""

from aster.environment.types import (
    Action,
    GateKind,
    Observation,
    Position,
    StepResult,
    WorkshopLayout,
)

_MOVE_EVENTS = {
    Action.MOVE_NORTH: "moved_north",
    Action.MOVE_SOUTH: "moved_south",
    Action.MOVE_EAST: "moved_east",
    Action.MOVE_WEST: "moved_west",
}


class Workshop:
    def __init__(self, layout: WorkshopLayout) -> None:
        self._validate_layout(layout)
        self.layout = layout
        self.agent = layout.agent_start
        self.key = layout.key
        self.has_key = False
        self.gate_open = False
        self.steps = 0
        self.done = self.agent == layout.target

    @classmethod
    def from_observation(cls, layout: WorkshopLayout, observation: Observation) -> "Workshop":
        """Restore a recorded state for deterministic counterfactual transitions."""
        if (
            observation.width != layout.width
            or observation.height != layout.height
            or observation.walls != layout.walls
            or observation.gate != layout.gate
            or observation.gate_kind is not layout.gate_kind
            or observation.target != layout.target
            or observation.switch != layout.switch
            or observation.max_steps != layout.max_steps
        ):
            raise ValueError("observation does not belong to this Workshop layout")
        if observation.has_key and observation.key is not None:
            raise ValueError("a carried key cannot also remain on the map")
        environment = cls(layout)
        environment.agent = observation.agent
        environment.key = observation.key
        environment.has_key = observation.has_key
        environment.gate_open = observation.gate_open
        environment.steps = observation.steps
        environment.done = (
            observation.agent == layout.target
            or observation.steps >= layout.max_steps
        )
        return environment

    def observe(self) -> Observation:
        return Observation(
            width=self.layout.width,
            height=self.layout.height,
            walls=self.layout.walls,
            agent=self.agent,
            target=self.layout.target,
            gate_kind=self.layout.gate_kind,
            gate=self.layout.gate,
            gate_open=self.gate_open,
            key=self.key,
            has_key=self.has_key,
            switch=self.layout.switch,
            steps=self.steps,
            max_steps=self.layout.max_steps,
        )

    def step(self, action: Action) -> StepResult:
        previous_observation = self.observe()
        if self.done:
            return StepResult(
                previous_observation,
                previous_observation,
                action,
                "already_complete",
                0.0,
                True,
                True,
            )

        self.steps += 1
        event, invalid = self._apply(action)
        succeeded = self.agent == self.layout.target
        timed_out = self.steps >= self.layout.max_steps and not succeeded
        self.done = succeeded or timed_out
        if timed_out:
            event = "time_limit_reached"
        reward = 1.0 if succeeded else -1.0 if timed_out else -0.01
        if invalid and not timed_out:
            reward = -0.05

        return StepResult(
            previous_observation,
            self.observe(),
            action,
            event,
            reward,
            self.done,
            invalid,
        )

    def _apply(self, action: Action) -> tuple[str, bool]:
        movement = action.movement
        if movement is not None:
            destination = Position(self.agent.x + movement[0], self.agent.y + movement[1])
            if not self._can_enter(destination):
                return "blocked", True
            self.agent = destination
            return _MOVE_EVENTS[action], False

        if action is Action.PICK_UP:
            if self.key != self.agent:
                return "no_key_here", True
            self.key = None
            self.has_key = True
            return "picked_up_key", False

        if action is Action.INTERACT:
            return self._interact()

        if action is Action.WAIT:
            return "waited", False

        raise ValueError(f"Unsupported action: {action!r}")

    def _interact(self) -> tuple[str, bool]:
        if self.layout.gate_kind is GateKind.DOOR and self._adjacent(self.agent, self.layout.gate):
            if not self.has_key:
                return "door_locked", True
            self.gate_open = True
            return "door_opened", False

        if self.layout.gate_kind is GateKind.BRIDGE and self.agent == self.layout.switch:
            self.gate_open = not self.gate_open
            return ("bridge_opened" if self.gate_open else "bridge_closed"), False

        return "nothing_to_interact_with", True

    def _can_enter(self, position: Position) -> bool:
        if not (0 <= position.x < self.layout.width and 0 <= position.y < self.layout.height):
            return False
        if position in self.layout.walls:
            return False
        if position == self.layout.gate and not self.gate_open:
            return False
        return True

    @staticmethod
    def _adjacent(left: Position, right: Position) -> bool:
        return abs(left.x - right.x) + abs(left.y - right.y) == 1

    @staticmethod
    def _validate_layout(layout: WorkshopLayout) -> None:
        if layout.width < 1 or layout.height < 1:
            raise ValueError("Workshop dimensions must be positive")
        if layout.max_steps < 1:
            raise ValueError("Workshop max_steps must be positive")
        if layout.gate_kind is GateKind.DOOR:
            if layout.key is None or layout.switch is not None:
                raise ValueError("Door layouts need a key and must not define a switch")
        elif layout.switch is None or layout.key is not None:
            raise ValueError("Bridge layouts need a switch and must not define a key")

        positions = [layout.agent_start, layout.target, layout.gate]
        positions.extend(layout.walls)
        if layout.key is not None:
            positions.append(layout.key)
        if layout.switch is not None:
            positions.append(layout.switch)
        if any(not (0 <= p.x < layout.width and 0 <= p.y < layout.height) for p in positions):
            raise ValueError("Every layout position must be inside the map")
        if len(positions) != len(set(positions)):
            raise ValueError("Layout positions and walls must not overlap")
