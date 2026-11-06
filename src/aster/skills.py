"""Skills are patches of the world, learned by talking and by doing.

The world model is not a separate store of facts: it is the same pathway
substrate as everything else, shaped the way training shapes weights --
by experience (acting in a world) and by conversation (being told things).
A skill here is one patch of the world: the yard, the kitchen, people and
their trades, a vocabulary, places, chores. The protocol is the one the
user asked for: teach skill 1 through conversation, test it, then keep
talking and teach skill 2, test whether it was learned -- and so on --
measuring after every skill whether each new skill landed and the old
ones survived. Run ``python -m aster.skills`` for the full study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aster.agent import Aster
from aster.environment.physics import PhysicsPlayground, learn_from_trials


@dataclass(frozen=True)
class Trial:
    """Act in the world; the outcome is perceived, not told."""

    action: str
    repetitions: int = 6


@dataclass(frozen=True)
class Say:
    """A turn of ordinary conversation."""

    text: str


@dataclass(frozen=True)
class Skill:
    name: str
    kind: str  # "experienced" (learned by acting) or "told" (learned by talking)
    lessons: tuple[Trial | Say, ...]
    tests: tuple[tuple[str, str], ...]  # (question, expected wording in the answer)


def build_curriculum() -> tuple[Skill, ...]:
    """Six patches of the world: one discovered, five taught by talking."""
    yard = Skill(
        name="yard",
        kind="experienced",
        lessons=(
            Trial("letting go"),
            Trial("pushing right"),
            Trial("pushing left"),
            Trial("lifting"),
            Say("gravity means falling"),
            Say("letting go means falling"),
            Say("the yard is open"),
            Say("the yard has a floor"),
        ),
        tests=(
            ("what does letting go cause?", "falling"),
            ("what does letting go cause?", "landing"),
            ("what does pushing right cause?", "moving right"),
            ("what does pushing left cause?", "moving left"),
            ("what does lifting cause?", "staying in the air"),
            ("what does gravity mean?", "falling"),
            ("what does letting go mean?", "falling"),
            ("what is the yard?", "open"),
            ("what does the yard have?", "a floor"),
        ),
    )
    kitchen = Skill(
        name="kitchen",
        kind="told",
        lessons=(
            Say("the pot is metal"),
            Say("the pan is metal"),
            Say("the plate is ceramic"),
            Say("the spoon is wood"),
            Say("the knife is steel"),
            Say("the fridge is cold"),
            Say("the oven is hot"),
            Say("the sink is deep"),
            Say("the counter is stone"),
            Say("the cup is glass"),
            Say("the bread is warm"),
            Say("the soup is hot"),
            Say("the salad is fresh"),
            Say("the water is clear"),
            Say("the dough is soft"),
            Say("heating milk makes warm milk"),
            Say("boiling water makes steam"),
            Say("mixing flour makes dough"),
            Say("baking dough makes bread"),
            Say("cutting bread makes slices"),
            Say("chilling juice makes cold juice"),
            Say("burning toast makes smoke"),
            Say("melting butter makes liquid butter"),
            Say("whisking eggs makes foam"),
            Say("roasting beans makes coffee"),
            Say("the pot has a handle"),
            Say("the pan has a lid"),
            Say("the knife has an edge"),
            Say("the kitchen has a stove"),
            Say("the stove has four burners"),
        ),
        tests=(
            ("what is the pot?", "metal"),
            ("what is the knife?", "steel"),
            ("what is the soup?", "hot"),
            ("what is the dough?", "soft"),
            ("what does heating milk make?", "warm milk"),
            ("what does boiling water make?", "steam"),
            ("what does baking dough make?", "bread"),
            ("what does melting butter make?", "liquid butter"),
            ("what does the kitchen have?", "a stove"),
            ("what does the pot have?", "a handle"),
            ("what does the stove have?", "four burners"),
            ("what does the pan have?", "a lid"),
        ),
    )
    trades = Skill(
        name="trades",
        kind="told",
        lessons=(
            Say("ada is a builder"),
            Say("ada is from the hill village"),
            Say("ada has a good hammer"),
            Say("kira is a teacher"),
            Say("kira is from the port town"),
            Say("kira has many books"),
            Say("mira is a pilot"),
            Say("mira is from the capital"),
            Say("mira has a silver map"),
            Say("tom is a farmer"),
            Say("tom is from the river bend"),
            Say("tom has a red tractor"),
            Say("ilya is a doctor"),
            Say("ilya is from the hill village"),
            Say("ilya has a black bag"),
            Say("sara is a sailor"),
            Say("sara is from the port town"),
            Say("sara has a blue boat"),
            Say("noor is a baker"),
            Say("noor is from the capital"),
            Say("noor has a stone oven"),
            Say("dev is a smith"),
            Say("dev is from the river bend"),
            Say("dev has a heavy anvil"),
            Say("ana is a painter"),
            Say("ana is from the port town"),
            Say("ana has bright paints"),
            Say("zoe is a ranger"),
            Say("zoe is from the hill village"),
            Say("zoe has a longbow"),
        ),
        tests=(
            ("who is ada?", "a builder"),
            ("what does ada have?", "a good hammer"),
            ("what is kira?", "a teacher"),
            ("what does kira have?", "many books"),
            ("who is mira?", "a pilot"),
            ("what does mira have?", "a silver map"),
            ("who is tom?", "a farmer"),
            ("what does tom have?", "a red tractor"),
            ("who is noor?", "a baker"),
            ("what does noor have?", "a stone oven"),
            ("who is sara?", "a sailor"),
            ("what does dev have?", "a heavy anvil"),
        ),
    )
    words = Skill(
        name="words",
        kind="told",
        lessons=(
            Say("flur means rain"),
            Say("brakk means stone"),
            Say("vel means bird"),
            Say("sint means river"),
            Say("tol means fire"),
            Say("halm means tree"),
            Say("renn means road"),
            Say("pel means fish"),
            Say("worn means wind"),
            Say("kesh means bread"),
            Say("miral means sea"),
            Say("dorn means hill"),
            Say("lask means light"),
            Say("venn means night"),
            Say("rudd means red"),
            Say("sorn means gold"),
            Say("tivel means winter"),
            Say("ashk means summer"),
            Say("pram means boat"),
            Say("cull means cup"),
            Say("fenn means friend"),
            Say("gorse means wolf"),
            Say("hest means horse"),
            Say("yara means star"),
            Say("quill means feather"),
            Say("namm means name"),
            Say("trug means box"),
            Say("welt means world"),
            Say("sable means black"),
            Say("runn means river"),
        ),
        tests=(
            ("what does flur mean?", "rain"),
            ("what does brakk mean?", "stone"),
            ("what does vel mean?", "bird"),
            ("what does tol mean?", "fire"),
            ("what does miral mean?", "sea"),
            ("what does yara mean?", "star"),
            ("what does tivel mean?", "winter"),
            ("what does pram mean?", "boat"),
            ("what does fenn mean?", "friend"),
            ("what does hest mean?", "horse"),
            ("what does welt mean?", "world"),
            ("what does sable mean?", "black"),
        ),
    )
    places = Skill(
        name="places",
        kind="told",
        lessons=(
            Say("the market is north of the well"),
            Say("the market has three gates"),
            Say("the market is busy"),
            Say("the river is cold"),
            Say("the river has many fish"),
            Say("the bridge is over the river"),
            Say("the bridge is old"),
            Say("the hill is east of the market"),
            Say("the hill has a tower"),
            Say("the tower is tall"),
            Say("the harbor is south of the well"),
            Say("the harbor has many boats"),
            Say("the harbor is quiet"),
            Say("the forest is west of the hill"),
            Say("the forest is dark"),
            Say("the forest has tall trees"),
            Say("the well is deep"),
            Say("the well is old"),
            Say("the road is long"),
            Say("the road is smooth"),
            Say("the bakery is near the market"),
            Say("the bakery has fresh loaves"),
            Say("the mill is by the river"),
            Say("the mill has a wheel"),
            Say("the castle is above the hill"),
            Say("the castle has high walls"),
            Say("the garden is behind the castle"),
            Say("the garden is green"),
            Say("the lake is beyond the forest"),
            Say("the lake is calm"),
        ),
        tests=(
            ("what is the market?", "north of the well"),
            ("what does the market have?", "three gates"),
            ("what is the river?", "cold"),
            ("what does the river have?", "many fish"),
            ("what is the bridge?", "over the river"),
            ("what does the hill have?", "a tower"),
            ("what is the harbor?", "quiet"),
            ("what does the harbor have?", "many boats"),
            ("what is the forest?", "dark"),
            ("what does the castle have?", "high walls"),
            ("what is the lake?", "calm"),
            ("what does the mill have?", "a wheel"),
        ),
    )
    chores = Skill(
        name="chores",
        kind="told",
        lessons=(
            Say("making tea is a chore"),
            Say("the first step is heating water"),
            Say("heating water makes steam"),
            Say("adding leaves makes tea"),
            Say("pouring tea is the last step"),
            Say("sweeping the floor is a chore"),
            Say("the first tool is a broom"),
            Say("the broom is long"),
            Say("mopping is a chore"),
            Say("the mop is wet"),
            Say("washing dishes is a chore"),
            Say("the sink is wide"),
            Say("hot water makes clean dishes"),
            Say("the sponge is soft"),
            Say("the sponge has many holes"),
            Say("feeding the cat is a chore"),
            Say("the cat is hungry"),
            Say("the cat has an empty bowl"),
            Say("filling the bowl makes the cat calm"),
            Say("watering plants is a chore"),
            Say("the watering can is heavy"),
            Say("the watering can has water"),
            Say("watering plants makes new leaves"),
            Say("the garden has many flowerpots"),
            Say("the flowerpot is clay"),
            Say("the shelf is wooden"),
            Say("the shelf has three boards"),
            Say("fixing the shelf is a chore"),
            Say("the screwdriver is small"),
            Say("the toolbox is heavy"),
        ),
        tests=(
            ("what is making tea?", "a chore"),
            ("what is the first step?", "heating water"),
            ("what does heating water make?", "steam"),
            ("what does adding leaves make?", "tea"),
            ("what is pouring tea?", "the last step"),
            ("what is sweeping the floor?", "a chore"),
            ("what is the cat?", "hungry"),
            ("what does filling the bowl make?", "the cat calm"),
            ("what is watering plants?", "a chore"),
            ("what is the flowerpot?", "clay"),
            ("what is the shelf?", "wooden"),
            ("what does the shelf have?", "three boards"),
        ),
    )
    return (yard, kitchen, trades, words, places, chores)


@dataclass
class Stream:
    """One conversation: talking, acting, asking -- time passes each turn."""

    agent: Aster = field(default_factory=Aster)
    playground: PhysicsPlayground = field(default_factory=PhysicsPlayground)
    turns: int = 0

    def teach(self, skill: Skill) -> None:
        self.teach_lessons(skill.lessons)

    def teach_lessons(self, lessons: tuple[Trial | Say, ...] | list[Trial | Say]) -> None:
        for lesson in lessons:
            if isinstance(lesson, Trial):
                learn_from_trials(
                    self.agent.memory.network,
                    self.playground,
                    lesson.action,
                    lesson.repetitions,
                )
            else:
                self.agent.respond(lesson.text)
            self.turns += 1

    def quiz(self, skill: Skill) -> dict[str, object]:
        hits = 0
        for question, expected in skill.tests:
            response = self.agent.respond(question)
            self.turns += 1
            if expected.casefold() in response.casefold():
                hits += 1
        return {
            "accuracy": hits / len(skill.tests),
            "hits": hits,
            "total": len(skill.tests),
        }

    def chat(self, text: str) -> str:
        self.turns += 1
        return self.agent.respond(text)

    def probe(self, subject: str, relation: str, obj: str) -> float:
        return self.agent.memory.network.path_strength(subject, relation, obj)


def run_protocol(
    max_neurons: int | None,
    curriculum: tuple[Skill, ...] | None = None,
    quiz_all: bool = True,
) -> dict[str, object]:
    """Teach each skill in turn; test it at once (and, optionally, re-test
    everything taught so far after each new skill)."""
    skills = curriculum if curriculum is not None else build_curriculum()
    stream = Stream(agent=Aster(max_neurons=max_neurons))
    matrix: dict[str, dict[str, float]] = {skill.name: {} for skill in skills}
    practice: list[dict[str, object]] = []
    for index, skill in enumerate(skills):
        stream.teach(skill)
        stream.chat("hello")
        stream.chat("status")
        heard = skills[: index + 1] if quiz_all else (skill,)
        for earlier in heard:
            result = stream.quiz(earlier)
            matrix[earlier.name][skill.name] = float(result["accuracy"])
        practice.append(
            {
                "after": skill.name,
                "yard_falling": round(stream.probe("letting go", "causes", "falling"), 4),
                "kitchen_pot": round(stream.probe("the pot", "is", "metal"), 4),
                "known_facts": stream.agent.memory.known_fact_count(),
                "neurons": stream.agent.memory.network.neuron_count,
            }
        )
    return {
        "max_neurons": max_neurons,
        "matrix": matrix,
        "practice": practice,
        "turns": stream.turns,
        "known_facts": stream.agent.memory.known_fact_count(),
        "neurons": stream.agent.memory.network.neuron_count,
    }


def run_reteach(max_neurons: int, curriculum: tuple[Skill, ...] | None = None) -> dict[str, object]:
    """In a tight brain: run the whole curriculum, then re-teach the fallen skill.

    Relearning re-shows the skill's practiced core first and lets the
    conversation practice it, then lets the rest arrive -- what is used
    survives the capacity pressure, what is never touched can go.
    """
    from aster.language import parse_input

    skills = curriculum if curriculum is not None else build_curriculum()
    stream = Stream(agent=Aster(max_neurons=max_neurons))
    matrix: dict[str, dict[str, float]] = {skill.name: {} for skill in skills}
    for skill in skills:
        # the light-touch protocol: each new skill is tested right after it
        # is taught; nobody keeps re-testing the old ones along the way
        stream.teach(skill)
        stream.chat("hello")
        result = stream.quiz(skill)
        matrix[skill.name][skill.name] = float(result["accuracy"])
    before = {skill.name: float(stream.quiz(skill)["accuracy"]) for skill in skills}
    fallen = min(skills, key=lambda skill: before[skill.name])
    core_subjects = {
        parsed.subject
        for question, _ in fallen.tests
        if (parsed := parse_input(question)).subject
    }
    core: list[Trial | Say] = []
    rest: list[Trial | Say] = []
    for lesson in fallen.lessons:
        fact = parse_input(lesson.text).fact if isinstance(lesson, Say) else None
        if fact is not None and fact.subject in core_subjects:
            core.append(lesson)
        else:
            rest.append(lesson)
    stream.teach_lessons(core)
    stream.quiz(fallen)  # practice the relearned core while talking
    stream.teach_lessons(rest)
    after = {skill.name: float(stream.quiz(skill)["accuracy"]) for skill in skills}
    return {
        "max_neurons": max_neurons,
        "matrix": matrix,
        "reteached": fallen.name,
        "before_reteach": before,
        "after_reteach": after,
        "turns": stream.turns,
        "known_facts": stream.agent.memory.known_fact_count(),
        "neurons": stream.agent.memory.network.neuron_count,
    }


def main() -> None:
    report = {
        "protocol": (
            "sequential conversational skills: teach by talking/doing, test by "
            "asking, one skill at a time, time passing every turn"
        ),
        "skills": [skill.name for skill in build_curriculum()],
        "roomy": run_protocol(max_neurons=3000),
        "tight": run_reteach(max_neurons=400),
    }
    out = Path("action-output/skill-stream/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
