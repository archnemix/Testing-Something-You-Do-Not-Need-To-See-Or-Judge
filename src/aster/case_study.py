"""The long conversation: English, then Python, then music -- with reloads.

The study the user asked for: teach the agent English so it can speak and
be understood, confirm it knows English, then talk about Python and how to
use it; unload the model and reload it from disk -- what was learned must
have been learned permanently -- then ask about Python again to see. Then
the whole cycle repeats with another skill (music). Every phase is timed
and every reply is captured so the conversation can be read afterwards.

Capacity runs from 5k to 100k neurons; the lexicon drill teaches real
English words (web2 list via the ``english-words`` package) so the brain
grows into the thousands of neurons. Run ``python -m aster.case_study``.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from aster.agent import Aster
from aster.language import parse_input
from aster.models import InputKind
from aster.skills import Say, Skill, Trial

# -- the skills -------------------------------------------------------------

ENGLISH = Skill(
    name="english",
    kind="told",
    lessons=(
        Say("hello is a greeting"),
        Say("goodbye is a farewell"),
        Say("please means asking kindly"),
        Say("thanks means gratitude"),
        Say("sorry means an apology"),
        Say("yes means agreement"),
        Say("no means disagreement"),
        Say("i means the speaker"),
        Say("you means the listener"),
        Say("he means a male person"),
        Say("she means a female person"),
        Say("we means a group including the speaker"),
        Say("they means other people"),
        Say("what means asking about a thing"),
        Say("who means asking about a person"),
        Say("where means asking about a place"),
        Say("when means asking about a time"),
        Say("why means asking about a reason"),
        Say("how means asking about a manner"),
        Say("run means moving fast on foot"),
        Say("eat means taking food"),
        Say("sleep means resting with eyes closed"),
        Say("speak means using words aloud"),
        Say("read means understanding written words"),
        Say("write means putting words on a surface"),
        Say("think means using the mind"),
        Say("learn means gaining knowledge"),
        Say("teach means giving knowledge"),
        Say("help means making things easier"),
        Say("make means creating something"),
        Say("open means not closed"),
        Say("close means not open"),
        Say("water is a clear liquid"),
        Say("food is what people eat"),
        Say("a house is where people live"),
        Say("a book has pages"),
        Say("a word is a unit of language"),
        Say("a sentence has words"),
        Say("time is measured by clocks"),
        Say("money is used to buy things"),
        Say("a family is related people"),
        Say("big means large"),
        Say("small means little"),
        Say("hot means high temperature"),
        Say("cold means low temperature"),
        Say("fast means high speed"),
        Say("slow means low speed"),
        Say("good means of high quality"),
        Say("bad means of poor quality"),
        Say("new means recently made"),
        Say("old means long existing"),
        Say("happy means feeling joy"),
        Say("sad means feeling sorrow"),
        Say("a noun is a word for a thing"),
        Say("a verb is a word for an action"),
        Say("an adjective means describing a noun"),
        Say("a sentence means expressing a thought"),
        Say("a question means asking for information"),
        Say("an answer means a reply to a question"),
        Say("grammar is the structure of language"),
        Say("punctuation means marking sentences"),
        Say("a period means ending a sentence"),
        Say("a comma means pausing inside a sentence"),
        Say("capital letters are sentence starters"),
        Say("spelling is the form of a word"),
        Say("meaning is what a word conveys"),
        Say("plural means more than one"),
        Say("tense means locating an action in time"),
    ),
    tests=(
        ("what is hello?", "a greeting"),
        ("what is goodbye?", "a farewell"),
        ("what does please mean?", "asking kindly"),
        ("what does thanks mean?", "gratitude"),
        ("what does i mean?", "the speaker"),
        ("what does we mean?", "a group including the speaker"),
        ("what does who mean?", "asking about a person"),
        ("what does where mean?", "asking about a place"),
        ("what does read mean?", "understanding written words"),
        ("what does teach mean?", "giving knowledge"),
        ("what is water?", "a clear liquid"),
        ("what does a book have?", "pages"),
        ("what is a house?", "where people live"),
        ("what does big mean?", "large"),
        ("what does happy mean?", "feeling joy"),
        ("what is a noun?", "a word for a thing"),
        ("what is a verb?", "a word for an action"),
        ("what is grammar?", "the structure of language"),
        ("what does plural mean?", "more than one"),
        ("what does a question mean?", "asking for information"),
    ),
)

PYTHON = Skill(
    name="python",
    kind="told",
    lessons=(
        Say("python is a programming language"),
        Say("code is instructions for a computer"),
        Say("a program has many instructions"),
        Say("running code means executing instructions"),
        Say("print means showing text on the screen"),
        Say("a variable is a named value"),
        Say("a string is text data"),
        Say("an integer is a whole number"),
        Say("a float is a decimal number"),
        Say("a boolean is either true or false"),
        Say("a list is an ordered collection"),
        Say("a tuple is an immutable collection"),
        Say("immutable means unchangeable"),
        Say("mutable means changeable"),
        Say("a dict is a mapping of keys to values"),
        Say("a set is a collection of unique items"),
        Say("def means defining a function"),
        Say("a function is a reusable block"),
        Say("return means sending back a value"),
        Say("a loop means repeating a block"),
        Say("the for loop means iterating over items"),
        Say("the while loop means repeating conditionally"),
        Say("iteration means repeating a block"),
        Say("an if statement means choosing a branch"),
        Say("an else branch means the remaining case"),
        Say("a condition is a yes or no check"),
        Say("indentation means block structure"),
        Say("a comment means explaining code"),
        Say("import means loading a module"),
        Say("a module is a python file"),
        Say("a library is a collection of modules"),
        Say("a class is a blueprint for objects"),
        Say("an object is an instance of a class"),
        Say("a method is a function in a class"),
        Say("self means the current instance"),
        Say("none means no value"),
        Say("len means measuring length"),
        Say("append means adding an item"),
        Say("range makes a sequence of numbers"),
        Say("the open function means file access"),
        Say("a bug is an error in code"),
        Say("debugging means finding bugs"),
        Say("syntax is the spelling of code"),
        Say("a keyword is a reserved word"),
        Say("indexing means reaching one item"),
        Say("slicing means cutting a section"),
        Say("sorting means putting items in order"),
        Say("comparison means checking how values relate"),
        Say("assignment means storing a value in a name"),
        Say("the plus sign means addition"),
        Say("the equals sign means assignment"),
        Say("recursion means a function calling itself"),
        Say("a parameter is a name for an incoming value"),
        Say("an argument is a value passed to a function"),
        Say("testing means checking behavior"),
        Say("an exception is an error raised while running"),
        Say("handling exceptions means dealing with errors"),
        Say("a python file has the py extension"),
        Say("the interpreter is what runs python code"),
    ),
    tests=(
        ("what is python?", "a programming language"),
        ("what does print mean?", "showing text"),
        ("what is a variable?", "a named value"),
        ("what is a string?", "text data"),
        ("what is a list?", "an ordered collection"),
        ("what is a dict?", "a mapping of keys to values"),
        ("what does def mean?", "defining a function"),
        ("what does return mean?", "sending back a value"),
        ("what does the for loop mean?", "iterating over items"),
        ("what does indentation mean?", "block structure"),
        ("what does import mean?", "loading a module"),
        ("what is a class?", "a blueprint for objects"),
        ("what does self mean?", "the current instance"),
        ("what does none mean?", "no value"),
        ("what does debugging mean?", "finding bugs"),
        ("what is syntax?", "the spelling of code"),
        ("what does recursion mean?", "a function calling itself"),
        ("what does an if statement mean?", "choosing a branch"),
    ),
)

MUSIC = Skill(
    name="music",
    kind="told",
    lessons=(
        Say("a note is a single sound"),
        Say("a chord is notes played together"),
        Say("a scale is notes in order"),
        Say("a scale has seven notes"),
        Say("the piano has keys"),
        Say("the guitar has strings"),
        Say("a drum is a percussion instrument"),
        Say("percussion means struck instruments"),
        Say("a violin is a string instrument"),
        Say("tempo means the speed of music"),
        Say("rhythm means the pattern of beats"),
        Say("a beat is a pulse in music"),
        Say("melody means a tune"),
        Say("harmony means notes sounding together"),
        Say("pitch means how high or low a sound is"),
        Say("a rest means a pause in music"),
        Say("practice means repeating to improve"),
        Say("a song has a melody"),
        Say("a song has lyrics"),
        Say("lyrics means the words of a song"),
        Say("tuning means adjusting pitch"),
        Say("a chord makes harmony"),
        Say("loud means high volume"),
        Say("soft means low volume"),
    ),
    tests=(
        ("what is a note?", "a single sound"),
        ("what is a chord?", "notes played together"),
        ("what is a scale?", "notes in order"),
        ("what does the piano have?", "keys"),
        ("what does the guitar have?", "strings"),
        ("what does tempo mean?", "the speed of music"),
        ("what does rhythm mean?", "the pattern of beats"),
        ("what does melody mean?", "a tune"),
        ("what does harmony mean?", "notes sounding together"),
        ("what does pitch mean?", "how high or low a sound is"),
        ("what does lyrics mean?", "the words of a song"),
        ("what does practice mean?", "repeating to improve"),
    ),
)


def lexicon_skill(words: list[str], quiz_count: int = 30) -> Skill:
    """Real English words taught as words of English (the lexicon drill)."""
    lessons = tuple(Say(f"the word {word} is an english word") for word in words)
    stride = max(1, len(words) // quiz_count)
    sampled = words[::stride][:quiz_count]
    tests = tuple((f"what is the word {word}?", "an english word") for word in sampled)
    return Skill(name="lexicon", kind="told", lessons=lessons, tests=tests)


def load_lexicon(size: int) -> list[str]:
    """Distinct real English words, alphabetically spread (web2 word list)."""
    try:
        from english_words import get_english_words_set

        pool = sorted(
            w
            for w in get_english_words_set(["web2"], alpha=True, lower=True)
            if 3 <= len(w) <= 9
        )
    except Exception:  # pragma: no cover - fallback keeps the study runnable
        pool = [
            "amber", "bridge", "candle", "dancer", "eagle", "fable", "garden",
            "hammer", "island", "jungle", "kettle", "ladder", "meadow", "nickel",
            "ocean", "puzzle", "quiver", "ribbon", "shadow", "timber", "velvet",
            "willow", "yellow", "zephyr",
        ]
    stride = max(1, len(pool) // max(size, 1))
    return pool[::stride][:size]


# -- the conversation runner -----------------------------------------------


@dataclass
class CaseStream:
    """One timed conversation; every input and every reply is recorded."""

    agent: Aster
    transcript: list[dict[str, object]] = field(default_factory=list)
    turns: int = 0

    def say(self, text: str, quiet: bool = False) -> str:
        reply = self.agent.respond(text)
        self.turns += 1
        if not quiet:
            self.transcript.append({"input": text, "reply": reply})
        return reply

    def teach(self, skill: Skill) -> int:
        start = len(self.transcript)
        for lesson in skill.lessons:
            if isinstance(lesson, Trial):
                self.transcript.append({"input": f"[acts] {lesson.action}", "reply": ""})
                self.turns += 1
            else:
                self.say(lesson.text, quiet=True)
        taught = len(skill.lessons)
        self.transcript.append(
            {"input": f"[taught {skill.name}: {taught} lessons]", "reply": ""}
        )
        return len(self.transcript) - start

    def quiz(self, skill: Skill) -> dict[str, object]:
        hits = 0
        for question, expected in skill.tests:
            reply = self.say(question)
            if expected.casefold() in reply.casefold():
                hits += 1
        return {
            "accuracy": hits / len(skill.tests),
            "hits": hits,
            "total": len(skill.tests),
        }

    def probe(self, subject: str, relation: str, obj: str) -> float:
        return self.agent.memory.network.path_strength(subject, relation, obj)


def run_case_study(
    max_neurons: int,
    lexicon_size: int = 1000,
    memory_path: Path | None = None,
) -> dict[str, object]:
    """English -> Python -> reload -> music -> reload, all timed and logged."""
    path = memory_path or Path(f"/tmp/case-study-{max_neurons}.json")
    phases: list[dict[str, object]] = []
    accuracies: dict[str, float] = {}
    transcript: list[dict[str, object]] = []
    probe_triples = (
        ("hello", "is", "a greeting"),
        ("python", "is", "a programming language"),
        ("print", "means", "showing text on the screen"),
    )

    def probe_state(stream: CaseStream) -> dict[str, float]:
        return {
            " ".join(triple): round(stream.probe(*triple), 6)
            for triple in probe_triples
        }

    def reload_stream(old: CaseStream, marker: str) -> tuple[CaseStream, float]:
        start = time.perf_counter()
        old.agent.memory.path = path
        old.agent.memory.save()
        del old.agent
        gc.collect()
        agent = Aster(memory_path=path, max_neurons=max_neurons)
        agent.memory.path = None  # learning is in-memory; saving happens at unload
        seconds = time.perf_counter() - start
        stream = CaseStream(agent=agent)
        stream.transcript.append({"input": marker, "reply": ""})
        return stream, seconds

    def timed(name: str, stream: CaseStream, action) -> object:
        mark = stream.turns
        start = time.perf_counter()
        result = action()
        phases.append(
            {
                "phase": name,
                "seconds": round(time.perf_counter() - start, 3),
                "turns": stream.turns - mark,
            }
        )
        return result

    stream = CaseStream(agent=Aster(max_neurons=max_neurons))
    lexicon = lexicon_skill(load_lexicon(lexicon_size))

    timed("teach english", stream, lambda: stream.teach(ENGLISH))
    timed("teach english lexicon", stream, lambda: stream.teach(lexicon))
    accuracies["english"] = float(
        timed("quiz english", stream, lambda: stream.quiz(ENGLISH))["accuracy"]
    )
    accuracies["lexicon"] = float(
        timed("quiz lexicon", stream, lambda: stream.quiz(lexicon))["accuracy"]
    )
    timed("teach python", stream, lambda: stream.teach(PYTHON))
    accuracies["python"] = float(
        timed("quiz python", stream, lambda: stream.quiz(PYTHON))["accuracy"]
    )
    accuracies["english_after_python"] = float(
        timed("quiz english again", stream, lambda: stream.quiz(ENGLISH))["accuracy"]
    )

    before_reload = dict(accuracies)
    strengths_before = probe_state(stream)
    transcript.extend(stream.transcript)
    stream, seconds = reload_stream(stream, "[unloaded and reloaded from disk]")
    phases.append({"phase": "save, unload, reload", "seconds": round(seconds, 3)})
    strengths_after = probe_state(stream)
    permanent = strengths_before == strengths_after

    accuracies["english_after_reload"] = float(
        timed("quiz english after reload", stream, lambda: stream.quiz(ENGLISH))[
            "accuracy"
        ]
    )
    accuracies["python_after_reload"] = float(
        timed("quiz python after reload", stream, lambda: stream.quiz(PYTHON))[
            "accuracy"
        ]
    )
    accuracies["lexicon_after_reload"] = float(
        timed("quiz lexicon after reload", stream, lambda: stream.quiz(lexicon))[
            "accuracy"
        ]
    )

    timed("teach music", stream, lambda: stream.teach(MUSIC))
    accuracies["music"] = float(
        timed("quiz music", stream, lambda: stream.quiz(MUSIC))["accuracy"]
    )
    transcript.extend(stream.transcript)
    stream, seconds = reload_stream(stream, "[unloaded and reloaded from disk again]")
    phases.append({"phase": "save, unload, reload (2)", "seconds": round(seconds, 3)})

    accuracies["english_final"] = float(
        timed("quiz english final", stream, lambda: stream.quiz(ENGLISH))["accuracy"]
    )
    accuracies["python_final"] = float(
        timed("quiz python final", stream, lambda: stream.quiz(PYTHON))["accuracy"]
    )
    accuracies["music_final"] = float(
        timed("quiz music final", stream, lambda: stream.quiz(MUSIC))["accuracy"]
    )
    accuracies["lexicon_final"] = float(
        timed("quiz lexicon final", stream, lambda: stream.quiz(lexicon))["accuracy"]
    )
    transcript.extend(stream.transcript)

    network = stream.agent.memory.network
    return {
        "max_neurons": max_neurons,
        "lexicon_size": lexicon_size,
        "phases": phases,
        "accuracies": accuracies,
        "before_reload": before_reload,
        "strengths_before_reload": strengths_before,
        "strengths_after_reload": strengths_after,
        "permanent": permanent,
        "neurons": network.neuron_count,
        "synapses": network.synapse_count,
        "known_facts": stream.agent.memory.known_fact_count(),
        "turns": stream.turns,
        "transcript": transcript,
    }


def main() -> None:
    report: dict[str, object] = {
        "study": (
            "teach english, confirm, teach python, unload/reload, ask python "
            "again, teach music, unload/reload again -- capacities 5k to 100k"
        ),
        "runs": [],
    }
    for budget in (5000, 10000, 25000, 50000, 100000):
        print(f"== running at {budget} neurons ==", flush=True)
        run = run_case_study(max_neurons=budget)
        if budget != 100000:
            run["transcript"] = run["transcript"][:40]  # keep the report small
        report["runs"].append(run)  # type: ignore[union-attr]
        acc = run["accuracies"]  # type: ignore[index]
        print(
            f"  english {acc['english']:.2f}  lexicon {acc['lexicon']:.2f}  "
            f"python {acc['python']:.2f}  music {acc['music']:.2f}  "
            f"permanent {run['permanent']}  neurons {run['neurons']}",
            flush=True,
        )
    out = Path("action-output/case-study/report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    headline = report["runs"][-1]
    print("\n== sample replies (100k run) ==")
    for entry in headline["transcript"]:  # type: ignore[index]
        if entry["input"].startswith("["):
            print(entry["input"])
        elif entry["reply"]:
            print(f"  Q: {entry['input']}\n  A: {entry['reply']}")
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
