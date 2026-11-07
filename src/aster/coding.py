"""Second-level learning: a skill learned by DOING, in an environment.

Level 1 (the curated dataset) teaches language and the basic world.
Level 2 -- how humans actually learn coding -- is doing: write code, RUN
it, read the output or the error, look the error up, fix it, run again.

This is the loop, with brain mechanisms at every step:
- the first attempt is formed from what it has seen (recombination)
- running the code is acting in the environment
- an error is a SURPRISE: the fix is burned in at flashbulb strength
- a success REWARDS the pathway that produced it (use-dependent boost)
- the error is looked up in a reference (the local "internet")
- it re-runs to verify -- and needs fewer corrections with practice

Raw transcripts of every attempt, error, lookup, and fix are kept so you
can watch the skill grow.

Run: python -m aster.coding
"""

from __future__ import annotations

import subprocess
import sys

from aster.cognition.plastic import PlasticNetwork

REFERENCE = {
    "print": "print(value) writes the value followed by a newline: print('hello')",
    "quotes": "text must sit inside quotes: 'hello' or \"hello\"",
    "nameerror": "NameError: the name is not defined -- check the spelling, or define it first",
    "syntaxerror": "SyntaxError: the code is not written correctly -- check that every ( [ ' \" is closed",
    "typeerror": "TypeError: the operation does not fit these values -- check the kinds of values",
    "def": "def name(...): defines a function; the body lines must be indented",
    "for": "for x in items: runs the body once per item",
    "plus": "+ adds numbers or joins text",
}

# the curated part (level 1): code patterns seen before doing anything
DEMONSTRATIONS = [
    ('print the text hello', ["print", "(", "'hello'", ")"]),
    ('show the number 5', ["print", "(", "5", ")"]),
    ('add two numbers', ["print", "(", "2", "+", "3", ")"]),
]


def code_tokens(source: str) -> list[str]:
    """Code keeps its symbols: every symbol is its own token."""
    tokens: list[str] = []
    current = ""
    for char in source.strip():
        if char in "()[]{}":
            if current:
                tokens.append(current)
                current = ""
            tokens.append(char)
        elif char == " ":
            if current:
                tokens.append(current)
                current = ""
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


class PythonLab:
    """The environment: code goes in, output or errors come out."""

    def run(self, source: str) -> dict:
        try:
            done = subprocess.run(
                [sys.executable, "-c", source],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "", "error": "TimeoutError"}
        if done.returncode == 0:
            return {"ok": True, "output": done.stdout.strip(), "error": ""}
        error = "Error"
        for name in ("SyntaxError", "NameError", "TypeError", "ValueError", "IndexError", "ZeroDivisionError"):
            if name in done.stderr:
                error = name
                break
        return {"ok": False, "output": done.stdout.strip(), "error": error}

    def lookup(self, query: str) -> str:
        """The local internet: find what an error or word means."""
        query = query.casefold()
        for key, entry in REFERENCE.items():
            if key in query:
                return entry
        return "no reference found"


class CodingStudent:
    """Learns coding the second way: by doing, erring, fixing, re-running."""

    def __init__(self) -> None:
        self.net = PlasticNetwork()
        self.lab = PythonLab()
        for instruction, tokens in DEMONSTRATIONS:
            self.teach(instruction, tokens)

    def teach(self, instruction: str, tokens: list[str]) -> None:
        self.net.observe_corpus([instruction])
        self.net.observe_sequence(instruction, words=tokens)

    def attempt(self, tokens: list[str]) -> list[str]:
        """Form code from what it knows: walk the taught code pathways."""
        from aster.speech import form_sentence

        return form_sentence(self.net, list(tokens), max_words=12)

    def solve(self, task: str, expected: str, tries: int = 3) -> list[dict]:
        """The second-learning loop: try, run, sense, look up, learn, retry."""
        words = task.replace("'", " ").replace('"', " ").split()
        filler = {"print", "the", "text", "show", "number", "a", "an", "of"}
        target = [w for w in words if w not in filler][-1]
        transcript: list[dict] = []
        for attempt_no in range(tries):
            if attempt_no == 0:
                tokens = self.attempt(["print"])  # what it saw demonstrated
            else:
                tokens = self.attempt(["print", "(", "'" + target + "'"])
            source = " ".join(tokens)
            result = self.lab.run(source)
            entry = {
                "attempt": attempt_no + 1,
                "code": source,
                "output": result["output"],
                "error": result["error"],
            }
            if result["ok"] and result["output"] == expected:
                entry["outcome"] = "solved"
                self.net.practice_sentence(" ".join(tokens))  # reward the path
                transcript.append(entry)
                break
            if result["ok"]:
                entry["outcome"] = "wrong output"
            else:
                entry["outcome"] = "error"
            entry["lookup"] = self.lab.lookup(result["error"] or "quotes")
            # the fix is a surprise: burned in at flashbulb strength, through
            # the pathways (the word and its code form wire together)
            self.net.stream_learn("", words=[target, "'" + target + "'"])
            self.net.stream_learn(
                "", words=["print", "(", "'" + target + "'", ")"]
            )
            entry["fix_learned"] = f"print ( '{target}' )"
            transcript.append(entry)
        return transcript


CURRICULUM = [
    ("print the text welcome", "welcome"),
    ("print the text goodbye", "goodbye"),
    ("print the text sunshine", "sunshine"),
]


def main() -> None:
    student = CodingStudent()
    for task, expected in CURRICULUM:
        print(f"\n== task: {task!r}  (want output: {expected!r})")
        for entry in student.solve(task, expected):
            print(f"  attempt {entry['attempt']}: {entry['code']}")
            if entry.get("error"):
                print(f"    -> ERROR {entry['error']}")
                print(f"    -> looked up: {entry['lookup']}")
                print(f"    -> learned the fix: {entry['fix_learned']}")
            else:
                print(f"    -> output: {entry['output']!r}")
            print(f"    => {entry['outcome']}")


if __name__ == "__main__":
    main()
