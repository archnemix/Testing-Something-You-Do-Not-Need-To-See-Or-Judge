# Aster

Aster is an early hybrid symbolic/neural cognition experiment. Its conversation interface stores a narrow set of explicit facts and goals; its separate Workshop experiment trains a small neural world model from scratch. It does not call an LLM, and it does not claim to reproduce a human brain.

The aim is to build a testable foundation and extend it in small, observable steps—not to disguise a narrow prototype as general intelligence.

## Try it

Requires Python 3.11 or newer. The basic conversation CLI uses only the standard library; neural training requires the optional PyTorch extra shown below.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m aster
```

Aster stores its memories in `~/.aster/memory.json` by default. To use another file:

```sh
ASTER_MEMORY_PATH=./memory.json python -m aster
```

Teach it a fact, then ask about it:

```text
> Ada is a mathematician.
> what is Ada?
Ada is a mathematician.
```

Facts currently use the form `subject is object` or `subject has object`. Ask `what is <subject>?` or `what does <subject> have?`. Set a persistent goal with `goal: <description>`, or type `status` to inspect active state. Type `help` for the command list and `quit` to exit.

## Train the first model

The training extra uses PyTorch. The first run starts from randomly initialized weights and trains an action-conditioned Workshop model; it does not download or call a pretrained LLM.

```sh
python -m pip install -e '.[train]'
python -m aster.train --config configs/workshop-first-run.json
```

The command prints a run directory containing the checkpoints, dataset splits, configuration, and evaluation report. The simulator runner remains an internal experiment: the learned policy has not met the reliability gate for user-facing play. Aster is not a general-purpose assistant or a human-level brain.

## Development

```sh
python -m unittest discover -s tests -v
```

See [brain-capabilities.md](brain-capabilities.md) for the capability inventory, [architecture.md](architecture.md) for the system design, [first-experiment.md](first-experiment.md) for the first training and evaluation experiment, [technical.md](technical.md) for the current implementation, [thinking.md](thinking.md) for the engineering journal, and [CHANGELOG.md](CHANGELOG.md) for notable changes.
