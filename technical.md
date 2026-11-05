# Technical notes

## Purpose and boundary

See [brain-capabilities.md](brain-capabilities.md) for the functional inventory, [architecture.md](architecture.md) for the system design, and [first-experiment.md](first-experiment.md) for the proposed first train-from-scratch benchmark. These documents are planning references; this file describes what is implemented today.

Aster is an early hybrid symbolic/neural experiment. It contains a small PyTorch transition model trained from scratch for the Workshop environment, but no LLM, no general-purpose language model, and no biological brain simulation. The model is narrow and the integrated task controller uses explicit Workshop rules. Nothing here demonstrates general intelligence or human-like long-horizon behavior.

## Runtime and layout

- Python 3.11+, standard library at runtime.
- `src/aster/models.py` defines shared records and input categories.
- `src/aster/language.py` classifies a deliberately narrow set of text patterns.
- `src/aster/memory.py` owns facts, a bounded recent episode window, active goal state, and JSON persistence.
- `src/aster/reasoning.py` retrieves facts and follows transitive `is` relations with cycle protection.
- `src/aster/agent.py` coordinates the modules and selects deterministic responses.
- `src/aster/cli.py` provides the terminal loop.
- `src/aster/environment/` contains a seeded Workshop simulator, typed observations/actions, a numeric encoder, and expert/coverage data policies.
- `src/aster/cognition/world_model.py` defines the feed-forward transition network and checkpoint loader.
- `src/aster/cognition/policy.py` defines a separate neural action classifier trained by behavior cloning.
- `src/aster/cognition/controller.py` combines the learned one-step predictor with explicit Workshop action affordances and subgoal costs; it is not a general learned planner.
- `src/aster/cognition/planner.py` contains an experimental learned-model beam search, not currently used by the internal simulator runner.
- `src/aster/training/` collects transition data, splits by canonical layout hash, trains the transition model and policy from random initialization, evaluates held-out examples, and saves checkpoints/reports.
- `src/aster/train.py` exposes training; the simulator runner is an internal experiment and is not documented as a user-facing command.
- `tests/` exercises parsing, persistence, model/data training, the agent loop, and environment transitions.

Each module owns one responsibility; the agent is the integration seam rather than a place to accumulate the implementation of every subsystem.

## Data and behavior

Facts are directed triples (`subject`, `relation`, `object`) and currently accept only `is` and `has`. Matching is case-insensitive; original text is retained for display. Duplicate triples are ignored. For `is` queries, the reasoner follows further stored `is` facts breadth-first and avoids cycles. This is a small, transparent rule, not open-ended reasoning.

A bounded episode deque (seven entries by default) represents recent working context. Facts and the current goal are long-lived state. JSON persistence is optional for library users and enabled by the CLI at `~/.aster/memory.json`; `ASTER_MEMORY_PATH` overrides that location. Files are written through a temporary sibling and then replaced.

The Workshop is a small deterministic environment: tasks require collecting a key and opening a door, or operating a switch and crossing a bridge. Layouts are seeded, transitions are explicit, and episodes have a configurable step limit. Its fixed encoder produces 214 features for the default 7-by-5 map. Expert routes and counterfactual transitions are used only to build training/evaluation data.

`TransitionModel` receives one encoded observation and an action, then predicts next-state features, reward, terminal status, and action invalidity. It has two hidden layers and is trained with backpropagation from random initialization. The initial config generates transitions from 60 seed values, two random episodes per task, and 40 epochs. A canonical layout hash groups all repeated instances of the same geometry into one split. An audit found only 38 unique layouts per gate family among those 60 seeds, so the original seed-only split could leak repeated layouts; data splitting now holds out exact canonical layouts. It does not yet hold out rule families or compositions. Split files record layout IDs and SHA-256 hashes.

A separate `ActionPolicy` learns to imitate the expert action labels. It is evaluated on held-out canonical layouts with known-illegal actions masked, but without the reference controller's goal-distance heuristic. This is an early behavior-cloning baseline, not a reliable planner. The interactive conversation agent still only records and reports one active goal; it does not launch either trained component or work on chat requests autonomously. The internal Workshop simulator runner uses a hybrid reference controller with explicit action affordances and a hand-coded subgoal cost, as well as model predictions. User-facing play instructions remain withheld until learned behavior passes reliability evaluation. No continual weight updates, broad concept learning, real-world perception, or general planner is implemented. Unknown conversational text is recorded and reported as unsupported rather than guessed at.

## Current measured results (2026-11-05)

All numbers come from local run `runs/workshop-20261105-165630-17` (run directories and checkpoints are gitignored) using data schema v2 with canonical-layout splits: 53 train, 11 validation, and 12 held-out layouts (17,496 / 3,326 / 4,304 transitions). They are Workshop-only measurements, not evidence of general intelligence.

- World model, one-step test: loss 0.074; agent-position accuracy 0.9998; gate-state 0.977; key-state 0.993; dynamic-state bit accuracy 0.933; invalid-action accuracy 0.997.
- Hybrid reference controller, closed loop on the exact held-out cases: 12/12 solved, about 9.9 steps on success, no invalid actions. This controller combines model predictions with explicit affordances and a hand-coded subgoal cost, so it is a plumbing check, not learned planning.
- Behavior-cloning policy: expert-action accuracy 0.748 on test transitions; closed loop it solved 1/12 held-out cases versus 0/12 for random legal actions. The reference-controller gate passed; the policy benchmark thresholds did not. With 12 test cases the policy gap is suggestive, not conclusive, so the policy stays experimental and user-facing play instructions remain withheld.

## Install, run, test

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m aster
python -m unittest discover -s tests -v
```

The optional training dependency and first CPU training run are:

```sh
python -m pip install -e '.[train]'
python -m aster.train --config configs/workshop-first-run.json
```

To run the suite without installing the package, use `PYTHONPATH=src python -m unittest discover -s tests -v`. Model-training tests are skipped when the optional training extra is absent.

## CI log workflow

`.github/workflows/symbolic-check.yml` is intentionally keyed to the tracked `scripts/run-github-checks` marker (or manual dispatch). The workflow runs the test suite and commits its console log to `action-output/run.log` on the same branch, so the log is visible in the repository rather than only as an inaccessible artifact. Editing the marker and pushing it is the explicit way to request that run. The workflow does not run on ordinary source edits unless the marker is changed too.

This is a project-specific convenience with tradeoffs: it adds generated output to version control and the workflow needs permission to push to the triggering branch. Keep the log free of secrets; do not print credentials or environment dumps.
