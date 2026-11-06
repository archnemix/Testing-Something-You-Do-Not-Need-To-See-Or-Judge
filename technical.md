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

`TransitionModel` receives one encoded observation and an action, then predicts next-state features, reward, terminal status, and action invalidity. It has two hidden layers and is trained with backpropagation from random initialization. The initial config generates transitions from 60 seed values, two random episodes per task, and 40 epochs. A canonical layout hash groups all repeated instances of the same geometry into one split. An early audit found only 38 unique layouts per gate family among those 60 seeds, so the original seed-only split could leak repeated layouts; data splitting now holds out exact canonical layouts. The generator was then expanded (layout space version 2: gate column, object column, and start row all vary) after measuring that the original space capped at 60 layouts per family, which made the required 100-scenario evaluation sample impossible; the expanded space yields 680+ unique layouts per family over 1000 seeds. It still does not hold out rule families or compositions. Split files record layout IDs and SHA-256 hashes.

A separate `ActionPolicy` learns to imitate the expert action labels. It is evaluated on held-out canonical layouts with known-illegal actions masked, but without the reference controller's goal-distance heuristic. Training supports dataset aggregation (DAgger): the learned policy rolls out on training layouts and the privileged expert labels the visited states before retraining; the expert only supplies labels and never acts at evaluation. Evaluation compares the policy against three reference points: random legal actions, a reactive baseline that walks toward the target but never touches keys or switches, and the privileged oracle solver that estimates solvability. Closed-loop metrics include diagnostic failure counts (key never collected, gate never opened, opened but never crossed, crossed but never reached). A per-action training-frequency baseline (modal next outcome, ignoring the current state) scores the world model's one-step predictions with identical field metrics, and free-running rollout error is measured at horizons 1, 3, 5, and 10. This is an early behavior-cloning baseline, not a reliable planner. The interactive conversation agent still only records and reports one active goal; it does not launch either trained component or work on chat requests autonomously. The internal Workshop simulator runner uses a hybrid reference controller with explicit action affordances and a hand-coded subgoal cost, as well as model predictions. User-facing play instructions remain withheld until learned behavior passes reliability evaluation. Restart recovery is tested at every interruption point of every episode family (exact state restoration and identical outcomes) and at the conversation layer (goal, episode history, and facts survive restart). Predictive uncertainty is measured with a five-member seed ensemble; it does not yet identify the model's errors. Held-out rule compositions remain unmeasured. No continual weight updates, broad concept learning, real-world perception, or general planner is implemented. Unknown conversational text is recorded and reported as unsupported rather than guessed at.

## Current measured results (2026-11-05/06)

Run directories under `runs/` are gitignored; reproduce with `python -m aster.train --config configs/workshop-first-run.json --seed <N>`. One-step, rollout, and hybrid-controller figures are from the seed-17 run and reproduce deterministically; policy figures are pooled over training seeds 17, 23, 41, and 73. Each run uses data schema v2 with canonical-layout splits: 81 train, 17 validation, and 18 held-out layouts (18,160 / 3,592 / 3,864 transitions). They are Workshop-only measurements, not evidence of general intelligence.

- World model, one-step test: loss 0.141; dynamic-state bit accuracy 0.998; agent position 0.999; gate state 0.978; key state 0.976; done accuracy 0.988. Against the state-blind frequency baseline the model wins every gating field (agent position 0.999 vs 0.044, gate 0.978 vs 0.755, key 0.976 vs 0.775), so the frequency gate passes. `reward_mae` is the one field the constant-outcome baseline wins (0.025 vs 0.055); it is reported but non-gating because predicting a constant per-action outcome is near-optimal under MAE for this reward signal. The rule was revised after the first completed evaluation, with the reason recorded in the run report and journal.
- Free-running rollout error: state MAE 0.0063 at horizon 1 rising to 0.0152 at horizon 10; agent-position accuracy 0.999 at horizon 1 and 0.973 at horizon 10.
- Hybrid reference controller, closed loop: 18/18 held-out cases solved with no invalid actions. This controller combines model predictions with explicit affordances and a hand-coded subgoal cost, so it is a plumbing check, not learned planning.
- Large-sample evaluation (gate check over at least 100 scenarios per family): 338 unique unseen layouts, 169 per gate family, drawn from seeds 100-299 with every train/validation/test layout ID excluded. Oracle 338/338 and hybrid controller 338/338 in every run. Single-run behavior cloning was fragile across four training seeds (learned success 3.3%-14.8%, random 3.0%), so all claims below are pooled over seeds 17, 23, 41, and 73.
- Behavior cloning alone (no aggregation), pooled: 142/1352 (10.5%) versus random 40/1352 (3.0%); door 15.1% versus 0.6%, bridge 5.9% versus 5.3% (bridge within noise). Failure analysis showed the dominant failure is never performing the subgoal interaction at all (e.g. 129-162 of 169 bridge episodes never touched the switch), the classic compounding-error signature of imitation learning.
- With dataset aggregation (two rounds of expert labels on policy-visited states, about 4,000-5,000 labeled states per round), pooled: 344/1352 (25.4%) versus random 40/1352 (3.0%), z = 16.7; door 30.6% versus 0.6% (z = 15.2); bridge 20.3% versus 5.3% (z = 8.2), establishing the bridge advantage that behavior cloning alone did not. Per-seed learned success rose to 13.6%-31.4% (worst case lifted from 3.3%), and the failure bottleneck moved past the gate (bridge "never opened" fell from 129-162 to 39-118 of 169). Expert-action accuracy on held-out expert transitions rose from 0.70 to 0.73.
- Even with aggregation the reliability gate is far away (30.6% door / 20.3% bridge against 80%/60%) and the oracle gap is about 75 points, so the policy stays experimental and user-facing play instructions remain withheld. All numbers are Workshop-only measurements; local run directories under `runs/` are gitignored and `scripts/aggregate-metrics` prints the cross-seed tables.
- Uncertainty gate (five-member seed ensemble, disagreement as uncertainty): unfamiliar states produce only slightly more disagreement than familiar ones (ratio 1.01-1.12), and disagreement does not identify errors at all (error-detection AUROC 0.49-0.54, chance level) in any of the four seeds. Confidently-wrong predictions are rare in absolute terms (about 1%) only because errors are rare. The gate fails: the committee learns one deterministic function and agrees even when all members are wrong together. A genuinely probabilistic model head is the next architectural requirement; nothing in this system can currently say "I don't know" usefully.
- Restart recovery gate: pass for the deterministic suite. Every interruption point of expert episodes on eight seeded layouts restores exact progress (position, key, gate, step count) and restarts to bit-identical outcomes; the conversation agent restores its goal, bounded episode history, and facts from disk after a restart.

## Architecture scaling study (2026-11-06)

Question: what does the learned part of this system look like, and what happens when it is scaled to millions of parameters, compared with a Transformer on the same task? Reproduce with `python -m aster.scaling` (options `--epochs`, `--patience`, `--seed`, `--only`, `--output-dir`); it writes `runs/scaling-study-<timestamp>/report.json` incrementally after every cell. The full six-cell study is normally trained on the GitHub Actions runner (4 vCPU / 16 GB): change `scripts/run-scaling-study` and push, and `.github/workflows/scaling-study.yml` trains the cells in parallel job groups, merges their reports, and commits `action-output/scaling-study/report.json` back to the branch.

What is being scaled is the learned world model only: `TransitionModel` (`src/aster/cognition/world_model.py`), a multi-head perceptron over a flat state-action vector (the Workshop's symbolic engine, rules, memory, and controller are fixed machinery and stay tiny). `ActionPolicy` (`src/aster/training/policy.py`) is a same-size-class MLP. The comparison baseline is `TransformerTransitionModel` (`src/aster/cognition/transformer.py`): an encoder-only Transformer over 36 cell tokens (35 board cells carrying their local features, plus one global token carrying the global features and one-hot action), with learned positional embeddings. It predicts exactly the same heads as `TransitionModel`, so every existing evaluation applies to it unchanged. Both families are trained by the same engine (`fit_transition_model` in `src/aster/training/engine.py`) with the same data and optimizer.

Method: six cells = {MLP, Transformer} x {tiny, about 5M parameters, about 10M parameters}. Sizes are chosen by `aster.training.scaling` from exact parameter formulas (MLP: `h^2 + 440h + 217` for two hidden layers; Transformer: grid over depth/width/heads). Data is the standard schema-v2 stream from Workshop seeds 0-59 on the same canonical-layout splits as the main experiments (81 train / 17 validation / 18 test layouts: 18,160 / 3,592 / 3,864 transitions). One optimizer policy for every cell (Adam, learning rate 0.001, batch 128, early stopping patience 5, budget 25 epochs), so this is a matched-budget comparison, not a tuned leaderboard. Caveats: no per-architecture hyperparameter tuning (a Transformer-friendly learning rate or training length could move its numbers up, and vice versa); 25 epochs is a shorter budget than the main experiments' 40, chosen so the whole study fits in CI jobs; the training set of 18,160 transitions is small relative to the larger cells, so large-model rows measure what scale buys on limited data, which is itself the honest setting for this project.

Results from CI run 42213445282 (2026-11-06); the merged raw report is committed at `action-output/scaling-study/report.json` with per-job logs beside it. A `*` marks a cell that early stopping (patience 5) ended before the 25-epoch budget.

| Cell | Parameters | Epochs | Test loss | Agent-position acc. | Rollout MAE @10 | Train s | Inference s/example |
|---|---|---|---|---|---|---|---|
| MLP tiny (`TransitionModel`, h=128) | 72,921 | 25 | 0.1745 | 0.9995 | 0.0180 | 18 | 1.2e-6 |
| MLP 5M (h=2029) | 5,009,818 | 15* | 0.1232 | 0.9997 | 0.0094 | 100 | 4.1e-5 |
| MLP 10M (h=2952) | 10,013,401 | 21* | 0.1015 | 0.9995 | 0.0064 | 299 | 7.5e-5 |
| Transformer tiny (d_model 64, 2 layers) | 104,269 | 25 | 0.0364 | 0.9995 | 0.0020 | 165 | 9.9e-5 |
| Transformer 5M (d_model 640, 1 layer) | 4,966,413 | 20* | 0.4212 | 0.9997 | 0.0018 | 2,560 | 2.3e-3 |
| Transformer 10M (d_model 640, 2 layers) | 9,889,933 | 16* | 1.6899 | 0.5727 | 0.0357 | 5,960 | 3.1e-3 |

What the table says:

- Scaling the project MLP helps, with diminishing returns: 72k -> 5M -> 10M cuts test loss 0.175 -> 0.123 -> 0.102 and rollout drift 0.018 -> 0.009 -> 0.006, with the key-state field rising from 0.917 to about 0.99. The 5M and 10M rows are close and swap order across repeated runs, so the safe claim is "several times better than tiny, then noisy gains". On this data volume (18k training transitions) the family is near its saturation point.
- Architecture beats raw scale: the 104k-parameter cell-token Transformer (test loss 0.036) beats the 10M-parameter MLP (0.102) by nearly 3x while holding every discrete field above 0.986. The grid-token inductive bias is worth more than 100x parameters on this task.
- Naive Transformer scaling fails under the shared, untuned recipe. The 5M Transformer still predicts the discrete dynamics excellently (agent position 0.9997; its free-running rollouts are the most stable of all six cells, MAE 0.0018) but its raw loss is inflated (0.42): its decoded predictions are right while its raw outputs are mis-scaled, and reward error roughly doubles. The 10M Transformer genuinely fails to train this way -- validation loss plateaus near initialization (1.9 to 1.8) and agent-position accuracy drops to 0.57. Large Transformers are known to need learning-rate warmup and tuned schedules; this table measures out-of-the-box behavior at fixed budget, not each family's tuned best.
- Cost: MLP predictions take microseconds and its cells trained in 18-299 s; Transformer predictions take milliseconds (roughly 100x-2000x slower per example on CPU), and its 5M and 10M cells needed 43 and 99 minutes on the 4-core CI runner.

Repeat-run caution: individual numbers move a few points between machines (the same cells on the local 2-CPU sandbox produced slightly different third digits and different early-stopping epochs); the family-level ordering above held on both.

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
