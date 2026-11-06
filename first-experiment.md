# First train-from-scratch experiment

This document turns the proposed architecture into the first falsifiable experiment. The aim is not to simulate a whole human brain in one model. It is to prove, from randomly initialized weights, that Aster can learn action consequences in a small world, use those predictions to solve held-out multi-step tasks, and preserve a useful internal state across interruptions.

## Experiment question

Can a small action-conditioned world model, trained only on experience from a controlled environment, predict what actions do well enough for a planner to solve new combinations of familiar objects and rules?

A “yes” is limited to this environment. It does not establish general intelligence, broad real-world knowledge, or human-equivalent learning.

## Environment: Workshop

Use a deterministic, seeded, grid-based workshop implemented in Python. The first version has a small map, an agent, a stateful gate, and a discrete action set. The environment is structured data, not pixels or free-form language, so failures can be traced to perception, prediction, planning, or control separately. A basic door/bridge simulator, seeded layout generator, numeric encoder, and rule-aware coverage baseline are now implemented under `src/aster/environment/`.

### Objects and rules

- Walls, open floor, and a target location.
- A movable key and a locked door; carrying the key and interacting with the door opens it.
- A switch and a bridge; interacting with the switch changes the bridge state.
- Optional distractor objects that are visually/structurally present but irrelevant to the goal.
- Actions: move north/south/east/west, pick up, interact, and wait.
- Invalid actions do not change object or agent state, are reported explicitly, and consume one step from the time limit.
- Every step returns observation, reward/progress signals, terminal status, and a structured event.
- Episodes have a configurable step limit; remaining time is included in the numeric observation encoding.

The environment must be reproducible from a seed. State transition code is the ground truth for evaluation and data collection. The Workshop reference controller receives structured layout facts and uses explicit action preconditions; this is a hand-built control prior, not a learned discovery.

### Task curriculum

1. **Single-effect tasks:** navigate to a target, pick up an item, or toggle a switch.
2. **Short dependencies:** acquire a key, reach a door, open it, and reach a target.
3. **Delayed consequences:** toggle a switch, cross the newly opened bridge, then reach the target.
4. **Distractor and interruption tasks:** ignore irrelevant objects, pause at a checkpoint, restart, restore the goal and progress, and continue.
5. **Held-out compositions:** use familiar rules in new layouts, object positions, and task combinations not seen together in training.

Task generators must state which rules and combinations are held out. Randomly splitting individual transitions is not an acceptable generalization test because adjacent transitions from one episode can leak into both splits.

## What is learned

The first learned component is a small neural, action-conditioned transition model initialized from random weights. Its input is an encoded current world state and action; its output predicts next-state fields, reward, terminal status, and whether the action is invalid. It currently has no event-class or calibrated uncertainty head. It receives no pretrained text model.

The initial encoding can be a fixed-size, typed representation: agent location and inventory; per-object type, location, and state; nearby map structure; current action. This is an engineering scaffold. The training/evaluation design should make it possible to replace the encoder with an object-centric representation as object counts and partial observability grow.

A separate neural action policy is trained only on expert actions as a behavior-cloning baseline. At evaluation it masks actions that violate known Workshop preconditions, but receives no hand-coded goal-distance ranking. Its reliability is measured against a random-legal policy on exactly the same held-out layouts. It remains experimental unless it clears the closed-loop gate.

The internal simulator runner uses a hybrid reference controller: it asks the model for one-step predictions, but uses known Workshop action affordances and a grid-aware subgoal cost to select a legal action. It then observes the real result and repeats. This proves the train/load/act/feedback plumbing, but the hand-coded mechanics and task heuristic do much of the planning; it is not evidence that the neural model learned long-horizon planning. A separate bounded beam-search planner uses model rollouts, but current tests found it unreliable. User-facing play examples remain withheld until learned-prediction reliability is demonstrated.

Initially, the goal is provided as structured data (for example, `at(target)`), not parsed from open-ended conversation. That isolates world learning from language understanding. Natural-language goal input comes later as an adapter into the same goal representation.

## Training-data protocol

- Generate episodes from many seeds and layouts.
- Include random exploration plus safe scripted coverage policies so every action and state-changing rule is observed often enough to learn.
- Store observations, action, next observation, outcome, goal, episode/seed ID, and environment version.
- Split by a canonical hash of each complete layout, so repeated seeds producing the same geometry stay in one partition and cannot leak across train, validation, and test. Gate family is part of the layout identity; this does not yet hold out entire task families or rule compositions.
- Record layout IDs and SHA-256 hashes in the manifest; never tune on the test split.
- Make data generation independent of learned-model predictions to avoid self-confirming errors.
- Keep a small hand-inspected set of transitions for debugging encoding and environment correctness.

A scripted coverage policy is a data-collection aid, not the deployed reasoner. The deployed planner must use its learned model and observed feedback.

## Training loop and first-run interface

Implemented first-run setup (the training extra installs PyTorch and NumPy):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[train]'
python -m aster.train --config configs/workshop-first-run.json
```

A first run is CPU-capable with a deliberately small network and dataset; a GPU is optional. The implemented `train` command validates settings and creates a unique run directory containing:

- resolved configuration, seed, Python/PyTorch versions, and environment version;
- JSONL train/validation/test splits, grouped by canonical layout hash, with layout IDs and SHA-256 hashes in the manifest;
- a best world-model checkpoint, a separate experimental behavior-cloning policy checkpoint, and their training histories;
- canonical-layout-held-out transition, policy, random-policy, and hybrid reference-controller metrics. The learned policy is not considered reliable for user-facing play.

Resume support, optimizer-state saving, detailed per-transition failure bundles, and calibrated uncertainty are not implemented yet.

No remote model/API is needed. The internal Workshop runner loads checkpoints locally; it is not connected to conversation memory or the general task system.

## Evaluation protocol

Use the same fixed scenarios and seeded task families for every comparison. Report distributions and per-family results, not just a single average.

### Baselines

- Random legal action selection.
- A reactive policy that moves toward the goal but does not model state-changing objects.
- A training-data-frequency transition predictor (most common next outcome).
- A privileged shortest-path/oracle planner for a small subset, used only to estimate task solvability and measure the gap—not as the deployed agent.

### Metrics

- One-step state-field and event prediction accuracy on held-out episodes.
- Rollout error at horizons 1, 3, 5, and 10, with uncertainty calibration.
- Goal completion rate, actions per completion, invalid actions, and catastrophic/irreversible mistakes by task family.
- Generalization to held-out layouts and held-out combinations of familiar rules.
- Recovery rate after an interruption/restart, including exact restoration of goal, episode history, and pending plan state.
- Resource use and inference latency on the reference CPU.

### Provisional first-run gates

These thresholds are initial engineering targets to be checked against the environment's difficulty before the test set is sealed. If a threshold is changed, document why and rerun all baselines.

- The model must beat the most-common-outcome predictor on held-out one-step transitions by a clear margin, not only on training data.
- The planner must complete at least 80% of held-out short dependency tasks and 60% of held-out delayed-consequence/composition tasks over at least 100 seeded scenarios per family.
- Performance on held-out layouts must exceed both the random and reactive baselines; report the measured gap to the oracle.
- Unfamiliar or ambiguous states must produce increased predictive uncertainty compared with familiar states; confident wrong predictions are a gating failure.
- No completion can be reported unless the environment's terminal condition confirms it.
- Restart recovery must retain the goal and verified progress for every interruption scenario in the deterministic suite.

Gate status after completed multi-seed evaluation (numbers in `technical.md`): the model beats the most-common-outcome frequency baseline on every gating field in all four training seeds, so gate one passes. The learned policy beats both the random and reactive baselines over 338 unique unseen layouts (169 per family, satisfying the scenario-count requirement) and the oracle gap is reported, satisfying the comparison part of gate three; with dataset aggregation this holds per-family including bridge tasks, pooled over seeds. The planning thresholds (80% door, 60% bridge) are not met — aggregated policy measured 30.6% and 20.3% pooled — so gate two fails and the policy remains experimental. Uncertainty calibration, interruption/restart recovery, and held-out rule compositions remain unmeasured. The reference controller can solve Workshop using explicit task rules, so its success rate does not count as a learned-planning pass. The beam-search model planner has not passed the planning gate.

These are deliberately bounded benchmark targets. Passing them means this training-and-planning loop works in Workshop; it does not mean the agent can independently do arbitrary real-world long-horizon work.

## Concept and thought generation in this experiment

The Workshop can test a narrow form of novel concept formation. After learning separate rules such as “key opens lock” and “switch changes bridge,” the system can compose a new plan or candidate relation in an arrangement absent from training. It should:

1. mark the plan as a prediction, not a memory or observed fact;
2. simulate its consequences with the learned model;
3. execute a safe first step and compare predicted with observed outcomes;
4. store the new episode and revise confidence when reality disagrees;
5. claim a learned concept only after the relation is supported across multiple independently seeded experiences.

To test novelty, compare the candidate against training episodes and stored concepts, then evaluate it on held-out compositions. “Novel to Aster's training memory” is a measurable claim; “never existed in the world” is not something this experiment can establish.

## Exit criteria and next expansion

Do not add open-ended autonomy or real-world tool access until the first experiment has reproducible training, held-out evaluation, failure analysis, and checkpoint rollback. If Workshop passes, expand one dimension at a time: partial observability, variable object sets, richer causal rules, longer delayed goals, then an integrated language adapter. Keep each expansion's tests and regressions in the suite.
