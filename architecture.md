# Proposed system architecture

This document turns the capability inventory into an architecture for the stated goal: train an agent from an initialized model, then let that same cognitive system learn, remember, plan, act, and communicate without relying on an attached LLM to do its reasoning. This is a research program and staged engineering plan, not a claim that the current prototype already has these abilities.

## Design rules

1. **The cognitive core owns decisions.** Perception, memory, prediction, goals, planning, and action selection belong to Aster. No external chat model supplies hidden reasoning.
2. **Ground learning in experience.** The initial trainable model learns from observations, actions, and outcomes in a controlled environment. Language is one kind of observation/action channel, not the only source of a world model.
3. **Use learned and explicit representations together.** Neural models can learn perception and dynamics; explicit records preserve goals, provenance, constraints, and commitments. Neither representation is treated as sufficient by itself.
4. **Keep hypotheses separate from beliefs.** Simulated possibilities and generated concepts are marked as hypothetical until supported by observations or successful tests.
5. **Make training repeatable.** A run declares its data, environment, seed, model settings, and evaluation suite. A checkpoint is usable only with its matching configuration and recorded results.
6. **Measure capabilities, not appearances.** Fluent text is not evidence of learning, understanding, planning, or task completion.

## Runtime components

```text
observations / user requests / tool results
                  |
        modality-specific input adapters
                  |
      grounded state and event perception
                  |
       shared concept and world state <------> episodic / semantic / procedural memory
                  |                                      |
         uncertainty and attention <---------------------+
                  |
       goal manager and task hierarchy
                  |
 world model -> candidate rollouts -> planner / policy
                  |                         |
          critic / verifier <---------------+
                  |
      permission-checked action adapters
                  |
       environment / tools / user
                  +-------- outcome feedback ---------> memory and learning

Internal thoughts are structured state, predictions, questions, and plans.
A language channel verbalizes selected internal content; it does not invent the plan.
```

### Component responsibilities

- **Input adapters:** convert text, structured observations, and later images/audio/tool results into typed observations. Preserve source, timestamp, confidence, and modality. Never silently turn a guess into a fact.
- **Perception and grounding:** identify entities, properties, events, and relations; track identity and state over time. Start with a small fully observable synthetic environment, then add partial observability and additional modalities.
- **World model (first implementation):** a trainable, action-conditioned predictor. Given a fixed numeric Workshop observation and action, it predicts next-state fields, reward, terminal status, and whether the action is invalid. It is trained from scratch; it is not a prompt-to-answer service. This first model does not yet estimate calibrated uncertainty or handle partial observations.

### Artificial neurons and synapses

The current `TransitionModel` is a conventional feed-forward neural network: inputs connect to two hidden layers of artificial units, and output heads predict state and outcome fields. Each connection has a trainable weight; each unit combines weighted inputs with a bias and nonlinear activation. Those weights are a useful software analogue of synaptic strengths. Training compares predictions with recorded transitions, computes loss, and uses backpropagation with AdamW to adjust the weights. No pretrained language weights are loaded.

This is **not** a biological neuron simulation. It does not model spikes, dendrites, neurotransmitters, local synaptic plasticity, sleep, brain regions, or the dense recurrent dynamics of a real brain. Nor does a standard neural network become brain-like merely by calling weights “synapses.” Aster currently has a small feed-forward predictor plus explicit symbolic state. Recurrent memory, continual learning, concept abstraction, and long-horizon planning remain separate capabilities to build and measure.
- **Memory:** distinct stores for recent working context, episodic experience, semantic concepts/facts, reusable procedures, and pending intentions. Every durable claim retains provenance and whether it was observed, instructed, inferred, or simulated.
- **Attention and goal manager:** select task-relevant memories and observations; maintain explicit goal trees, priorities, deadlines, constraints, and completion conditions. A goal is not considered progress by merely being restated.
- **Planner and policy:** the current Workshop reference controller uses a learned one-step model together with explicit action affordances and a grid-aware subgoal cost. That hybrid can solve the deterministic toy tasks, but the hand-written task structure is doing most of the planning; this is not a learned general planner. A separate beam-search experiment exists, but is not accepted as a reliable controller yet. The long-term target is to shift more of prediction and planning to learned representations only as measured tests justify it.
- **Critic and verifier:** compare predictions with actual outcomes, check plan preconditions, track uncertainty, detect repeated failure, and decide whether to continue, revise, ask, or stop.
- **Action adapters:** execute only allowed, bounded operations; log intended action, actual result, and side effects. Training environments are isolated from real tools.
- **Language channel:** understand and express content through its own input/output adapter. The cognitive core selects the thoughts and conclusions. The first interface can use a constrained grammar; later language competence must be trained as a modality of the same agent, not delegated to a separate LLM that does the thinking.

## Training and operation lifecycle

Training produces the cognitive core; operation continues learning from new experience under explicit safeguards. These are separate modes with separate state and permissions.

### First training run: from initialized weights

The first end-to-end training target should be modest enough for an ordinary development machine and meaningful enough to test learning. Use a procedurally generated object-and-tool environment with visible state, action feedback, simple physical or logical rules, and multi-step goals. Randomize object attributes and layouts. Hold out combinations and task templates for generalization tests.

The training command is a target interface, not implemented yet:

```sh
python -m aster.train --config configs/first_run.json
python -m aster.evaluate --run runs/<run-id>
python -m aster.run --checkpoint runs/<run-id>/best.pt
```

The first run should:

1. Validate the configuration, dataset/environment version, output path, and resource budget.
2. Set and record random seeds; initialize model weights from scratch (no pretrained language model).
3. Collect or generate experience tuples: observation, action, next observation, outcome, goal, and episode boundary.
4. Train the perception/state encoder and action-conditioned world model on next-state prediction, object/event consistency, and outcome prediction.
5. Train goal-conditioned value/policy components on short tasks, then progressively longer tasks and delayed outcomes.
6. Keep scenarios—not random individual transitions—separated between training, validation, and held-out evaluation to prevent leakage.
7. Evaluate checkpoints on prediction accuracy, calibrated uncertainty, task success, novel combinations, and recovery after interruption.
8. Save the best checkpoint, configuration, data/environment manifest, metrics, and failure examples in one run directory.
9. Fail the run or mark the model not ready if required gates are missed; never silently label an untested checkpoint capable.

The initial environment can be small and synthetic, so the first training run can be affordable and reproducible. That only demonstrates that the learning loop works in that environment. It does not establish broad real-world intelligence. Expanding to real-world multimodal competence requires suitable data, environments, compute, safety work, and repeated evaluation.

### Learning after initial training

At runtime, aster may add episodes and update memory immediately. Neural weight updates should initially happen in controlled, replay-based update jobs, not uncontrolled after each conversation. Each update is evaluated against a fixed regression suite and older capabilities before a checkpoint can replace the deployed one. This limits catastrophic forgetting and makes rollback possible. Continual learning research can later reduce this separation if tests show it is safe and useful.

## How new concepts and thoughts are formed

“New” must be defined operationally. Aster can propose a concept or plan that is new to its stored knowledge by composing learned parts; it cannot treat a generated possibility as a discovered fact.

1. **Acquire grounded primitives:** learn objects, attributes, relations, actions, and effects from varied episodes.
2. **Abstract repeatable structure:** compare episodes, identify stable relations, and form a candidate category, rule, skill, or causal model. Keep its supporting episodes attached.
3. **Compose:** combine concepts and relations into candidate explanations, plans, designs, or imagined scenarios. Search can use analogy, constraint solving, and rollouts of the world model.
4. **Check novelty:** compare the proposal against stored concepts and prior episodes. Report “new to this system’s memory,” not “never existed” or “first ever.”
5. **Test:** predict consequences, run safe simulations, seek counterexamples, or request a real observation/experiment when necessary.
6. **Classify and retain:** store the candidate as a hypothesis, simulated result, or verified belief with confidence and provenance. Promote it only when evidence meets the relevant standard.

Novel thought generation is therefore an explicit loop of abstraction, recombination, prediction, and checking—not a claim that the system creates knowledge from nothing. A vivid or coherent imagined result remains imagined until tested.

## Long-horizon task loop

For each authorized goal, the runtime should:

1. Parse the desired outcome and confirm ambiguous success conditions.
2. Create a goal record with priority, constraints, resources, and a measurable completion test.
3. Decompose it into dependencies and manageable subgoals.
4. Retrieve relevant episodic, semantic, and procedural memory with source and confidence.
5. Predict outcomes for candidate actions; avoid plans whose key predictions are too uncertain.
6. Execute one bounded action, record the actual result, and compare it with the prediction.
7. Update the world state and memory; replan when progress stalls or assumptions fail.
8. Persist progress across restarts and interruptions without confusing an intention with an accomplished action.
9. Verify the original completion test and report evidence, remaining uncertainty, and unfinished work.

## Proposed implementation stack

- **Language:** Python for orchestration, experiments, data tooling, and readable subsystem boundaries.
- **Trainable models:** PyTorch, introduced when the first learned world model is implemented. Start with a small model and generated environment; no pretrained LLM dependency.
- **Structured durable state:** SQLite once memory needs transactions, provenance, and queries beyond the current JSON prototype. Keep model checkpoints and run manifests separate from the database.
- **Configuration and runs:** versioned JSON/TOML configurations and immutable per-run output folders containing checkpoint, data/environment hashes, metrics, and logs.
- **Evaluation:** deterministic unit tests plus seeded scenario suites and held-out long-horizon tasks. Report success curves, not only an aggregate score.

These are proposed choices. Add each dependency only with its first actual subsystem and a tested reason.

## Acceptance gates

Before each stage, add a benchmark set with fixed seeds, held-out variants, baselines, and failure reporting. Initial gates below are minimum engineering checks, not evidence of human-level cognition.

| Stage | Required behavior | Evidence required to advance |
|---|---|---|
| 0. Task specification | Clear long-horizon tasks, outcomes, permissions, and failure costs | Written scenarios, success checks, baselines, and regression set |
| 1. Grounded world learning | Predict changes caused by actions in the training environment | Held-out next-state/outcome metrics beat a no-learning baseline; uncertainty rises on unfamiliar cases |
| 2. Memory and concepts | Store episodes, derive candidate concepts, and retrieve supporting evidence | Cross-episode transfer on held-out combinations; provenance intact; corrections do not erase unrelated facts |
| 3. Goals and planning | Complete multi-step tasks with delayed outcomes | Success measured over increasing horizons; progress survives restart/interruption; no false completion reports |
| 4. Novel hypotheses | Generate and test combinations not present verbatim in training episodes | Novel-to-memory candidates are labeled correctly, tested safely, and rejected when counterexamples invalidate them |
| 5. Continual learning | Incorporate new experience without erasing stable prior skills | New-task improvement plus non-regression on a fixed old-task suite; rollback demonstrated |
| 6. Expanded modalities/tools | Ground language and other input/action channels in the same world model | Cross-modal task performance and action/result traces; no hidden LLM reasoning dependency |

## Current status

The repository now has the symbolic interface, seeded Workshop simulator, deterministic expert/coverage data generation, leakage-safe seed splits, a PyTorch transition network trained from random initialization, a held-out transition evaluation, checkpoint loading, and a simulated-play CLI. One full CPU run trained on 17,510 transitions and evaluated on 3,848 held-out transitions. In that run the agent-position metric was 99.95%, but the benchmark is a small, finite structured world and the controller's subgoal logic is hand-written. This number is not evidence of human-like reasoning.

Closed-loop checks now run on held-out Workshop layouts. The integrated reference controller combines the learned predictor with explicit Workshop affordances and task-distance rules; its success must not be attributed to the neural model alone. The general beam-search controller remains experimental. No persistent autonomous goal loop, continual weight learning, concept invention, real-world perception, or language understanding beyond the original narrow grammar has been implemented. Next, separate learned-policy contribution from the symbolic baseline, improve model-based planning, and add tasks with held-out causal compositions before scaling the architecture.
