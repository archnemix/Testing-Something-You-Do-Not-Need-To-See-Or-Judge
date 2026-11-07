# Changelog

## 2026-11-06 (4) — English, Python, music: permanent learning proven across reloads

- `src/aster/case_study.py`: the long conversation — teach English (68
  lessons + a 1000-word real-English lexicon drill), confirm it, teach
  Python (59 lessons: syntax, concepts, usage), unload the model and
  reload it from disk, ask Python again; then the cycle repeats with
  music. Every phase timed, every reply recorded
  (`action-output/case-study/report.json`).
- Capacity sweep 5k -> 100k neurons: all accuracies 1.00 at every stage
  in every budget; learning is permanent (strengths byte-identical
  across save/unload/reload); per-phase timings recorded (100k run:
  english 0.4s, lexicon 77s, python 7.4s, music 2.7s, reload ~6-8s).
- Study fixes: learning runs in-memory with saving at unload (per-turn
  saving was re-serializing the whole memory after every reply), and a
  guard test requires every teaching line to parse as a teachable fact
  (seven lines with unsupported verbs were silently not taught). 65
  tests green.

## 2026-11-06 (3) — Skills through conversation: the world model is the model

- `src/aster/skills.py`: the requested protocol at study scale — six
  skills (one discovered by acting, five taught by talking), 180 facts,
  70 test questions, 404 conversational turns; teach skill 1, test it,
  teach skill 2 while talking, test whether it landed, and so on.
  Report: `action-output/skill-stream/report.json`. Roomy brain retains
  everything (all cells 1.00) and practiced pathways strengthen over the
  engagement; tight brain (400) shows the oldest skill fading (1.00→0.67)
  and a fallen skill relearning fully when shown again (0.50→1.00).
- Grammar extended for real teaching: facts and questions now cover
  `is/are/has/means/causes/makes` ("gravity means falling", "what does
  letting go cause?"); hop-2 wording names the relation it followed.
- Recognition fixes found by the study: a cue must reach a memory by its
  word-beginnings (a dead "vel" no longer gets confidently answered by
  look-alike "tivel"), and capacity eviction weighs connection strength
  only (imprints out — short-named concepts are no longer evicted first).
- Framing: the physics world is not a side demo — it is the world model
  proper, one substrate shaped by experience and talk, no separate fact
  store. 63 tests green.

## 2026-11-06 (2) — Selective forgetting, skills that survive years, grounded world model

- Consolidation: strong pathways decay far slower than weak traces
  (`w*(1-decay*(1-0.8w))`); trivia fades, practiced knowledge persists,
  and a forgotten detail first loses its answer while staying familiar
  (tip-of-the-tongue) before pruning.
- Savings: pathways remember peak strength and re-learn far faster than
  first learning (one re-telling reaches 0.85 vs 0.6 naive).
- Capacity: optional `max_neurons` budget with weakest-idle eviction
  (in-use ensembles protected; ties evict oldest) — used skills survive
  pressure, larger budgets hold more, and smaller ones still learn.
- `environment/physics.py` + `tests/test_physics_grounding.py`: causal
  knowledge (gravity, walls, pushing) formed from action trials, not
  telling; names bound after discovery ("gravity" → the experienced
  falling); the world, not a prior, decides outcomes. 58 tests green.

Notable user-facing and architectural changes are recorded here.

## Unreleased

- Established the first Aster symbolic cognition prototype with deterministic fact parsing, JSON-backed memory, goal tracking, transitive `is` retrieval, and a terminal interface.
- Added tests, project documentation, and an opt-in GitHub Actions marker workflow that stores its test log in the repository.
- Added a brain-capability inventory, a staged architecture for a from-scratch trainable world model, and a concrete first training/evaluation experiment.
- Added a seeded Workshop environment with key/door and switch/bridge tasks, a numeric observation encoder, and a privileged coverage baseline for collecting transitions.
- Added a PyTorch transition model and a separate expert-imitation action policy, both trained from random initialization, with local checkpoints and held-out evaluation.
- Audited data leakage: the seed-based split could place identical generated layouts in train and test. Splits now use canonical layout hashes, and manifests record layout IDs and SHA-256 checksums.
- Added a hybrid Workshop reference controller and a random-legal baseline. Neither the hybrid controller's success nor the current behavior-cloning results establish reliable learned planning; user-facing play examples remain withheld.
- Reran training and evaluation on the corrected canonical-layout split (data schema v2). The behavior-cloning policy solved 1/12 held-out cases versus 0/12 for random legal actions; it did not pass the policy benchmark thresholds and remains an experimental baseline.
- Completed the evaluation protocol: training-frequency and reactive baselines, privileged-oracle gap, free-running rollout error at horizons 1/3/5/10, and large-sample closed-loop evaluation on 338 unique unseen layouts (169 per gate family). The learned policy beats random and reactive baselines overall (14.8% vs 3.0% vs 0.0%) and clearly on door tasks, but stays far below the reliability gate (21.9% door / 7.7% bridge against 80%/60% targets), so play examples remain withheld.
- Expanded the generator's layout space (version 2) after measuring a hard cap of 60 layouts per gate family that made the required 100-scenario evaluation sample impossible; a regression test now guards the expanded space.
- Documented a frequency-baseline gate rule revision: `reward_mae` is reported but non-gating because constant per-action outcome prediction is near-optimal under MAE for the Workshop reward signal.
- Added closed-loop failure diagnostics (key never collected, gate never opened, opened never crossed, crossed never reached), a `--seed` run override, and `scripts/aggregate-metrics` for cross-seed reporting. Four-seed behavior cloning was fragile (3.3%-14.8% success; bridge advantage within noise).
- Added dataset aggregation (DAgger): the expert labels policy-visited states before retraining, never acting at evaluation. Pooled over four seeds, policy success rose from 10.5% to 25.4% and the bridge advantage over random is now established (20.3% vs 5.3%, z = 8.2), but the reliability gate is still not met and play examples remain withheld.
- Measured the uncertainty gate with a five-member seed ensemble: it fails in all four seeds (error-detection AUROC 0.49-0.54, chance level; unfamiliar disagreement only 1-12% higher). The system cannot yet say "I don't know" usefully; a probabilistic model head is the next architectural requirement. Fixed a float32 sigmoid-saturation bug in error flagging and locked it with a regression test.
- Verified restart recovery for the deterministic suite: exact progress restoration at every interruption point of expert episodes, and conversation-layer goal, episode history, and facts survive restart.
- Built and ran the architecture scaling study on GitHub Actions (4 vCPU / 16 GB): {project MLP, cell-token Transformer baseline} x {tiny, ~5M, ~10M} on identical Workshop data and training budget. Scaling the MLP helps with diminishing returns (test loss 0.175 -> 0.102); a 104k-parameter Transformer beats the 10M MLP (0.036 vs 0.102); naive Transformer scaling under the shared untuned recipe degrades (5M: loss 0.42 despite excellent decoded dynamics and the most stable rollouts; 10M: fails to train, agent-position accuracy 0.57). Full table and caveats in technical.md; raw report committed under `action-output/scaling-study/`.
- Replaced the fact dictionary of the conversation brain with a plastic neuron-and-synapse network (`PlasticNetwork` in `src/aster/cognition/plastic.py`). Telling things strengthens pathways immediately (Hebbian learning); answers form by spreading activation along learned pathways (no lookup, transitive `is` chains emerge from two hops, cycles resolve without bookkeeping); used pathways strengthen while idle ones decay and prune (use-dependent plasticity); new concepts recruit neurons and forgetting removes them (structural plasticity); input is a variable-length stream with no padding or context window. Persistence now stores synaptic strengths themselves. Each requirement has a behavioral proof in `tests/test_plastic.py`; design and next stages in `brain-plasticity.md`.
