# Changelog

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
