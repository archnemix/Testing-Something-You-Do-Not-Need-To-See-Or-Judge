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
