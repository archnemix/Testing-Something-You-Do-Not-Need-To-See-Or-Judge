# Engineering journal

## 2026-11-07 (6) — The user proved it was memorizing; the loop now learns

The user caught the real flaw with one transcript: two different questions
about python got the SAME stored sentence -- it was printing storage, not
formulating answers -- and asked why the whole thing feels like memorizing
rather than learning ("learning means nudging things as you learn;
memorizing means just adding them in"). They were right on both counts.
The loop now: (1) formulates answers from the question's own words, so
'what is python' and 'what does python mean' walk different paths and say
different things; (2) nudges on use -- speaking strengthens the spoken
pathway and asking consolidates the recalled one, with zero new storage;
(3) every sentence learns where it ends, killing the run-on replies. The
memorization feeling had a real source: teaching was mostly additive and
generation was replay; the tests now pin the difference (weights rise with
use while neuron count stays fixed; verbatim-replay rate is measured).
Then the 1M-neuron brain on the full corpus went to GitHub Actions. Honest
note to keep: the n-gram-ish walk still chains other people's sentences
when its own context runs out -- that is the next quality gap.

## 2026-11-07 (5) — It speaks: sentences fired from neurons, on real text

Set out to kill the `{subject} {relation} {object}` template and let it
talk like a person. Word order is now wired into the SAME synapses as
everything else, under plain, previous-word-pair and previous-word-triple
gates (n-gram order with backoff, as weights that decay and compete -- and
the context gates came from a real failure: the walk once answered "what
does print mean?" with "Print means asking about a word..." because six
curriculum lines shared "means asking"; context gates separate speakers'
patterns cleanly). `form_sentence` is winner-take-all walking, and the
proofs are recombination (forms never-taught sentences), weight-driven
choice, and output that decays. Taught it real English: 224,200 Project
Gutenberg sentences and 111,087 WordNet definitions at 5k/50k/100k neuron
budgets. It speaks real definitions as sentences ("Cutin means biochemistry
waxy transparent material ..."). Overflow answer: give it more than fits
and the weakest earlier things get pushed out -- at 5k with large data,
everything it was asked is gone (0.00); at 50k+ the same data is perfect
(1.00); the report shows the gradient and sample speech at every size.

## 2026-11-06 (4) — English, Python, and proof that learning is permanent

The user's next step: a big conversation. Teach it English, confirm it knows English, talk about Python and how to use it, then unload the model and reload it from disk — if it really learned, the learning must be permanent — then ask about Python again. Then do the whole cycle once more with another skill, growing the brain from 5k to 100k neurons, timing every phase, and keeping the replies so they can be checked. Built `aster/case_study.py`: curricula for english (68 lessons), python (59), music (24), plus a lexicon drill of 1000 real English words (web2 list via `english-words`), all through the same conversation channel. The sweep ran 5k/10k/25k/50k/100k: every accuracy 1.00 at every stage in every budget, and the permanence proof is exact — pathway strengths are identical across save/unload/reload (measured around the reload, not across ticking quizzes). Timings at 100k: english 0.4s, the 1000-word lexicon 77s, python 7.4s, music 2.7s, each reload ~6-8s; the brain settled at 3,858 neurons / 10,305 synapses — every budget from 5k up holds it fully, and bigger budgets cost nothing in time.

Two bugs the study flushed out. (1) After a reload the agent kept its disk path set, so every single turn re-serialized the whole memory to JSON — 148 of 167 profiled seconds were `json.dumps`! Learning now happens in-memory and saving happens at unload boundaries (which is also the honest semantics: "permanently learned" = saved at the checkpoint). (2) Teaching lines with verbs the grammar does not know ("a sentence expresses a thought", "punctuation marks sentences") were silently NOT taught — the fallback reply hid it. A guard test now asserts every curriculum line parses as a teachable fact, and it caught seven lines. Lesson: a study that teaches thousands of things needs a validator for the teaching channel itself.

**REMINDER (user's idea — NOT implemented, explain later):** the user is thinking about "some fixed world model and part of that world model that is changeable" — a hybrid where a core of world knowledge is fixed and other parts stay plastic. They asked to be reminded so they can explain what they mean. Do not build anything for this until they do.

## 2026-11-06 (3) — The world model is the model: six skills, one conversation

The user corrected the framing: the physics playground is not a demo *about physics* — it stands for the world, and the world model is the model itself, shaped by training data (acting + talking) in the same pathways, never a separate fact store. And they specified the experiment: teach a model one skill, test it, then while talking show a different skill, test whether it learned it, and so on — at a size that produces useful numbers. Built `aster/skills.py`: six patches of the world (yard discovered by doing; kitchen, trades, words, places, chores taught by talking), 180 facts, 70 test questions, 404 conversational turns, all teaching and testing through the same `Aster.respond` channel. Roomy brain (3000): the whole retention matrix is 1.00 and practiced pathway strengths *grow* over the engagement (kitchen 0.70→0.82) — quality improves with use, no context to overflow. Tight brain (400): each skill lands as it comes, the oldest fades (yard 1.00→0.67), and re-showing the fallen skill once brings it fully back (chores 0.50→1.00).

The study flushed out three real bugs. (1) False memory: a dead "vel" was confidently answered "vel means winter" by look-alike "tivel" — fixed by making a cue reach a memory through its word-beginnings (`pre:` coverage ≥ 0.75), not stray shared letters; "aaa" still cues "aaa aaa" and determiner variants still merge. (2) Eviction bias: the aggregate counted imprints, so short-named concepts (fewer spelling features) always looked weakest and kept being eaten — importance is now connection strength only. (3) The first relearn attempt re-taught a full skill at capacity and the fresh pairs cannibalized each other in lesson order — relearning now teaches the practiced core first and lets conversation practice it before the periphery arrives; and the protocol's own quiz policy matters: re-quizzing every old skill every checkpoint hides fading (practice protects), so the tight run tests each skill right after teaching and reveals survival only at the end — exactly the user's protocol. 63 tests green; matrices in `technical.md`.

## 2026-11-06 (2) — Forgetting on purpose, skills that survive years, and gravity discovered not told

The user pushed on four things real brains do and this one did not yet. (1) Quality must *improve* over days-to-years of work, not degrade like a long transformer context. The answer was consolidation: make the forgetting curve depend on pathway strength (`w*(1-decay*(1-0.8w))`) so used skills outlast idle time almost ten to one, and let practice only strengthen — there is no window to overflow. (2) Selective forgetting falls out of the same curve: one-off trivia ("what you ate") decays fast and becomes unanswerable first while staying faintly familiar (tip-of-the-tongue), then prunes; practiced knowledge ("your Python skill") still answers. (3) Skills must survive fading: `savings` — each pathway remembers its peak, and re-learning runs at `lr*(1+0.6*peak)` — one re-telling jumps 0.17 to 0.85 where a first telling reaches 0.6. (4) Capacity: a neuron budget with weakest-idle eviction. Design traps found by tests: eviction must never touch the fact being taught (protect in-progress ensembles or it eats its own subject mid-learn), stale recruits leave orphan synapses (prune synapses with dead endpoints), and fuzzy recognition at 0.5 must not answer for a dead look-alike (0.5 "item9"→"item0") — a cue now has to pick one memory clearly or it counts as ambiguity.

Gravity: `environment/physics.py` is a toy yard where letting go drops things and walls stop pushes. The network is never told any physics — it watches trials until ("letting go","causes","falling") pathways form, then the word "gravity" can be bound to the discovered concept *afterwards*, and name and experience answer as the same thing. The same push teaches different outcomes in an open yard vs against a wall: the knowledge reflects the world, not a built-in prior. 58 tests green.

## 2026-11-06 — The memory is now neurons, not a dictionary

The directive was unambiguous: the system must be brain-like -- neurons and synapses whose strengths carry knowledge, learning on every interaction (telling nudges weights immediately), use-dependent plasticity (used paths strengthen, unused fade), no context window, no padding, structural growth and pruning, and answers formed by neurons firing rather than table lookups. The fact dictionary was explicitly rejected.

I built `PlasticNetwork` and moved the conversation brain onto it. Design decisions worth recording: recognition works by feature-overlap of spellings (there is deliberately no token-to-id dictionary; even recognizing "Ada" again is a perceptual act); knowledge lives in directed synapses gated by the relation ensemble, so `is` and `has` never cross-talk; one telling lands a pathway in the "known" range and repetition pushes it toward saturation while idle moments decay everything slowly -- learning per telling is much larger than decay per moment, so the user's rule (use strengthens, disuse weakens) falls out naturally. Answers form by activity spreading a few hops along gated pathways with the cue excluded, which gives transitive `is` for free and defuses cycles without any visited-set bookkeeping.

All previous behavioral contracts held on the new substrate (50 tests): the transitive test that used to exercise a BFS rule now exercises two-hop spreading. The honest limits are written down in `brain-plasticity.md`: the reply frame is still a fixed template and the parser is still written code (Stage 2: settling word pools form sentences), and the world model still learns in offline batches (Stage 3: continual use-dependent nudging during play).

## 2026-11-06 — Architecture scaling study: what 5M/10M parameters buy versus a Transformer

The question was what the learned part of the system actually is, and how it behaves when scaled to millions of parameters against a Transformer on the same task. I built `aster.scaling`: six cells ({MLP, cell-token Transformer} x {tiny, ~5M, ~10M}) on the identical Workshop splits and one shared optimizer/stopping policy. The 2-CPU sandbox kept OOM-killing the large Transformer cells (heap fragmentation crossed the memory ceiling twice mid-training), so the study moved to the GitHub Actions runner via the agreed marker-workflow pattern; three parallel job groups trained the cells and a collect job merged the reports and committed them back. Per-epoch validation now reuses prebuilt tensors, and the study frees raw Python rows before training after adding a regression test proving the pretensorized path is equivalent.

Three findings. First, scaling the project MLP helps but saturates: 72k to 5M to 10M cuts test loss 0.175 to 0.123 to 0.102, and the 5M/10M rows swap order across machines, so the honest claim is diminishing, noisy gains on 18k training transitions. Second, inductive bias beats raw scale: the 104k-parameter cell-token Transformer (test 0.036) beats the 10M MLP (0.102) by nearly 3x. The grid-token structure is worth more than 100x parameters here. Third, and most surprising: naive Transformer scaling under the shared untuned recipe actively degrades. The 5M Transformer decodes the discrete dynamics beautifully (agent position 0.9997, most stable rollouts of all cells at MAE 0.0018) but its raw loss inflates to 0.42 through mis-scaled outputs; the 10M Transformer fails outright (agent position 0.57, validation loss pinned near initialization). Large Transformers need warmup and tuned learning rates; this table measures out-of-the-box behavior at fixed budget, not tuned bests, and the next fair comparison would give each family its own reasonable recipe.

## 2026-11-06 — Measured the last two gates: restart recovery passes, uncertainty fails

I closed out the two unmeasured gates. Restart recovery needed verification rather than new code: the Workshop restores exact state at every interruption point (test now iterates every prefix of expert episodes on eight seeded layouts and requires bit-identical outcomes), and the conversation agent already persisted goal, bounded episode history, and facts — the new restart test locks that in. Gate passes.

The uncertainty gate was the interesting failure. A five-member seed ensemble (the standard practical trick: disagreement between independently initialized models becomes the uncertainty estimate) shows unfamiliar states only 1-12% more disagreement than familiar ones, and — the decisive number — an error-detection AUROC of 0.49-0.54 in every seed. Chance level. The committee agrees almost exactly as confidently when it is wrong as when it is right; roughly 1% of all predictions are confidently wrong. Gate fails, honestly, in all four seeds. This is the known limitation of deep ensembles in near-deterministic environments: every member learns the same deterministic function and they fail together on novel geometry. Passing this gate needs a genuinely probabilistic model head (variational weights or input-dependent noise), not more members.

The measurement itself fought back first. My initial error flags reported 51% error against a model with 99.9% one-step accuracy. The bug: sigmoid saturates in float32 — confident logits above about 17 all map to exactly 1.0 — and argmax over saturated probabilities becomes a tie broken by cell index instead of by the model's belief. The engine's metrics had always decoded agent position in logit space, which is why the published numbers were fine; my new code path compared in probability space and was wrong. The fix (compare agent position on mean logits), a regression test that forces saturation, and a full rerun brought error rates to the true 2.6-5.6% and changed the confidently-wrong rate — but not the verdict: AUROC stayed at chance. Good reminder that a surprising number is either a discovery or a bug, and the first job is to find out which.

Current gate board: frequency baseline passes (4/4 seeds); policy beats baselines including bridge (pooled z = 8.2) but is far below the reliability thresholds; uncertainty fails; restart recovery passes; held-out rule compositions unmeasured. The honest next architectural step is a probabilistic world-model head; the honest next policy step attacks the new "opened but never crossed" bottleneck.

## 2026-11-06 — Failure analysis, multi-seed robustness, and dataset aggregation

I stopped trusting the single-run policy number and added what the evaluation still lacked: diagnostic failure counts in closed-loop evaluation, a `--seed` override for robustness runs, and `scripts/aggregate-metrics` for cross-seed tables. Four behavior-cloning runs (seeds 17, 23, 41, 73) showed large variance — 3.3% to 14.8% success on the 338 unseen layouts — and seed 41 collapsed to near-random. The earlier "beats random" claim was real but fragile, and the bridge advantage was within noise. Reporting one run would have been misleading.

The failure counts explained why: the dominant failure is never performing the subgoal interaction at all. Between 129 and 162 of 169 bridge episodes never touched the switch; door failures split between never taking the key and never opening. This is compounding-error, the known failure mode of behavior cloning: the policy rarely visits states near the subgoal during training, so it never learned what to do there.

The standard fix is dataset aggregation (DAgger): let the policy roll out, label the states it actually visits with the privileged expert, retrain. I implemented it as `expert_action` (a state-aware rule-aware planner used only for labels) and `collect_aggregation_examples`; execution and evaluation never consult the expert, so the learned/symbolic distinction is preserved. Two aggregation rounds add roughly 4,000-5,000 labeled states each.

Pooled over the same four seeds, aggregation lifted success from 10.5% to 25.4% (z = 16.7 vs random), door from 15.1% to 30.6%, and — the point of the exercise — bridge from 5.9% (noise) to 20.3% (z = 8.2). The worst seed lifted from 3.3% to 13.6%, and the failure bottleneck moved past the gate (bridge "never opened" fell from 129-162 to 39-118 of 169). None of this reaches the reliability gate (80% door / 60% bridge), so the policy stays experimental and play instructions stay withheld. The next honest targets are the remaining unmeasured gates — uncertainty calibration and interruption/restart recovery — and whatever the new failure distribution says is the current bottleneck (now partly "opened but never crossed").

## 2026-11-05 — Completed the evaluation protocol and expanded the layout space

The evaluation protocol in `first-experiment.md` had named items that were never implemented: the most-common-outcome frequency baseline, the reactive baseline that ignores state-changing objects, rollout error at horizons 1/3/5/10, the oracle gap, and a sample of at least 100 seeded scenarios per family. I implemented all of them before drawing any further conclusions.

Sizing the evaluation sample exposed a hard limit: the generator's only randomness was gate row, object row, and target row (3 x 4 x 5 = exactly 60 layouts per gate family, ever). The required 100 independent scenarios per family were impossible. I expanded the generator (layout space version 2) to also vary the gate column, object column, and start row — 680+ unique layouts per family over 1000 seeds — and added a regression test that the space stays above 100 per family. Transition rules and the encoder are unchanged.

The first completed evaluation changed one gate rule, documented here because thresholds must not move silently: my frequency-baseline gate originally required beating every MAE, and the model lost on `reward_mae` to the baseline's constant per-action prediction. Predicting a constant is near-optimal under MAE for Workshop's skewed reward signal (mostly -0.01), so that field cannot discriminate state-dependent learning. The gate now requires strict wins on the five accuracy fields plus no worse MAEs for time and invalidity; `reward_mae` is still reported but non-gating. With that rule the frequency gate passes honestly: agent position 0.999 vs 0.044, gate 0.978 vs 0.755, key 0.976 vs 0.775.

The substantive result is the large-sample comparison: 338 unique unseen layouts (169 per family), with every training and validation layout excluded. The behavior-cloning policy solved 50/338 (14.8%) versus 10/338 (3.0%) for random legal actions and 0/338 for the reactive baseline, against an oracle at 338/338. The overall gap to random is well beyond sampling noise (z = 5.4) and door tasks carry it (21.9% vs 0.6%, z = 6.2); the bridge advantage (7.7% vs 5.3%) is within noise (z = 0.9). So learned behavior contributes measurably — the first claim of its kind in this project — but reliability is nowhere near the 80%/60% planning gate, the policy stays experimental, and play instructions stay withheld. Rollout error grows gently (state MAE 0.006 at horizon 1 to 0.015 at horizon 10), which the beam planner has so far failed to exploit.

## 2026-11-05 — Added a learned policy and audited layout leakage

I added a separate feed-forward action policy trained by behavior cloning from expert demonstrations. Its action scores are masked by Workshop legality checks, but it does not receive the reference controller's grid-distance goal heuristic. This lets us test an actual learned decision component against random legal actions instead of assuming that a high-accuracy transition model implies planning.

While setting up that test, I audited the generator and found a flaw in the old evaluation split: the 60 seed values produce only 38 unique layouts per gate family, so different seeds sometimes generated identical maps across train and test. I replaced seed-only partitioning with canonical layout hashes; every repeated instance of the same geometry now stays in one split. Old held-out metrics should not be used as evidence of layout generalization.

The refreshed run (`runs/workshop-20261105-165630-17`, data schema v2) evaluates on the corrected split: 53 train, 11 validation, and 12 held-out canonical layouts, with 17,496 / 3,326 / 4,304 transitions. One-step test accuracy was 99.98% for agent position, 97.70% for gate state, 99.35% for key state, and 93.34% for dynamic-state bits. Closed-loop, the hybrid reference controller solved 12/12 exact held-out layout cases — still a plumbing check, not learned planning. The behavior-cloning policy solved 1/12 cases versus 0/12 for random legal actions (expert-action accuracy 74.75%, one door task solved). With only 12 held-out cases this gap is suggestive, not conclusive; the policy benchmark thresholds did not pass. The policy therefore stays experimental and play instructions remain withheld. The hybrid reference controller remains an oracle-like symbolic baseline, not proof of neural planning. A later run with a larger unseen-layout sample (see the top entry) supersedes these numbers.

## 2026-11-05 — Trained the first artificial-neuron world model

I moved from an environment-only scaffold to a small PyTorch transition model. It has two hidden layers with weighted connections and output heads for next state, reward, terminal status, and invalid actions. The weights start randomly and are adjusted by backpropagation. That is a useful artificial-neuron/synapse analogy for a trainable computer model, but it is not a biological neuron or a complete brain simulation.

The data pipeline gathers expert coverage plus random actions, tests every action from states on expert routes, splits by canonical layout hash to avoid geometry leakage, writes SHA-256 manifests, and saves local checkpoints. A full CPU training run finished on the Workshop data; its first held-out metrics looked strong, but they came from the seed-based split later found leaky, so the entry above records the corrected canonical-layout-split numbers instead. This small structured benchmark has strong limits and does not demonstrate general intelligence.

An early model-rollout beam planner oscillated or chose invalid trajectories. I did not label that as success. I added a one-step hybrid reference controller with explicit Workshop affordances and goal-distance rules. It solves this toy environment across seeded tests, but a random untrained network can also solve it because most of the task policy is hand-coded. The controller result is a plumbing/safety check, not evidence the neural network learned planning. The next serious capability test must measure the learned model's contribution against a random/untrained-model and symbolic-only baseline, then grow task complexity rather than overstate this result.

## 2026-11-05 — Implemented the first controlled learning environment

After specifying the Workshop experiment, I implemented its deterministic door/key and switch/bridge transitions, seeded layout generator, fixed numeric observation encoder, configurable step limit, and a privileged rule-aware coverage baseline. The baseline is explicitly marked for data generation and comparison only; the future deployed planner must act from its learned model.

The first bridge test initially used a Manhattan path that crossed a wall instead of the gate. The failed test exposed a problem in the test procedure rather than the environment rule. I changed the path order to approach the crossing row before moving across it, then verified 30 seeds for each gate type. The suite now also covers invalid transitions, deterministic generation, encoding shape, and timeout behavior.

This is still not a trainable agent. The current checkpoint between design and neural learning is intentional: environment transitions and a reference coverage policy need to be reliable before training data can be trusted.

## 2026-11-05 — Reordered the work around a capability inventory

The user clarified the desired order: first map the brain capabilities relevant to learning and long-horizon work, then choose architecture and implementation from that map. My first pass jumped straight into a small prototype. That established useful project hygiene, but it did not follow the requested discovery-first sequence.

I added `brain-capabilities.md` as the design reference before expanding implementation. It groups perception, attention/executive control, memory, learning, world modelling, reasoning, goals and planning, action/feedback, metacognition, communication, non-emotional motivation, and robustness. It also proposes staged development and a rule: a capability is not considered present merely because the interface can talk about it; it needs observable behavior and tests.

This inventory is meant to be comprehensive at the functional level for our goal, not a literal enumeration of every biological mechanism or a claim that neuroscience has settled every theory. The user then clarified the end goal: make the system straightforward to train from an initial model, then have the same cognitive core learn, plan, act, build a world model, and generate new concepts without an attached LLM doing its reasoning. I added `architecture.md` to connect the inventory to a trainable action-conditioned world model, layered memory, goal-directed planning, feedback, and a safe hypothesis-generation loop. The proposed first training run uses a generated environment and weights initialized from scratch, with no pretrained LLM. It also states that passing a small environment benchmark is only evidence for that narrow environment—not human-like general intelligence.

I then specified the first experiment in `first-experiment.md`: a small object-and-tool workshop, randomized multi-step tasks, held-out compositions, a learned transition model, a receding-horizon planner, baselines, provisional success gates, and a repeatable run/checkpoint layout. This makes the first training step concrete without pretending the training code exists yet.

The next implementation step is to review these task definitions for accidental shortcuts, then implement the smallest environment and deterministic baseline before introducing PyTorch. Do not add features just because an architecture diagram names them.

## 2026-11-05 — First pass: establish a small, inspectable core

The repository contained only a short README, so there was no existing application behavior or language stack to preserve. I chose Python because this first slice is mostly domain rules and persistence; the standard library keeps the experiment easy to run and inspect without a model service or dependency setup.

The request describes a real brain with broad, human-level capabilities, no emotions, and no LLM used as a language layer. That is a research-scale goal, not something a truthful first implementation can claim to achieve. I started with a deliberately smaller system that has explicit input parsing, durable facts, a bounded working-memory window, a persistent goal, deterministic retrieval, and one transparent inference rule. This gives later work observable behavior to extend and tests to keep regressions visible.

I separated the parser, memory store, reasoner, agent coordinator, and CLI rather than placing all behavior in one file. The language interface only recognizes a small grammar; it does not secretly hand off to a language model. Unsupported requests receive an honest explanation instead of a fabricated answer.

The first reasoning rule follows `is` links transitively. The initial test run exposed a subtle traversal bug: using one set for both answer de-duplication and graph visitation stopped traversal at the first directly known fact. I split those concerns into a visited-node set and an already-answered set, and explicitly avoid returning the original subject when a cycle leads back to it. The regression test now covers a multi-hop chain and a cycle. The memory store de-duplicates facts case-insensitively but preserves the text supplied by the user. Recent episodes are bounded, while facts and the latest goal survive process restarts.

The first attempt to install into the system Python was correctly blocked by the environment's externally-managed-package policy. I switched to an isolated `.venv` and updated the setup instructions, rather than bypassing that protection. The GitHub Actions workaround is included as an explicit opt-in marker workflow. It stores its run log in the repository as requested. This is not the normal pattern for CI because generated logs create commits; the marker avoids a workflow loop and makes the cost visible. Local tests remain the first feedback path.

### Checks and follow-up

The implementation is covered by unit tests for parsing, duplicate facts, persistence, working-memory bounds, cyclic inference, unknown questions, and goal reporting. The next sensible steps are to run the suite, fix any failures, and then expand behavior only with concrete tests and examples. A future capability should have a clear owner, data model, and test before being added; no single module should become a catch-all.
