# Brain-like plasticity: the design we are building toward

## The vision

The system should behave like brain tissue, not like a reference book:

- It is made of **neurons and synapses**. Knowledge lives in the strengths of
  pathways between neurons, not in a dictionary that gets looked up.
- **Telling it things is learning.** There is no separate "training mode" for
  knowledge: every conversation nudges the pathways (weights) immediately.
- **Use strengthens, disuse weakens.** The more a pathway fires, the stronger
  it becomes; pathways that stop firing fade and eventually disappear, like
  real synapses.
- **No context-length limit, no padding.** Input is a stream of variable
  length. Nothing is padded or truncated to fit a fixed window; the network
  state itself is the accumulated context.
- **Knowledge changes the structure.** New concepts recruit new neurons;
  forgotten ones are pruned. The tissue physically grows and shrinks.
- **Answers form by firing, not lookup.** A question lights up neurons and
  activity spreads along learned pathways until the answer neurons win.
  Sentences and logic should eventually form the same way.

## Stage 1 (built): the plastic substrate

`src/aster/cognition/plastic.py` implements `PlasticNetwork`:

- **Neurons and ensembles.** Concepts live in small ensembles of neurons.
  New concepts recruit new neurons (growth); long disuse prunes them.
- **Gated synapses.** A fact `Ada is a mathematician` strengthens directed
  pathways from the `Ada` ensemble to the `a mathematician` ensemble, gated
  by the `is` ensemble (so `is` and `has` knowledge never cross-talk).
  Strengths are the only store: there is no fact list anywhere.
- **Perception without a dictionary.** Words and phrases become sets of
  spelling features (letters, bigrams, prefixes) -- perception, not lookup.
  Existing concepts are recognized by feature overlap; overlap below the
  identity threshold recruits a new ensemble instead.
- **Telling = Hebbian learning.** Co-active ensembles strengthen their
  pathways immediately (one telling already counts as "known"; repetition
  makes it stronger).
- **Asking = spreading activation.** The cue's neurons fire and activity
  travels along gated pathways for a few hops; whichever ensembles fire
  above threshold win the answer slots, in hop order. Transitive `is`
  answers emerge from two hops; cycles cannot loop because the cue never
  answers itself and activity fades with distance.
- **Use-dependent plasticity.** Retrieval re-strengthens the pathways it
  used. Idle time decays every strength a little and prunes what falls away.
  Learning (per telling) is much larger than decay (per moment), so used
  knowledge grows while neglected knowledge fades -- the requested rule.
- **Persistence is consolidation.** Saving stores the synaptic strengths
  themselves; loading restores them. Old version-1 fact files are replayed
  through the network on load (like re-telling).

The conversation brain (`Aster`) now runs on this substrate for all facts.
Goals and the small recent-conversation window remain a plain notepad
(short-term buffer); everything learnable is synaptic.

## Stage 1.5 (built): how the substrate ages -- consolidation, savings, capacity

Four more brain properties live in the same substrate now.

- **Consolidation.** The forgetting curve depends on how strong a pathway
  already is: `w *= (1 - decay * (1 - 0.8 * w))`. A half-learned trace
  fades about ten times faster than a well-used skill. Trivia you heard
  once decays quickly; things you keep using barely notice time. Imprints
  (the recognizer) fade at a tenth of that rate, so a forgotten fact first
  stops being answerable while still feeling familiar -- the tip-of-the-
  tongue state -- before it is pruned entirely.
- **Savings (re-learning).** Every pathway remembers its peak strength.
  Re-strengthening runs at `learning_rate * (1 + 0.6 * peak)`: bringing
  back a half-forgotten skill with one telling jumps it to 0.85 where a
  first telling only reaches 0.6. Skills may fade, but they never really
  start over.
- **Capacity.** The network can be given a neuron budget. When teaching
  would exceed it, the weakest idle ensemble gives up its neurons
  (aggregate of its synapse weights plus imprints; ties go to the oldest).
  Whatever is in use at that moment is never evicted, and what you keep
  practicing stays heavier than untouched material -- so under pressure
  the used skill survives and the abandoned one goes. Larger budgets hold
  more skills at once.
- **Pattern separation.** A cue must pick ONE memory clearly to count as
  recognition; a tie between two look-alike memories is treated as
  ambiguity, not as either answer. This is why "item9" cannot silently be
  answered by "item0" once item9 is gone.

Selective forgetting falls out of consolidation: after long idling,
one-off details are unanswerable while practiced knowledge still answers
(`test_selective_forgetting_keeps_skills_loses_trivia`). Practice over a
long engagement keeps quality high instead of degrading -- pathways only
get stronger with use, and there is no context window to overflow
(`test_skills_get_better_with_practice_over_long_engagement`).

## The world model learns like this too (experiments started)

`environment/physics.py` is a tiny deterministic world (a held object in a
yard with walls and a floor). The network is never *told* about gravity:
it watches trials -- "letting go" followed by "falling, landing" -- until
the causal pathways form from repetition. Then the name can arrive last:
telling it "that is called gravity" binds the word to the concept the
experience already built, and the name and the experience answer as the
same thing. The same push in an open yard and against a wall teaches two
different outcomes -- the knowledge reflects the world, not a prior
(`tests/test_physics_grounding.py`). This is the "discovering it is the
ability" idea in runnable form.

## Requirement to proof map

| Requirement | Proof |
|---|---|
| Knowledge is pathways, not a dictionary | `test_retrieval_is_associative_not_string_lookup` |
| Telling learns immediately | `test_telling_learns_immediately_and_repetition_is_recognized` |
| Use strengthens, disuse weakens | `test_use_strengthens_paths_and_disuse_weakens_them` |
| Forgetting and pruning like a brain | `test_long_disuse_forgets_and_prunes_like_a_brain` |
| Selective forgetting (trivia goes, skills stay) | `test_selective_forgetting_keeps_skills_loses_trivia` |
| Re-learning is faster than first learning | `test_faded_knowledge_relearns_faster_than_first_learning` |
| Capacity competition under a neuron budget | `test_capacity_competition_evicts_the_weakest` |
| Practice improves, never degrades, over long work | `test_skills_get_better_with_practice_over_long_engagement` |
| Causal knowledge from experience, not telling | `test_falling_is_learned_from_experience_not_telling` |
| World knowledge reflects the world, not a prior | `test_the_world_not_a_prior_decides_what_is_learned` |
| Discovery first, naming after (grounded names) | `test_discovery_precedes_naming_and_the_name_is_grounded` |
| Structural growth | `test_network_physically_grows_when_told_new_things` |
| No padding, no context window | `test_variable_length_stream_needs_no_padding_or_window` |
| Logic by firing (transitive, cyclic) | `test_transitive_chains_resolve_and_cycles_do_not_loop` |
| Memory survives restart as strengths | `test_snapshot_round_trips_the_learned_strengths` |

## Honest limits

- "No context limit" means no hard window or truncation. Real brains still
  forget and interfere, and so does this: unused pathways decay. That is a
  property of the vision (use-dependence), not a hidden limit. Trivia
  fades while practiced skills persist -- the intended selectivity.
- The capacity budget is a knob the operator sets; the brain's capacity is
  not known in neurons, and real consolidation also *reorganizes*
  knowledge (sleep replay) rather than only protecting it. Here strong
  pathways are protected by their own strength only.
- The physics world is a toy: four actions, one object, no objects
  colliding. It demonstrates the discovery mechanism honestly, not the
  richness of a real world model.
- Sentence *content* forms by firing, but the grammatical *frame* of replies
  is still a fixed template, and the parser that hears sentences is still
  written code. Stage 2 replaces both with settling dynamics over word
  pools ("neurons fire forming sentences").
- The Workshop world model still learns in offline batches from recorded
  experience. Stage 3 gives it the same continual, use-dependent plasticity
  (visited state-action pathways strengthen during play, idle ones fade).

## Next stages

1. **Stage 2 -- sentence formation.** Word pools with winner-take-all
   settling; the reply sentence emerges from which word neurons stay active.
2. **Stage 3 -- plastic world model.** Continual nudging during use plus
   Hebbian strengthening of frequently used state-action pathways; episodes
   are variable-length event streams with no padding. The physics
   playground experiments above are the seed of this stage.
3. **Stage 4 -- telling things about the world.** Verbal instruction nudges
   world-model pathways directly ("the door opens when you use the key"
   strengthens that pathway before ever acting).
4. **Days-to-years engagement.** Skill quality across stretches of
   thousands of ticks with interleaved practice (the capacity and
   consolidation laws are built for this; the long curve study is not run
   yet).
