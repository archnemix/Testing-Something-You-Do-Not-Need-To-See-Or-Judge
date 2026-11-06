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

## Requirement to proof map

| Requirement | Proof |
|---|---|
| Knowledge is pathways, not a dictionary | `test_retrieval_is_associative_not_string_lookup` |
| Telling learns immediately | `test_telling_learns_immediately_and_repetition_is_recognized` |
| Use strengthens, disuse weakens | `test_use_strengthens_paths_and_disuse_weakens_them` |
| Forgetting and pruning like a brain | `test_long_disuse_forgets_and_prunes_like_a_brain` |
| Structural growth | `test_network_physically_grows_when_told_new_things` |
| No padding, no context window | `test_variable_length_stream_needs_no_padding_or_window` |
| Logic by firing (transitive, cyclic) | `test_transitive_chains_resolve_and_cycles_do_not_loop` |
| Memory survives restart as strengths | `test_snapshot_round_trips_the_learned_strengths` |

## Honest limits

- "No context limit" means no hard window or truncation. Real brains still
  forget and interfere, and so does this: unused pathways decay. That is a
  property of the vision (use-dependence), not a hidden limit.
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
   are variable-length event streams with no padding.
3. **Stage 4 -- telling things about the world.** Verbal instruction nudges
   world-model pathways directly ("the door opens when you use the key"
   strengthens that pathway before ever acting).
