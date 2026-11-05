# Cognitive capability inventory

See [architecture.md](architecture.md) for the system design and [first-experiment.md](first-experiment.md) for the first proposed training benchmark.

This is the capability map for Aster's design. It focuses on functions that support learning and sustained, multi-step work. It is a functional inventory, not a claim that every brain region, cellular mechanism, or debated theory is settled. For each capability we need to define observable behavior and tests before implementing it.

## 1. Perception and grounding

A cognitive system needs inputs that refer to something beyond its own text. The human brain integrates vision, hearing, touch, body state, and other sensory signals; it filters noise and uses context to interpret incomplete observations.

- Ingest observations from distinct modalities and preserve their source, time, and reliability.
- Bind features that belong to the same object or event; maintain identity when an object changes or disappears.
- Represent spatial, temporal, and causal context, not only word sequences.
- Use active perception: select a question, tool call, or observation that can reduce uncertainty.
- Detect missing, conflicting, stale, or out-of-distribution input.

**Long-horizon value:** goals need to stay connected to changing conditions and evidence. A plan must be revisable when the world differs from its assumptions.

## 2. Attention and executive control

Attention allocates limited processing capacity. Executive control helps keep behavior aligned with a goal when distractions, competing tasks, or tempting immediate actions appear.

- Select relevant input using goals, salience, context, and expected value.
- Sustain attention across a task and switch deliberately when priorities change.
- Maintain and update working context; remove obsolete or superseded items.
- Inhibit irrelevant responses and detect when a habitual action is inappropriate.
- Sequence operations, manage a limited compute/time budget, and resume interrupted work.
- Track dependencies and protect the current task from unrelated requests without ignoring urgent evidence.

**Long-horizon value:** supports focus, interruption recovery, and sensible allocation of limited resources.

## 3. Memory systems

Memory is not a single store. A useful design distinguishes what is being retained, how it was learned, and how reliable or current it is.

- **Working memory:** small, rapidly updated context for the current step and active subgoals.
- **Episodic memory:** particular experiences with time, context, action, and outcome.
- **Semantic memory:** concepts, facts, categories, and relations that generalize across episodes.
- **Procedural memory:** learned skills and reusable ways of performing tasks.
- **Prospective memory:** intentions that must be remembered and acted on later or when a condition occurs.
- **Autobiographical/self-model memory:** history of the system's own actions, commitments, capabilities, and limitations.
- **Source and confidence tracking:** where a belief came from, whether it was observed, inferred, or told, and how certain it is.
- **Consolidation and replay:** strengthen useful knowledge by revisiting experiences and extracting stable patterns.
- **Retrieval and cueing:** retrieve by meaning, context, time, entity, goal, and associations, not only exact wording.
- **Forgetting and revision:** decay or archive low-value details, merge duplicates carefully, and correct beliefs when stronger evidence arrives.

**Long-horizon value:** lets a system resume work, avoid repeating failed approaches, and use earlier learning on new tasks.

## 4. Learning and adaptation

Learning includes more than adding facts. It changes what the system can predict, represent, and do.

- Learn from explicit instruction, examples, observation, action outcomes, and feedback.
- Acquire concepts and relations; distinguish a specific example from a general rule.
- Generalize to new cases while retaining uncertainty and exceptions.
- Transfer relevant methods between tasks without assuming superficial similarity is enough.
- Learn causal structure and action consequences, not just co-occurrence.
- Improve procedural skill through practice and error feedback.
- Learn from other agents while tracking trust, source, and possible disagreement.
- Detect distribution change and decide when an old rule should be rechecked.
- Balance exploration (gather useful new evidence) with exploitation (apply reliable knowledge).
- Support incremental updates without erasing older, still-valid knowledge.

**Long-horizon value:** progress accumulates across tasks instead of resetting after each conversation.

## 5. Representation and world modelling

Reasoning requires internal structures that preserve the distinctions needed to act correctly.

- Represent entities, attributes, relations, categories, events, and changing state.
- Handle compositional structure: combine known concepts into a representation of a new situation.
- Maintain temporal order, duration, recurrence, and deadlines.
- Distinguish observed state from predicted state, hypothetical state, and desired state.
- Represent constraints, dependencies, resources, affordances, and uncertainty.
- Keep a self-model of available tools, skills, limits, and current commitments.
- Support multiple levels of abstraction, from concrete episodes to reusable schemas.

**Long-horizon value:** enables plans and knowledge to be composed, compared, and checked for consistency.

## 6. Reasoning and decision-making

A capable system should select a reasoning method suited to the question, rather than relying on one association rule.

- Deduction: derive consequences that follow from explicit rules and premises.
- Induction: form cautious generalizations from repeated observations.
- Abduction: compare explanations for an observation and seek discriminating evidence.
- Analogy: map relational structure between situations and check where the mapping fails.
- Causal reasoning: predict intervention effects and distinguish causes from correlations.
- Counterfactual reasoning: compare actual outcomes with plausible alternatives.
- Temporal and spatial reasoning: evaluate sequences, durations, locations, and motion.
- Quantitative reasoning: calculate, estimate, and check units and magnitudes.
- Belief revision: resolve inconsistency using evidence quality, source, time, and explicit rules.
- Uncertainty calibration: represent confidence, alternatives, and unknowns; avoid presenting inference as observation.
- Decision-making: compare expected outcomes, cost, risk, reversibility, and value of information.

**Long-horizon value:** supports choices under incomplete information and permits plans to improve as evidence arrives.

## 7. Goals, planning, and persistence

Long-horizon work requires a goal system and an execution loop, not merely a goal written in memory.

- Translate a request into an explicit desired outcome and success conditions.
- Check feasibility, ambiguity, safety constraints, resources, and authority before acting.
- Decompose a goal into ordered and parallel subgoals with prerequisites.
- Estimate effort, time, uncertainty, and value; prioritize among competing commitments.
- Maintain plans and intentions over time, including deadlines and trigger conditions.
- Track progress with evidence, not optimistic status text.
- Detect blocked, stale, failed, or completed subgoals; replan when assumptions change.
- Preserve useful partial work and resume after interruptions or restarts.
- Verify the final outcome against the original success conditions before reporting completion.
- Stop, ask for clarification, or seek missing evidence when continuing would be unreliable.

**Long-horizon value:** turns intent into measured, recoverable progress and guards against confidently claiming unfinished work.

## 8. Action, feedback, and tool use

Cognition is coupled to acting and observing consequences. For a software agent, tools and APIs play the role of limited action channels.

- Select an action that can advance the current subgoal.
- Check preconditions, permissions, reversibility, and potential side effects.
- Execute with bounded scope and retain the exact action/result record.
- Compare predicted and observed outcomes; detect errors and unintended effects.
- Retry only when safe and informative; otherwise diagnose, roll back, or ask.
- Use tools to gather evidence or change state, rather than simulating tool results in text.

**Long-horizon value:** enables the system to make actual progress and learn from outcomes instead of only describing plans.

## 9. Metacognition and self-correction

A system needs ways to monitor its own processing and know when its answer or plan may be unreliable.

- Estimate confidence and identify which premise or step is uncertain.
- Check consistency, calculations, citations/sources, and plan preconditions.
- Detect errors, contradictions, unexpected outcomes, and repeated failed strategies.
- Choose whether to answer, reason further, retrieve memory, use a tool, or ask a question.
- Estimate when to stop searching and what evidence would change the conclusion.
- Maintain an auditable trail from evidence through inference to action.

**Long-horizon value:** catches drift and prevents small unsupported assumptions from compounding through a long task.

## 10. Communication and shared context

Language is one input/output channel, not the reasoning engine itself.

- Map an utterance to intent, entities, constraints, and uncertainty.
- Ask focused clarifying questions when interpretations differ materially.
- Report conclusions with the evidence and limits relevant to the user.
- Distinguish plans, attempts, observed results, and verified completion.
- Preserve commitments and shared context across turns without pretending to remember what was not stored.

The implementation must keep cognition and presentation separate. Language generation can be deterministic and limited at first; it must not fabricate internal conclusions or conceal missing reasoning.

## 11. Motivation without emotion

For this project, motivation means goal-directed control rather than simulated feelings.

- Maintain explicit user-authorized goals and their priorities.
- Persist effort by tracking unfinished subgoals and resuming them when appropriate.
- Prefer actions with measurable progress and useful information gain.
- Represent urgency, cost, and task value as explicit control variables.
- Respect stop conditions, permissions, and resource limits; persistence is not permission to act indefinitely.

Aster will not add emotional experience or claim to have it. Functional persistence and prioritization can be implemented and measured without assigning feelings to the system.

## 12. State regulation and robustness

The brain is an adaptive, resource-limited system. A software analogue should make its operating limits explicit.

- Manage time, memory, compute, and action budgets.
- Handle interruptions, partial failures, stale state, and restart recovery.
- Protect durable knowledge from accidental corruption and preserve provenance.
- Degrade safely when a tool, modality, or memory store is unavailable.
- Keep operations observable and testable; make state changes reviewable.

## Capability-to-build sequence

The sequence below is a proposed order, not a claim that later capabilities are solved by earlier ones. Each phase needs a behavior specification, subsystem boundary, tests, and review before implementation.

1. **Specification and evaluation:** define a capability matrix and long-horizon benchmark tasks; decide what evidence counts as success and what the system must never claim.
2. **Structured knowledge and provenance:** typed observations, source/confidence, timestamps, contradiction handling, and explicit belief revision.
3. **Task state and executive control:** goal trees, dependencies, priority, working context, interruption/resume, budgets, and progress checks.
4. **Planning and action loop:** plan, execute one bounded action, observe result, compare prediction, update plan, verify completion.
5. **Learning and memory consolidation:** episodes to reusable knowledge, retrieval by context/meaning, feedback, correction, safe generalization, and forgetting.
6. **Reasoning and metacognition:** causal, temporal, quantitative, counterfactual, uncertainty-aware reasoning with traceable evidence.
7. **Perception and tool grounding:** add input modalities and tools as separate adapters with reliability and permission boundaries.
8. **Long-horizon evaluation:** run tasks across interruptions and restarts; measure completion, regressions, unsupported claims, recovery, and resource use.

Do not move to a later phase just because an interface can describe it. The capability is present only when its behavior is implemented and its tests demonstrate it.
