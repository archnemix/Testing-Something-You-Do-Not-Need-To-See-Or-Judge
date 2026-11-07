"""Native speech: sentences formed by word neurons firing along learned order.

No template anywhere. Word order lives in the same synapses as everything
else (word A "comes after" word B), and a sentence is formed by letting
activation walk those links: at every step the strongest next word wins
(winner-take-all) and stays active -- the sentence IS the sequence of
words that fire. Repetition makes a path heavier and therefore more
likely to be spoken; unused word order decays and falls out of speech.

`Speaker` is who you talk to: it teaches from what you say, answers
questions with formed sentences, and for anything else it simply talks --
generating from the words you used.
"""

from __future__ import annotations

from aster.case_study import ENGLISH, MUSIC, PYTHON
from aster.cognition.plastic import PlasticNetwork, sequence_words
from aster.language import parse_input
from aster.models import InputKind

ACK_NEW = "okay i will remember that"
ACK_OLD = "yes i already know that"
UNKNOWN = "hmm i do not know"

STOPWORDS = {
    "a", "an", "and", "about", "are", "can", "did", "do", "does", "for",
    "has", "have", "how", "is", "it", "me", "mean", "means", "of", "on",
    "please", "tell", "the", "to", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}


def form_sentence(
    net: PlasticNetwork,
    seed_words: list[str],
    max_words: int = 18,
    practice: bool = False,
) -> list[str]:
    """Grow a sentence from seed words by walking the learned word order.

    With ``practice=True`` the sentence that is spoken strengthens its own
    pathways as it goes -- saying something is using it, and use nudges the
    weights. Nothing new is stored by speaking; the paths that fired grow.
    """
    words = list(seed_words)
    if not words:
        return []
    seen: dict[str, int] = {}
    while len(words) < max_words:
        seen[words[-1]] = seen.get(words[-1], 0) + 1
        if seen[words[-1]] > 2:
            break  # the pathway is looping; stop
        following = [
            label for label, _ in net.successors(words[-1], tuple(words[:-1]))
        ]
        picked = None
        for label in following:
            if label != words[-1] and (len(words) < 2 or label != words[-2]):
                picked = label
                break
        if picked is None:
            break
        words.append(picked)
    if practice:
        net.practice_sentence(" ".join(words))
    return words


def render(words: list[str]) -> str:
    if not words:
        return ""
    text = " ".join(words)
    return text[0].upper() + text[1:] + "."


class Speaker:
    """Talks: learns from statements, answers questions, chats freely."""

    def __init__(self, net: PlasticNetwork | None = None, warm: bool = True) -> None:
        self.net = net if net is not None else PlasticNetwork()
        if warm:
            self.warmup()

    def warmup(self) -> None:
        """Teach the conversation basics: curricula sentences and replies."""
        seeds = [
            ACK_NEW,
            ACK_OLD,
            UNKNOWN,
            "i know words and what they mean",
            "i am a network of neurons that learns from talking",
        ]
        for skill in (ENGLISH, PYTHON, MUSIC):
            for lesson in skill.lessons:
                seeds.append(lesson.text)
        self.net.observe_corpus(seeds * 3)
        for skill in (ENGLISH, PYTHON, MUSIC):
            for lesson in skill.lessons:
                parsed = parse_input(lesson.text)
                if parsed.kind is InputKind.FACT and parsed.fact is not None:
                    fact = parsed.fact
                    self.net.observe(fact.subject, fact.relation, fact.object)

    def respond(self, text: str) -> str:
        parsed = parse_input(text)
        if parsed.kind is InputKind.FACT and parsed.fact is not None:
            fact = parsed.fact
            is_new = self.net.observe(fact.subject, fact.relation, fact.object)
            self.net.observe_sequence(text)
            start = sequence_words(ACK_NEW if is_new else ACK_OLD)[:1]
            return render(form_sentence(self.net, start, max_words=5, practice=True))
        if parsed.kind is InputKind.QUESTION and parsed.subject and parsed.relation:
            # asking nudges what is recalled (use-dependent consolidation)
            answers = self.net.retrieve(parsed.subject, parsed.relation)
            formulated = self._formulate(parsed.subject, parsed.relation, answers)
            if formulated:
                return render(formulated)
            return render(form_sentence(self.net, sequence_words(UNKNOWN)[:1], max_words=5))
        return render(self._talk(text))

    def _formulate(self, subject, relation, answers) -> list[str]:
        """Answer the question asked: start from ITS words, complete with knowledge.

        'what is python' and 'what does python mean' seed different paths,
        so they get different sentences -- no replaying one stored answer.
        """
        seed = sequence_words(subject) + [relation]
        formed = form_sentence(self.net, seed, practice=True)
        if len(formed) > len(seed):
            return formed
        for label, _ in answers:  # subject path is silent; speak the recalled words
            from_answer = form_sentence(self.net, sequence_words(label), practice=True)
            if len(from_answer) > 1:
                return from_answer
        return []

    def _talk(self, text: str) -> list[str]:
        words = sequence_words(text)
        content = [word for word in words if word not in STOPWORDS] or words

        def known(word: str) -> tuple[int, float]:
            following = self.net.successors(word)
            weight = following[0][1] if following else 0.0
            return (1 if following else 0), weight

        content.sort(key=known, reverse=True)  # talk about what it knows
        best: list[str] = []
        for word in content:
            formed = form_sentence(self.net, [word], practice=True)
            if len(formed) > len(best):
                best = formed
            if len(best) >= 6:
                break
        return best if best else form_sentence(self.net, sequence_words(UNKNOWN)[:1], max_words=5)


def main() -> None:
    speaker = Speaker()
    print("Talk to it. (Ctrl-D to leave.)")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text or text in {"quit", "exit"}:
            break
        print(f"it>  {speaker.respond(text)}")


if __name__ == "__main__":
    main()
