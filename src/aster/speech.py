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


def form_sentence(net: PlasticNetwork, seed_words: list[str], max_words: int = 18) -> list[str]:
    """Grow a sentence from seed words by walking the learned word order."""
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
        seeds = [ACK_NEW, ACK_OLD, UNKNOWN]
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
            return render(form_sentence(self.net, start))
        if parsed.kind is InputKind.QUESTION and parsed.subject and parsed.relation:
            answers = self.net.retrieve(parsed.subject, parsed.relation)
            if answers:
                return render(self._answer_sentence(parsed.subject, answers[0][0]))
            return render(form_sentence(self.net, sequence_words(UNKNOWN)[:1]))
        return render(self._talk(text))

    def _answer_sentence(self, subject: str, answer: str) -> list[str]:
        subject_words = sequence_words(subject)
        answer_words = sequence_words(answer)
        formed = form_sentence(self.net, subject_words)
        if set(answer_words) & set(formed):
            return formed
        from_answer = form_sentence(self.net, answer_words)
        return from_answer if len(from_answer) > len(formed) else formed

    def _talk(self, text: str) -> list[str]:
        words = sequence_words(text)
        content = [word for word in words if word not in STOPWORDS] or words
        best: list[str] = []
        for word in content:
            formed = form_sentence(self.net, [word])
            if len(formed) > len(best):
                best = formed
            if len(best) >= 6:
                break
        return best if best else form_sentence(self.net, sequence_words(UNKNOWN)[:1])


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
