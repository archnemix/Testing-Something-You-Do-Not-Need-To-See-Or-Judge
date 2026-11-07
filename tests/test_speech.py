import unittest

from aster.cognition.plastic import PlasticNetwork
from aster.speech import Speaker, form_sentence, render


class SpeechTests(unittest.TestCase):
    """Sentences are words firing along learned order -- no template."""

    def test_a_sentence_forms_from_word_neurons(self) -> None:
        net = PlasticNetwork()
        net.observe_sequence("hello there friend")
        self.assertEqual(
            form_sentence(net, ["hello"]), ["hello", "there", "friend"]
        )
        self.assertEqual(render(["hello", "there", "friend"]), "Hello there friend.")

    def test_generation_recombines_sentences_never_taught(self) -> None:
        net = PlasticNetwork()
        net.observe_corpus(["the cat sat on the mat"] * 3)
        net.observe_corpus(["the dog sat on the rug"])
        # the heavier "the -> mat" path wins, producing a sentence that
        # was never taught in full: "the dog sat on the mat"
        self.assertEqual(
            form_sentence(net, ["the", "dog", "sat", "on", "the"]),
            ["the", "dog", "sat", "on", "the", "mat"],
        )

    def test_winner_take_all_follows_the_heavier_path(self) -> None:
        net = PlasticNetwork()
        net.observe_corpus(["the bird sings"])
        net.observe_corpus(["the bird flies"] * 3)
        self.assertEqual(
            form_sentence(net, ["the", "bird"]), ["the", "bird", "flies"]
        )

    def test_output_changes_when_the_pathways_decay(self) -> None:
        net = PlasticNetwork()
        net.observe_sequence("fresh bread tastes great")
        before = form_sentence(net, ["fresh"])
        self.assertEqual(before, ["fresh", "bread", "tastes", "great"])
        net.tick(25000)
        after = form_sentence(net, ["fresh"])
        self.assertNotEqual(before, after)  # words come from weights, not a mold

    def test_speaker_answers_talks_and_teaches_natively(self) -> None:
        speaker = Speaker(warm=True)
        reply = speaker.respond("what does print mean?")
        self.assertIn("showing text", reply)
        self.assertEqual(speaker.respond("hello"), "Hello is a greeting.")
        reply = speaker.respond("the sky is blue")
        self.assertIn("remember", reply.lower())
        self.assertIn("Okay", reply)  # a formed acknowledgment, not a template
        self.assertIn("python", speaker.respond("tell me about python").lower())
        self.assertNotIn("Remembered:", speaker.respond("the moon is bright"))

    def test_corpus_teaching_scales_past_a_thousand_sentences(self) -> None:
        net = PlasticNetwork()
        sentences = [f"the word{n} comes after word{n + 1}" for n in range(1000)]
        wired = net.observe_corpus(sentences)
        self.assertGreater(wired, 900)
        formed = form_sentence(net, ["word0"])
        self.assertEqual(formed[:3], ["word0", "comes", "after"])

    def test_different_questions_get_different_formulated_answers(self) -> None:
        """The user's complaint, as a test: no stored replay regardless of question."""
        speaker = Speaker(warm=True)
        speaker.net.observe("python", "means", "a large heavy-bodied constrictor snake")
        speaker.net.observe_sequence("python means a large heavy-bodied constrictor snake")
        what_is = speaker.respond("what is python")
        what_means = speaker.respond("what does python mean")
        verify = speaker.respond("is python a fruit")
        self.assertIn("programming language", what_is)
        self.assertIn("constrictor", what_means)
        self.assertNotEqual(what_is, what_means)  # formulated per question
        self.assertNotIn("fruit", verify)  # answers from knowledge, not echo

    def test_speaking_practices_the_pathway(self) -> None:
        net = PlasticNetwork()
        net.observe_sequence("quiet rivers flow slowly")
        before = net.successors("quiet")[0][1]
        neurons_before = net.neuron_count
        form_sentence(net, ["quiet"], practice=True)  # it says the sentence
        after = net.successors("quiet")[0][1]
        self.assertGreater(after, before)  # saying it nudged the pathway up
        self.assertEqual(net.neuron_count, neurons_before)  # speaking added nothing

    def test_asking_nudges_what_it_recalls(self) -> None:
        speaker = Speaker(warm=True)
        speaker.net.observe("saffron", "means", "a golden spice from crocus flowers")
        speaker.net.observe_sequence("saffron means a golden spice from crocus flowers")
        before = speaker.net.successors("saffron")[0][1]
        speaker.respond("what does saffron mean")
        after = speaker.net.successors("saffron")[0][1]
        self.assertGreater(after, before)  # recall is use; use strengthens


if __name__ == "__main__":
    unittest.main()
