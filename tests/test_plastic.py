import itertools
import unittest

from aster.cognition.plastic import PlasticNetwork, phrase_features


class PlasticNetworkTests(unittest.TestCase):
    """The user's requirements as behavioral proofs.

    Knowledge lives in pathway strengths (not a dictionary), telling things
    learns them immediately, use strengthens paths, disuse weakens them,
    input streams need no padding or context window, and the network
    physically grows and forgets like neural tissue.
    """

    def test_telling_learns_immediately_and_repetition_is_recognized(self) -> None:
        net = PlasticNetwork()
        self.assertTrue(net.observe("Ada", "is", "a mathematician"))
        self.assertFalse(net.observe("ada", "IS", "A MATHEMATICIAN"))
        self.assertEqual(net.known_fact_count(), 1)
        self.assertGreater(net.path_strength("Ada", "is", "a mathematician"), 0.5)

    def test_retrieval_is_associative_not_string_lookup(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        # case and punctuation variants cue the same concept: activation
        # travels along pathways, no exact string matching happens
        self.assertEqual(
            [label for label, _ in net.retrieve("ada!", "Is")],
            ["a mathematician"],
        )
        self.assertEqual(net.retrieve("Grace", "is"), [])

    def test_use_strengthens_paths_and_disuse_weakens_them(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        before = net.path_strength("Ada", "is", "a mathematician")
        for _ in range(12):
            net.retrieve("Ada", "is")
        after = net.path_strength("Ada", "is", "a mathematician")
        self.assertGreater(after, before)
        net.tick(400)
        faded = net.path_strength("Ada", "is", "a mathematician")
        self.assertLess(faded, after)

    def test_long_disuse_forgets_and_prunes_like_a_brain(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        self.assertTrue(net.retrieve("Ada", "is"))
        grown = net.neuron_count
        net.tick(25000)
        self.assertEqual(net.retrieve("Ada", "is"), [])
        self.assertEqual(net.known_fact_count(), 0)
        self.assertLess(net.neuron_count, grown)

    def test_selective_forgetting_keeps_skills_loses_trivia(self) -> None:
        net = PlasticNetwork()
        net.observe("I had soup", "for", "lunch")  # a one-off detail
        net.observe("Ada", "is", "a mathematician")  # knowledge that gets used
        for _ in range(5):
            net.retrieve("Ada", "is")
        net.tick(3000)
        self.assertEqual(net.retrieve("I had soup", "for"), [])
        self.assertEqual(
            [label for label, _ in net.retrieve("Ada", "is")],
            ["a mathematician"],
        )

    def test_faded_knowledge_relearns_faster_than_first_learning(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        net.tick(1200)  # fades to a weak trace but is not pruned yet
        self.assertLess(net.path_strength("Ada", "is", "a mathematician"), 0.5)
        net.observe("Ada", "is", "a mathematician")  # one re-telling
        relearned = net.path_strength("Ada", "is", "a mathematician")
        self.assertGreater(relearned, 0.65)  # a naive first telling only reaches 0.6

    def test_capacity_competition_evicts_the_weakest(self) -> None:
        small = PlasticNetwork(max_neurons=36)
        large = PlasticNetwork(max_neurons=300)
        for index in range(10):
            small.observe(f"item{index}", "is", f"class{index}")
            large.observe(f"item{index}", "is", f"class{index}")
            if index == 0:
                for _ in range(5):
                    small.retrieve("item0", "is")  # keep practicing the early one
        self.assertLess(small.known_fact_count(), 10)
        self.assertGreater(small.known_fact_count(), 0)
        self.assertEqual(large.known_fact_count(), 10)
        self.assertEqual(
            [label for label, _ in small.retrieve("item0", "is")],
            ["class0"],
        )  # what stays in use survives the pressure
        self.assertEqual(
            [label for label, _ in small.retrieve("item9", "is")],
            ["class9"],
        )  # the newest learning lands too

    def test_network_physically_grows_when_told_new_things(self) -> None:
        net = PlasticNetwork()
        self.assertEqual(net.neuron_count, 0)
        net.observe("Ada", "is", "a mathematician")
        self.assertEqual(net.neuron_count, 9)  # three concepts, three neurons each
        net.observe("Grace", "is", "a mathematician")
        self.assertEqual(net.neuron_count, 12)  # one new concept recruited

    def test_variable_length_stream_needs_no_padding_or_window(self) -> None:
        net = PlasticNetwork()
        tokens = ["".join(parts) for parts in itertools.product("abcde", repeat=3)]
        tiers = ["red", "tall", "green", "round", "soft"]
        for index, token in enumerate(tokens[:120]):
            phrase = token if index % 3 else f"{token} {token[::-1]}"
            net.observe(phrase, "is", tiers[index % 5])
        first = [label for label, _ in net.retrieve("aaa", "is")]
        last = [label for label, _ in net.retrieve("ede", "is")]
        self.assertEqual(first, ["red"])
        self.assertEqual(last, ["soft"])
        self.assertEqual(net.known_fact_count(), 120)

    def test_transitive_chains_resolve_and_cycles_do_not_loop(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        net.observe("a mathematician", "is", "a person")
        net.observe("a person", "is", "Ada")  # cycle back to the cue
        self.assertEqual(
            [label for label, _ in net.retrieve("Ada", "is")],
            ["a mathematician", "a person"],
        )

    def test_relation_gates_keep_facts_separate(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        net.observe("Ada", "has", "a notebook")
        self.assertEqual(
            [label for label, _ in net.retrieve("Ada", "is")],
            ["a mathematician"],
        )
        self.assertEqual(
            [label for label, _ in net.retrieve("Ada", "has")],
            ["a notebook"],
        )

    def test_snapshot_round_trips_the_learned_strengths(self) -> None:
        net = PlasticNetwork()
        net.observe("Ada", "is", "a mathematician")
        net.retrieve("Ada", "is")
        payload = net.snapshot()
        restored = PlasticNetwork()
        restored.restore(payload)
        self.assertEqual(restored.snapshot(), payload)
        self.assertEqual(
            [label for label, _ in restored.retrieve("Ada", "is")],
            ["a mathematician"],
        )

    def test_features_are_case_folded_perception(self) -> None:
        self.assertEqual(phrase_features("Ada!"), phrase_features("ada"))


if __name__ == "__main__":
    unittest.main()
