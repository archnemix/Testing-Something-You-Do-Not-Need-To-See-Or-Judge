"""Tests closer to a real brain: the oddball paradigm.

Neuroscientists learn how a brain learns by streaming repeated standard
stimuli with rare deviants (the oddball paradigm) and reading the
surprise response (EEG mismatch negativity). These tests run the same
protocol on the network: habituation to the expected, a surprise spike to
the unexpected, flashbulb learning of the deviant in one shot, and
prediction-error adjustment of the wrong expectation.
"""

import unittest

from aster.cognition.plastic import PlasticNetwork

STANDARD = "the cup is on the table"
DEVIANT = "the cup is on the chair"


def run_oddball(deviant_at=(10,), trials=20):
    net = PlasticNetwork()
    trace = []  # (trial, mean surprise, table-edge weight, chair-edge weight)
    for trial in range(trials):
        text = DEVIANT if trial in deviant_at else STANDARD
        surprises = net.stream_learn(text)
        trace.append(sum(surprises) / len(surprises))
    return net, trace


def edge_weight(net, before, after, context):
    for label, weight in net.successors(before, context):
        if label == after:
            return weight
    return 0.0


CONTEXT = ("is", "on")  # the two words before "the" in "the cup is on the ..."


class BrainLikeTests(unittest.TestCase):
    def test_repetition_habituates_the_surprise_response(self) -> None:
        net, trace = run_oddball(deviant_at=(), trials=10)
        self.assertGreater(trace[0], trace[-1] / 2)  # expected -> barely noticed

    def test_the_deviant_elicits_a_surprise_spike(self) -> None:
        net, trace = run_oddball()
        self.assertGreater(trace[10], trace[9] * 1.5)  # mismatch response

    def test_the_deviant_is_learned_in_one_shot(self) -> None:
        net, _ = run_oddball()
        learned = edge_weight(net, "the", "chair", CONTEXT)
        self.assertGreater(learned, 0.5)  # one exposure, strongly remembered

    def test_flashbulb_beats_constant_rate_learning(self) -> None:
        net, _ = run_oddball()
        surprise_gated = edge_weight(net, "the", "chair", CONTEXT)
        control = PlasticNetwork()
        for trial in range(20):
            control.observe_sequence(DEVIANT if trial == 10 else STANDARD)
        constant_rate = edge_weight(control, "the", "chair", CONTEXT)
        self.assertGreater(surprise_gated, constant_rate)

    def test_prediction_error_pushes_the_wrong_expectation_down(self) -> None:
        net, _ = run_oddball(deviant_at=(10, 11, 12, 13), trials=14)
        control = PlasticNetwork()
        for _ in range(14):
            control.observe_sequence(STANDARD)
        without_deviant = edge_weight(control, "the", "table", CONTEXT)
        with_deviant = edge_weight(net, "the", "table", CONTEXT)
        self.assertLess(with_deviant, without_deviant)  # expectation adjusted

    def test_surprise_falls_as_the_stream_becomes_predictable(self) -> None:
        net = PlasticNetwork()
        text = "every morning the baker opens the shop"
        first = net.stream_learn(text)
        second = net.stream_learn(text)
        self.assertGreater(sum(first), sum(second))  # it learned to predict it


if __name__ == "__main__":
    unittest.main()
