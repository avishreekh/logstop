"""
Unit tests for the LogSTOP evaluation function.

Test basic operators:
python -m unittest tests.test_logstop.TestLogSTOPBasicOps

Test logical equivalences:
python -m unittest tests.test_logstop.TestLogSTOPEquivalences

Test memoization functionality:
python -m unittest tests.test_logstop.TestLogSTOPWithMemoization
"""

from unittest.mock import patch
from src.logstop import LTLFormula, logstop as score_processed_trace
from src.preprocess import preprocess_trace
import unittest
import math


def logstop(trace, phi, start_idx, end_idx, smoothing_radii=0, memo=None):
    processed_trace = preprocess_trace(trace, smoothing_radii)
    return score_processed_trace(processed_trace, phi, start_idx, end_idx, memo)


class TestPreprocessing(unittest.TestCase):
    def test_smoothing_uses_complete_candidate_bounds(self):
        processed = preprocess_trace({"p": [0.2, 0.4, 0.6, 0.8]}, 1)
        self.assertEqual(len(processed["p"]), 4)
        self.assertAlmostEqual(processed["p"][0], 0.3)
        self.assertAlmostEqual(processed["p"][1], 0.4)
        self.assertAlmostEqual(processed["p"][3], 0.7)

    def test_property_specific_radii(self):
        processed = preprocess_trace(
            {"p": [0.0, 1.0], "q": [0.25, 0.75]}, {"p": 1}
        )
        self.assertEqual(processed["p"], [0.5, 0.5])
        self.assertEqual(processed["q"], [0.25, 0.75])

class TestLogSTOPBasicOps(unittest.TestCase):
    def test_basic_true(self):
        trace = {"p": [0.1, 0.5, 0.9]}
        phi = LTLFormula("True")
        result = logstop(trace, phi, 0, 2)
        self.assertEqual(result, 0.0)  # log(1) = 0

    def test_basic_false(self):
        trace = {"p": [0.1, 0.5, 0.9]}
        phi = LTLFormula("False")
        result = logstop(trace, phi, 0, 2)
        self.assertEqual(result, float("-inf"))  # log(0) = -inf

    def test_local_formula_with_centered_smoothing(self):
        trace = {"p": [0.2, 0.4, 0.6, 0.8]}
        phi = LTLFormula("class", "p")
        result = logstop(trace, phi, 1, 3, smoothing_radii=1)
        expected_prob = (0.2 + 0.4 + 0.6) / 3
        expected_log = math.log(expected_prob)
        self.assertAlmostEqual(result, expected_log)

    def test_not_formula(self):
        trace = {"p": [0.3, 0.7, 0.5]}
        phi = LTLFormula("not", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 2, smoothing_radii=2)
        prob = (0.3 + 0.7 + 0.5) / 3
        expected_log = math.log(1.0 - prob)
        self.assertAlmostEqual(result, expected_log)

    def test_and_formula(self):
        trace = {"p": [0.9, 0.8], "q": [0.6, 0.7]}
        phi = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 1, smoothing_radii=1)
        prob_p = (0.9 + 0.8) / 2
        prob_q = (0.6 + 0.7) / 2
        expected_log = math.log(prob_p * prob_q)
        self.assertAlmostEqual(result, expected_log)

    def test_or_formula(self):
        trace = {"p": [0.2, 0.4], "q": [0.5, 0.3]}
        phi = LTLFormula("or", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 1, smoothing_radii=1)
        prob_p = (0.2 + 0.4) / 2
        prob_q = (0.5 + 0.3) / 2
        expected_prob = prob_p + prob_q - (prob_p * prob_q)
        expected_log = math.log(expected_prob)
        self.assertAlmostEqual(result, expected_log)

    def test_next_advances_one_frame_with_smoothing(self):
        trace = {"p": [0.1, 0.9, 0.8, 0.7]}
        phi = LTLFormula("next", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 3, smoothing_radii=1)
        prob = (0.1 + 0.9 + 0.8) / 3
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

    def test_property_specific_smoothing_radii(self):
        trace = {
            "p": [0.0, 0.6, 0.9],
            "q": [0.3, 0.6, 0.9],
        }
        phi = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 1, 2, smoothing_radii={"p": 0, "q": 1})
        expected = math.log(0.6) + math.log((0.3 + 0.6 + 0.9) / 3)
        self.assertAlmostEqual(result, expected)

    def test_integer_radius_applies_to_every_property(self):
        trace = {"p": [0.2, 0.8], "q": [0.4, 0.6]}
        phi = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 1, smoothing_radii=1)
        self.assertAlmostEqual(result, math.log(0.5) + math.log(0.5))

    def test_missing_radius_defaults_to_zero(self):
        trace = {"p": [0.2, 0.8], "q": [0.4, 0.6]}
        phi = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 1, smoothing_radii={"p": 1})
        self.assertAlmostEqual(result, math.log(0.5) + math.log(0.4))

    def test_eventually_formula(self):
        trace = {"p": [0.1, 0.4, 0.9, 0.2]}
        phi = LTLFormula("eventually", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 3, smoothing_radii=0)
        prob = 1 - (1 - 0.1) * (1 - 0.4) * (1 - 0.9) * (1 - 0.2)
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

    def test_always_formula(self):
        trace = {"p": [0.9, 0.8, 0.7]}
        phi = LTLFormula("always", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 2, smoothing_radii=0)
        prob = 0.9 * 0.8 * 0.7
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

    def test_until_formula(self):
        trace = {"p": [0.2, 0.5, 0.8], "q": [0.1, 0.4, 0.9]}
        phi = LTLFormula("until", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 2, smoothing_radii=0)
        # Manual computation of until probability
        prob = (0.1) + ((1 - 0.1) * 0.2 * (0.4 + ((1 - 0.4) * 0.5 * (0.9))))
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

class TestLogSTOPEquivalences(unittest.TestCase):
    # Test if eventually == not always not
    def test_eventually_equals_not_always_not(self):
        trace = {"p": [0.3, 0.6, 0.9]}
        phi_eventually = LTLFormula("eventually", LTLFormula("class", "p"))
        phi_not_always_not = LTLFormula("not", LTLFormula("always", LTLFormula("not", LTLFormula("class", "p"))))
        result_eventually = logstop(trace, phi_eventually, 0, 2, smoothing_radii=0)
        result_not_always_not = logstop(trace, phi_not_always_not, 0, 2, smoothing_radii=0)
        self.assertAlmostEqual(result_eventually, result_not_always_not)

    # Test if eventually == phi or next eventually phi
    def test_eventually_equals_or_next(self):
        trace = {"p": [0.4, 0.7, 0.5]}
        phi_eventually = LTLFormula("eventually", LTLFormula("class", "p"))
        phi_or_next = LTLFormula("or", LTLFormula("class", "p"),
                                 LTLFormula("next", phi_eventually))
        result_eventually = logstop(trace, phi_eventually, 0, 2, smoothing_radii=0)
        result_or_next = logstop(trace, phi_or_next, 0, 2, smoothing_radii=0)
        self.assertAlmostEqual(result_eventually, result_or_next)

    # Test if always == not eventually not
    def test_always_equals_not_eventually_not(self):
        trace = {"p": [0.5, 0.9, 0.6]}
        phi_always = LTLFormula("always", LTLFormula("class", "p"))
        phi_not_eventually_not = LTLFormula("not", LTLFormula("eventually", LTLFormula("not", LTLFormula("class", "p"))))
        result_always = logstop(trace, phi_always, 0, 2, smoothing_radii=0)
        result_not_eventually_not = logstop(trace, phi_not_eventually_not, 0, 2, smoothing_radii=0)
        self.assertAlmostEqual(result_always, result_not_eventually_not)

    # Test if p and q == not (not p or not q)
    def test_and_equals_not_or_not(self):
        trace = {"p": [0.7, 0.8], "q": [0.6, 0.9]}
        phi_and = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        phi_not_or_not = LTLFormula("not", LTLFormula("or", LTLFormula("not", LTLFormula("class", "p")), LTLFormula("not", LTLFormula("class", "q"))))
        result_and = logstop(trace, phi_and, 0, 1, smoothing_radii=0)
        result_not_or_not = logstop(trace, phi_not_or_not, 0, 1, smoothing_radii=0)
        self.assertAlmostEqual(result_and, result_not_or_not)

class TestLogSTOPWithMemoization(unittest.TestCase):
    def test_suffix_scores_share_a_memo_for_fixed_end(self):
        trace = preprocess_trace({"p": [0.2, 0.5, 0.8], "q": [0.1, 0.4, 0.9]})
        phi = LTLFormula("until", LTLFormula("class", "p"), LTLFormula("class", "q"))
        memo = {}

        score_processed_trace(trace, phi, 0, 2, memo=memo)
        memo_size = len(memo)
        cached_suffix = score_processed_trace(trace, phi, 1, 2, memo=memo)

        self.assertEqual(memo_size, len(memo))
        self.assertEqual(cached_suffix, memo[(phi, 1)])

    def test_memoization_effectiveness(self):
        trace = {"p": [0.2, 0.5, 0.8], "q": [0.1, 0.4, 0.9]}
        phi = LTLFormula("until", LTLFormula("class", "p"), LTLFormula("class", "q"))
        memo = {}
        result1 = logstop(trace, phi, 0, 2, smoothing_radii=0, memo=memo)
        memo_size_after_first = len(memo)
        result2 = logstop(trace, phi, 0, 2, smoothing_radii=0, memo=memo)
        memo_size_after_second = len(memo)
        self.assertEqual(result1, result2)
        self.assertEqual(memo_size_after_first, memo_size_after_second)  # No new entries added on second call

    def test_memoization_keys(self):
        trace = {"p": [0.3, 0.6], "q": [0.2, 0.5]}
        phi = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        memo = {}
        logstop(trace, phi, 0, 1, smoothing_radii=0, memo=memo)
        expected_keys = [
            (LTLFormula("class", "p"), 0),
            (LTLFormula("class", "q"), 0),
            (phi, 0)
        ]
        # check string representations of keys for equality
        memo_keys_str = [ (k[0].__str__(), k[1]) for k in memo.keys() ]
        expected_keys_str = [ (k[0].__str__(), k[1]) for k in expected_keys ]
        for key in expected_keys_str:
            self.assertIn(key, memo_keys_str)

    def test_memoization_values(self):
        trace = {"p": [0.4, 0.7], "q": [0.3, 0.6]}
        phi = LTLFormula("or", LTLFormula("class", "p"), LTLFormula("class", "q"))
        memo = {}
        logstop(trace, phi, 0, 1, smoothing_radii=1, memo=memo)
        prob_p = (0.4 + 0.7) / 2
        prob_q = (0.3 + 0.6) / 2
        expected_prob = prob_p + prob_q - (prob_p * prob_q)
        expected_log = math.log(expected_prob)
        # check that the memo contains correct value for the main formula
        key = (phi, 0)
        self.assertIn((key[0].__str__(), key[1]), [(k[0].__str__(), k[1]) for k in memo.keys()])
        self.assertAlmostEqual(memo[key], expected_log)
