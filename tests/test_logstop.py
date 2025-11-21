"""
Unit tests for the LogSTOP evaluation function.

Test basic operators:
python -m unittest tests.test_logstop.TestLogSTOPBasicOps

Test logical equivalences:
python -m unittest tests.test_logstop.TestLogSTOPEquivalences
"""

from unittest.mock import patch
from src.logstop import LTLFormula, logstop
import unittest
import math

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

    def test_local_formula(self):
        trace = {"p": [0.2, 0.4, 0.6, 0.8]}
        phi = LTLFormula("class", "p")
        result = logstop(trace, phi, 1, 3, w=2)  # average of p[1] and p[2]
        expected_prob = (0.4 + 0.6) / 2
        expected_log = math.log(expected_prob)
        self.assertAlmostEqual(result, expected_log)

    def test_not_formula(self):
        trace = {"p": [0.3, 0.7, 0.5]}
        phi = LTLFormula("not", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 2, w=3)  # average of p[0], p[1], p[2]
        prob = (0.3 + 0.7 + 0.5) / 3
        expected_log = math.log(1.0 - prob)
        self.assertAlmostEqual(result, expected_log)

    def test_and_formula(self):
        trace = {"p": [0.9, 0.8], "q": [0.6, 0.7]}
        phi = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 1, w=2)
        prob_p = (0.9 + 0.8) / 2
        prob_q = (0.6 + 0.7) / 2
        expected_log = math.log(prob_p * prob_q)
        self.assertAlmostEqual(result, expected_log)

    def test_or_formula(self):
        trace = {"p": [0.2, 0.4], "q": [0.5, 0.3]}
        phi = LTLFormula("or", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 1, w=2)
        prob_p = (0.2 + 0.4) / 2
        prob_q = (0.5 + 0.3) / 2
        expected_prob = prob_p + prob_q - (prob_p * prob_q)
        expected_log = math.log(expected_prob)
        self.assertAlmostEqual(result, expected_log)

    def test_next_formula(self):
        trace = {"p": [0.1, 0.9, 0.8, 0.7]}
        phi = LTLFormula("next", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 3, w=2)  # next from index 0 to 3 with window 2 means evaluate at indices 2 and 3
        prob = (0.8 + 0.7) / 2
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

    def test_eventually_formula(self):
        trace = {"p": [0.1, 0.4, 0.9, 0.2]}
        phi = LTLFormula("eventually", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 3, w=1)
        prob = 1 - (1 - 0.1) * (1 - 0.4) * (1 - 0.9) * (1 - 0.2)
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

    def test_always_formula(self):
        trace = {"p": [0.9, 0.8, 0.7]}
        phi = LTLFormula("always", LTLFormula("class", "p"))
        result = logstop(trace, phi, 0, 2, w=1)
        prob = 0.9 * 0.8 * 0.7
        expected_log = math.log(prob)
        self.assertAlmostEqual(result, expected_log)

    def test_until_formula(self):
        trace = {"p": [0.2, 0.5, 0.8], "q": [0.1, 0.4, 0.9]}
        phi = LTLFormula("until", LTLFormula("class", "p"), LTLFormula("class", "q"))
        result = logstop(trace, phi, 0, 2, w=1)
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
        result_eventually = logstop(trace, phi_eventually, 0, 2, w=1)
        result_not_always_not = logstop(trace, phi_not_always_not, 0, 2, w=1)
        self.assertAlmostEqual(result_eventually, result_not_always_not)

    # Test if eventually == phi or next eventually phi
    def test_eventually_equals_or_next(self):
        trace = {"p": [0.4, 0.7, 0.5]}
        phi_eventually = LTLFormula("eventually", LTLFormula("class", "p"))
        phi_or_next = LTLFormula("or", LTLFormula("class", "p"),
                                 LTLFormula("next", phi_eventually))
        result_eventually = logstop(trace, phi_eventually, 0, 2, w=1)
        result_or_next = logstop(trace, phi_or_next, 0, 2, w=1)
        self.assertAlmostEqual(result_eventually, result_or_next)

    # Test if always == not eventually not
    def test_always_equals_not_eventually_not(self):
        trace = {"p": [0.5, 0.9, 0.6]}
        phi_always = LTLFormula("always", LTLFormula("class", "p"))
        phi_not_eventually_not = LTLFormula("not", LTLFormula("eventually", LTLFormula("not", LTLFormula("class", "p"))))
        result_always = logstop(trace, phi_always, 0, 2, w=1)
        result_not_eventually_not = logstop(trace, phi_not_eventually_not, 0, 2, w=1)
        self.assertAlmostEqual(result_always, result_not_eventually_not)

    # Test if p and q == not (not p or not q)
    def test_and_equals_not_or_not(self):
        trace = {"p": [0.7, 0.8], "q": [0.6, 0.9]}
        phi_and = LTLFormula("and", LTLFormula("class", "p"), LTLFormula("class", "q"))
        phi_not_or_not = LTLFormula("not", LTLFormula("or", LTLFormula("not", LTLFormula("class", "p")), LTLFormula("not", LTLFormula("class", "q"))))
        result_and = logstop(trace, phi_and, 0, 1, w=1)
        result_not_or_not = logstop(trace, phi_not_or_not, 0, 1, w=1)
        self.assertAlmostEqual(result_and, result_not_or_not)