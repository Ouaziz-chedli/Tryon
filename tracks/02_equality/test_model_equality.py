"""Offline parser, linkage and holdout regressions: python -m unittest test_model_equality."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze_ep2 import cluster, code_value, load_answers, pair_similarities, parse_value
from validate_references import PROMPTS, REFERENCES, split_answers


class ModelEqualityTests(unittest.TestCase):
    def test_formatting_does_not_replace_answer_identity(self):
        for bare, verbose, pid in [
            ("pepper", "Salt and pepper. These are common seasonings.", 106),
            ("decrease", "The opposite of increase is decrease.", 104),
            ("H2O", "The chemical formula of water is H₂O.", 107),
            ("thanks", "The translation is thank you.", 108),
            ("O", "The fifteenth letter of the alphabet is O.", 103),
            ("3901", "47 * 83 = 3,901", 101),
            ("pepper", "I'll let you finish: pepper.", 106),
            ("thanks", "Thankyou", 108),
        ]:
            with self.subTest(pid=pid):
                self.assertEqual(parse_value(pid, bare), parse_value(pid, verbose))
        self.assertNotEqual(parse_value(101, "3881"), parse_value(101, "3901"))
        self.assertNotEqual(parse_value(106, "sugar"), parse_value(106, "pepper"))
        self.assertNotEqual(parse_value(108, "merci"), parse_value(108, "thanks"))
        self.assertNotEqual(parse_value(104, "increase"), parse_value(104, "decrease"))
        self.assertIsNone(parse_value(106, "Sure! Please provide more context."))

    def test_code_equivalence_without_executing_generated_code(self):
        left = "Certainly!\n```python\ndef double(x):\n    return 2 * x\n```\nThis doubles the argument."
        right = "def double(value): return value * 2"
        self.assertEqual(code_value(left), code_value(right))
        self.assertNotEqual(code_value(left), code_value("def double(x): return x * 3"))
        self.assertEqual(code_value(left), code_value("double = lambda value: value * 2"))
        conditional = "def double(x):\n    if x > 0:\n        return 3*x\n    return 2*x"
        self.assertNotEqual(code_value(left), code_value(conditional))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            malicious = f"def double(x): return __import__('pathlib').Path({str(target)!r}).touch()"
            self.assertNotEqual(code_value(malicious), code_value(left))
            self.assertFalse(target.exists())

    def test_linkage_does_not_merge_through_a_bridge(self):
        sims = {("a", "b"): 0.95, ("b", "c"): 0.95, ("a", "c"): 0.2}
        groups = cluster(["a", "b", "c"], sims, 0.9)
        self.assertEqual(groups["a"], groups["b"])
        self.assertNotEqual(groups["a"], groups["c"])

    def test_two_unparseable_responses_are_not_equality_evidence(self):
        same = {"a": {106: ["pepper", "pepper"]},
                "b": {106: ["Salt and pepper.", "pepper"]},
                "c": {106: ["sugar", "sugar"]}}
        sims = pair_similarities(same)
        self.assertGreater(sims["a", "b"], sims["a", "c"])
        invalid = {"a": {100: ["unrelated text"]}, "b": {100: ["unrelated text"]}}
        self.assertEqual(pair_similarities(invalid)["a", "b"], 0)

    def test_duplicate_answers_and_conflicting_prompts_are_rejected(self):
        row = {"endpoint": "ref-01", "id": "100", "repeat": 0,
               "prompt": "A question", "response": "yes"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            for second in [row, {**row, "repeat": 1, "prompt": "Different question"}]:
                path.write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n")
                with self.assertRaises(ValueError):
                    load_answers(path)

    def test_reference_holdout_is_disjoint_and_equal_depth(self):
        answers = {ep: {pid: [f"{ep}/{pid}/{i}" for i in range(9)] for pid in PROMPTS}
                   for ep in REFERENCES}
        training, held_out, _, indices = split_answers(answers, 4, 2666)
        for ep in REFERENCES:
            for pid in PROMPTS:
                self.assertEqual(len(training[ep][pid]), 4)
                self.assertEqual(len(held_out[ep][pid]), 4)
                self.assertTrue(set(training[ep][pid]).isdisjoint(held_out[ep][pid]))
                self.assertTrue(set(indices[ep][str(pid)]["training_indices"]).isdisjoint(
                    indices[ep][str(pid)]["held_out_indices"]))
        answers["ref-04"][100].pop()
        answers["ref-04"][100].pop()
        with self.assertRaises(ValueError):
            split_answers(answers, 4, 2666)


if __name__ == "__main__":
    unittest.main()
