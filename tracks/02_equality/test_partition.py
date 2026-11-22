"""Offline CSV partition regressions: python -m unittest test_partition."""
import csv
import io
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

from analyze import partition_rows, read_prompt_csv


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["endpoint", "family", "model"])
    writer.writerows(rows)
    return stream.getvalue()


class PromptCSVTests(unittest.TestCase):
    def test_quoted_prompt_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.csv"
            path.write_text('id,prompt\n111,"Which is heavier, iron or feathers?"\n'
                            '112,"He said ""yes"""\n', encoding="utf-8")
            self.assertEqual(read_prompt_csv(path), [
                ("111", "Which is heavier, iron or feathers?"), ("112", 'He said "yes"')])

    def test_truncated_or_duplicate_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.csv"
            for content in ("id,prompt\n111,Which is heavier, iron or feathers?\n",
                            "id,prompt\n100,First\n100,Second\n"):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_prompt_csv(path)


class PartitionTests(unittest.TestCase):
    def test_pair_relation_survives_csv_roundtrip(self):
        models = {"ep-01": "x", "ep-02": "x", "ep-03": "y",
                  "ep-04": "y", "ep-05": "z", "ep,06\nquoted": "z"}
        families = {ep: "alpha" if model != "z" else "beta"
                    for ep, model in models.items()}
        rows = partition_rows(models, families)
        decoded = {row["endpoint"]: (row["family"], row["model"])
                   for row in csv.DictReader(io.StringIO(csv_text(rows)))}
        self.assertEqual(set(decoded), set(models))
        self.assertEqual(len(rows), len(models))
        for a, b in combinations(models, 2):
            with self.subTest(pair=(a, b)):
                self.assertEqual(decoded[a] == decoded[b], models[a] == models[b])
                self.assertEqual(decoded[a][1] == decoded[b][1], models[a] == models[b])
                self.assertEqual(decoded[a][0] == decoded[b][0], families[a] == families[b])
        self.assertEqual(rows, partition_rows(dict(reversed(list(models.items()))),
                                             dict(reversed(list(families.items())))))

    def test_rejects_model_spanning_families(self):
        with self.assertRaises(ValueError):
            partition_rows({"a": "same", "b": "same"}, {"a": "one", "b": "two"})

    def test_rejects_incomplete_endpoint_coverage(self):
        for models, families in [({"a": "x"}, {}), ({}, {"a": "f"})]:
            with self.subTest(models=models, families=families):
                with self.assertRaises(ValueError):
                    partition_rows(models, families)
        self.assertEqual(partition_rows({}, {}), [])



if __name__ == "__main__":
    unittest.main()
