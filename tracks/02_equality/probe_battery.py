#!/usr/bin/env python3
"""Parse the first labelled numeric answer block, without executing model output."""
import json
import re
import unicodedata

from experiment_numbers import numeric_value

LABELS = "ABCD"
ASSIGNMENT = re.compile(r"(?<!\w)[\"'`*]*([A-D])[\"'`*]*\s*[:=]\s*", re.I)


def parse_battery(text):
    text = unicodedata.normalize("NFKC", text)
    matches = list(ASSIGNMENT.finditer(text))
    if matches:
        result, seen = {}, set()
        for index, match in enumerate(matches):
            label = match.group(1).upper()
            if label in seen:
                break
            seen.add(label)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            fragment = text[match.end():end].split("\n", 1)[0].strip().rstrip(",;}] ")
            value = numeric_value(fragment)
            if value is not None:
                result[label] = value
            if len(seen) == len(LABELS):
                break
        return result
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = "\n".join(candidate.splitlines()[1:-1]).strip()
    try:
        values = json.loads(candidate if candidate.startswith("[") else "[" + candidate + "]")
    except (ValueError, TypeError):
        return {}
    if not isinstance(values, list) or len(values) != len(LABELS):
        return {}
    result = {}
    for label, value in zip(LABELS, values):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        parsed = numeric_value(str(value))
        if parsed is not None:
            result[label] = parsed
    return result


def self_check():
    expected = {"A": 391, "B": 494, "C": 3901, "D": 2183}
    assert parse_battery("A=391\nB=494\nC=3901\nD=2183") == expected
    assert parse_battery(json.dumps(expected)) == expected
    assert parse_battery("391,494,3901,2183") == expected
    assert parse_battery("```json\n[391,494,3901,2183]\n```") == expected
    assert parse_battery("A=17*23\nB=19*26") == {}
    assert parse_battery("A=391\nB=494\nC=3901\nD=2183\nA=999") == expected
    assert parse_battery("A=391\nA=999\nB=494") == {"A": 391}
    assert parse_battery("A=NaN\nB=Infinity") == {}
    assert parse_battery("[true,494,3901,2183]") == {"B": 494, "C": 3901, "D": 2183}
    assert parse_battery("[391,494,3901]") == {}
    print("battery parser self-check passed")


if __name__ == "__main__":
    self_check()
