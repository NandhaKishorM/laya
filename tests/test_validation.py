"""Validation of question definitions at the system_one entry point.

A malformed question used to surface several frames down as an AttributeError,
TypeError, or a RuntimeError from inside the decision head, without naming the
question or the offending field. Every case below must instead raise a
ValueError whose message carries the question id and the shape it expected,
before any tokenization or model call happens.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import validate_question  # noqa: E402


# ----------------------------------------------------------------- happy paths
class TestValidQuestions:
    def test_choice_dict_criteria(self):
        validate_question("q", {"type": "choice", "instructions": "pick one",
                                "criteria": {"a": "first", "b": "second"}})

    def test_choice_list_criteria(self):
        validate_question("q", {"type": "choice", "instructions": "pick one",
                                "criteria": ["a", "b"]})

    def test_score_list_criteria(self):
        validate_question("q", {"type": "score", "instructions": "rate",
                                "criteria": ["low", "high"]})

    def test_noul_without_criteria(self):
        validate_question("q", {"type": "noul", "instructions": "yes or no?"})

    def test_noul_with_dict_criteria(self):
        validate_question("q", {"type": "noul", "instructions": "yes or no?",
                                "criteria": {"true": "holds", "false": "does not"}})

    def test_non_string_instructions_allowed(self):
        # _to_internal json-dumps these; validation must not reject them
        validate_question("q", {"type": "noul", "instructions": {"nested": "ok"}})


# ------------------------------------------------------------- malformed type
class TestBadType:
    def test_typo_bool(self):
        with pytest.raises(ValueError, match="unknown type"):
            validate_question("q", {"type": "bool", "instructions": "?",
                                    "criteria": {"true": "y", "false": "n"}})

    def test_missing_type(self):
        with pytest.raises(ValueError, match="unknown type"):
            validate_question("q", {"instructions": "?"})

    def test_message_names_question(self):
        with pytest.raises(ValueError, match="'urgency'"):
            validate_question("urgency", {"type": "bool", "instructions": "?"})


# -------------------------------------------------------------- missing fields
class TestMissingFields:
    def test_missing_instructions(self):
        with pytest.raises(ValueError, match="missing required field 'instructions'"):
            validate_question("q", {"type": "noul"})

    def test_instructions_error_names_question(self):
        with pytest.raises(ValueError, match="'department'"):
            validate_question("department", {"type": "score", "criteria": ["a"]})


# ------------------------------------------------------------------ choice
class TestChoice:
    def test_no_criteria(self):
        with pytest.raises(ValueError, match="missing required field 'criteria'"):
            validate_question("q", {"type": "choice", "instructions": "?"})

    def test_criteria_none(self):
        with pytest.raises(ValueError, match="missing required field 'criteria'"):
            validate_question("q", {"type": "choice", "instructions": "?", "criteria": None})

    def test_empty_dict_criteria(self):
        with pytest.raises(ValueError, match="empty criteria"):
            validate_question("q", {"type": "choice", "instructions": "?", "criteria": {}})

    def test_empty_list_criteria(self):
        with pytest.raises(ValueError, match="empty criteria"):
            validate_question("q", {"type": "choice", "instructions": "?", "criteria": []})

    def test_tuple_criteria(self):
        with pytest.raises(ValueError, match="expected a dict or a list"):
            validate_question("q", {"type": "choice", "instructions": "?", "criteria": ("a", "b")})


# ------------------------------------------------------------------- score
class TestScore:
    def test_no_criteria(self):
        with pytest.raises(ValueError, match="missing required field 'criteria'"):
            validate_question("q", {"type": "score", "instructions": "?"})

    def test_empty_list_criteria(self):
        with pytest.raises(ValueError, match="empty criteria"):
            validate_question("q", {"type": "score", "instructions": "?", "criteria": []})

    def test_dict_criteria_rejected(self):
        # This one did not raise before and was the worst outcome: the keys
        # rendered as levels and every description was dropped silently.
        with pytest.raises(ValueError, match="criteria as a dict"):
            validate_question("q", {"type": "score", "instructions": "?",
                                    "criteria": {"low": "no rush", "high": "blocking"}})

    def test_string_criteria(self):
        with pytest.raises(ValueError, match="expected a list"):
            validate_question("q", {"type": "score", "instructions": "?", "criteria": "low to high"})


# ------------------------------------------------------------------- noul
class TestNoul:
    def test_list_criteria(self):
        with pytest.raises(ValueError, match="expected a dict"):
            validate_question("q", {"type": "noul", "instructions": "?", "criteria": ["a", "b"]})

    def test_string_criteria(self):
        with pytest.raises(ValueError, match="expected a dict"):
            validate_question("q", {"type": "noul", "instructions": "?", "criteria": "spam?"})


# --------------------------------------------------- non-dict question definition
class TestNotADict:
    def test_string_definition(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_question("q", "is this spam?")

    def test_none_definition(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_question("q", None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
