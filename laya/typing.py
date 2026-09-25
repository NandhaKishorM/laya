"""Type definitions for Laya question schemas and public APIs.

This module provides TypedDict definitions for question schemas, enabling
IDE autocomplete, static type checking, and documentation of expected formats.
"""

import sys
from typing import Any, Literal, NotRequired, TypedDict, Union

# Question type literal - matches QTYPES in common.py
QType = Literal["choice", "score", "noul"]

# Runtime mapping (kept in common.py for backward compatibility)
# QTYPES: Dict[QType, int] = {"choice": 0, "score": 1, "noul": 2}

if sys.version_info >= (3, 11):
    from typing import NotRequired, TypedDict
else:
    from typing import TypedDict

    from typing_extensions import NotRequired


# Type-specific question schemas using TypedDict
class ChoiceQuestion(TypedDict):
    """A choice question: pick one option from a set.

    Attributes:
        type: Must be "choice"
        instructions: Text describing what to decide
        criteria: Option labels -> descriptions (dict), or just labels (list)
    """

    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str] | list[str]


class ScoreQuestion(TypedDict):
    """A score question: rate on an ordinal scale.

    Attributes:
        type: Must be "score"
        instructions: Text describing what to rate
        criteria: List of level descriptions, index 0 = lowest
    """

    type: Literal["score"]
    instructions: str
    criteria: list[str]


class NoulQuestion(TypedDict):
    """A noul (binary probability) question: P(true).

    Attributes:
        type: Must be "noul"
        instructions: Text describing the yes/no question
        criteria: Optional dict with "true"/"false" descriptions
    """

    type: Literal["noul"]
    instructions: str
    criteria: NotRequired[dict[str, str]]


# Union of all question types for type checking
Question = ChoiceQuestion | ScoreQuestion | NoulQuestion

# Dictionary mapping question IDs to question definitions
Questions = dict[str, Question]


# Result-related types
class UsageDict(TypedDict):
    """Token usage information."""

    input_tokens: int
    output_tokens: int


class RouteDecisionDict(TypedDict):
    """Routing decision from Router.route()."""

    model: str
    repo: str
    reason: str
    detection: NotRequired[dict[str, Any] | None]
    workflow: NotRequired[str | None]


# State type accepted by predict()
State = Union[str, dict[str, Any], list[Any]]
