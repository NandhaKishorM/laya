"""Typed answer objects returned by :meth:`laya.Agent.system_one`.

These are what callers write code against: ``answers["intent"].choice`` and
``answers["intent"].confidence`` instead of raw dict indexing. Every object also behaves
like the dict it replaces (``answer["choice"]``, ``.get()``, ``**answer``) so code written
against the previous raw-dict return keeps working.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Mapping


class _DictCompat(Mapping):
    """Mapping view over a dataclass, for backward compatibility with raw-dict returns."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


@dataclass(frozen=True)
class Action(_DictCompat):
    """The act/escalate signal accompanying an answer."""

    act_probability: float


@dataclass(frozen=True)
class ChoiceAnswer(_DictCompat):
    """Answer to a ``choice`` question."""

    choice: str
    probabilities: Dict[str, float]
    confidence: float
    action: Action
    type: str = "choice"


@dataclass(frozen=True)
class ScoreAnswer(_DictCompat):
    """Answer to a ``score`` question. ``score`` is the probability-weighted level."""

    score: float
    legend: Dict[str, Any]
    probabilities: Dict[str, float]
    confidence: float
    action: Action
    type: str = "score"


@dataclass(frozen=True)
class NoulAnswer(_DictCompat):
    """Answer to a ``noul`` question. ``noul`` is P(true)."""

    noul: float
    confidence: float
    action: Action
    type: str = "noul"

    @property
    def is_true(self) -> bool:
        """Whether the statement holds more likely than not."""
        return self.noul >= 0.5


@dataclass(frozen=True)
class Usage(_DictCompat):
    """Token usage for a request."""

    input_tokens: int
    output_tokens: int = 0


@dataclass(frozen=True)
class Truncation(_DictCompat):
    """Records that a question's state did not fit the context window."""

    question_id: str
    kept_tokens: int
    state_tokens: int

    @property
    def dropped_tokens(self) -> int:
        return max(0, self.state_tokens - self.kept_tokens)


@dataclass(frozen=True)
class Response(_DictCompat):
    """Full result of a :meth:`system_one` call."""

    answers: Dict[str, Any]
    usage: Usage
    model: str = "laya-rl-agent"
    request_id: str = ""
    truncated: List[Truncation] = field(default_factory=list)


__all__ = [
    "Action",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "Usage",
    "Truncation",
    "Response",
    "_DictCompat",
]
