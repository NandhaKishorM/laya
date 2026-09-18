"""P1: typed exception hierarchy, including backward compatibility with the old bare types."""
import pytest

import laya
from laya.agent import Agent
from laya.errors import (
    ConfigurationError,
    IncompatibleModelError,
    InferenceError,
    InvalidCriteriaError,
    LayaError,
    ModelNotFoundError,
    OptionsTooLongError,
    QuestionError,
    UnknownQuestionTypeError,
)


class TestHierarchy:
    @pytest.mark.parametrize("cls", [
        ConfigurationError, ModelNotFoundError, IncompatibleModelError,
        QuestionError, UnknownQuestionTypeError, InvalidCriteriaError,
        OptionsTooLongError, InferenceError,
    ])
    def test_everything_derives_from_layaerror(self, cls):
        assert issubclass(cls, LayaError)

    def test_question_subtypes(self):
        for c in (UnknownQuestionTypeError, InvalidCriteriaError, OptionsTooLongError):
            assert issubclass(c, QuestionError)

    def test_exported_from_package(self):
        assert laya.LayaError is LayaError and laya.QuestionError is QuestionError


class TestBackwardCompatibility:
    """Existing `except ValueError` / `except FileNotFoundError` code must keep working."""

    @pytest.mark.parametrize("cls", [ConfigurationError, IncompatibleModelError, QuestionError])
    def test_value_error_compatible(self, cls):
        assert issubclass(cls, ValueError)

    def test_file_not_found_compatible(self):
        assert issubclass(ModelNotFoundError, FileNotFoundError)

    def test_runtime_error_compatible(self):
        assert issubclass(InferenceError, RuntimeError)

    def test_caught_by_old_clause(self):
        try:
            raise InvalidCriteriaError("bad")
        except ValueError as e:
            assert isinstance(e, LayaError)


class TestQuestionValidation:
    """_to_internal is a pure function -- no weights needed."""

    def test_unknown_type(self):
        with pytest.raises(UnknownQuestionTypeError, match="unknown type"):
            Agent._to_internal({"type": "bogus", "instructions": "x"}, "q1")

    def test_missing_type(self):
        with pytest.raises(QuestionError, match="missing required key 'type'"):
            Agent._to_internal({"instructions": "x"}, "q1")

    def test_missing_instructions(self):
        with pytest.raises(QuestionError, match="missing required key 'instructions'"):
            Agent._to_internal({"type": "noul"}, "q1")

    def test_not_a_dict(self):
        with pytest.raises(QuestionError, match="must be a dict"):
            Agent._to_internal("nope", "q1")

    def test_error_names_the_question(self):
        with pytest.raises(QuestionError, match="my_qid"):
            Agent._to_internal({"type": "nope", "instructions": "x"}, "my_qid")

    @pytest.mark.parametrize("crit", [None, {}, [], "string"])
    def test_choice_requires_usable_criteria(self, crit):
        with pytest.raises(InvalidCriteriaError):
            Agent._to_internal({"type": "choice", "instructions": "x", "criteria": crit}, "q")

    def test_choice_accepts_list_shorthand(self):
        out = Agent._to_internal({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}, "q")
        assert out["crit"] == {"a": None, "b": None}

    @pytest.mark.parametrize("crit", [None, ["only-one"], "string", {}])
    def test_score_needs_two_levels(self, crit):
        with pytest.raises(InvalidCriteriaError):
            Agent._to_internal({"type": "score", "instructions": "x", "criteria": crit}, "q")

    def test_score_accepts_two_levels(self):
        out = Agent._to_internal({"type": "score", "instructions": "x", "criteria": ["a", "b"]}, "q")
        assert out["crit"] == ["a", "b"]

    def test_noul_criteria_optional(self):
        assert Agent._to_internal({"type": "noul", "instructions": "x"}, "q")["crit"] is None

    def test_noul_rejects_list_criteria(self):
        with pytest.raises(InvalidCriteriaError):
            Agent._to_internal({"type": "noul", "instructions": "x", "criteria": ["a"]}, "q")

    def test_structured_instructions_are_json(self):
        out = Agent._to_internal({"type": "noul", "instructions": {"task": "check"}}, "q")
        assert out["ins"] == '{"task": "check"}'

    def test_structured_criteria_survive_validation(self):
        """P0 #1 end to end: structured criteria pass validation and render as JSON."""
        from laya.common import render_options
        out = Agent._to_internal(
            {"type": "noul", "instructions": "x", "criteria": {"true": {"a": 1}, "false": "no"}}, "q"
        )
        assert render_options(out) == ['false: no', 'true: {"a": 1}']


class TestModuleHygiene:
    """Catches names used on rarely-exercised error paths but never imported."""

    @pytest.mark.parametrize("mod", ["laya", "laya.agent", "laya.aio", "laya.patterns",
                                     "laya.types", "laya.errors", "laya.common"])
    def test_module_has_no_undefined_names(self, mod):
        import ast
        import builtins
        import importlib

        m = importlib.import_module(mod)
        tree = ast.parse(open(m.__file__).read())
        defined = set(dir(m)) | set(dir(builtins))

        # collect locally-bound names (params, assignments, comprehensions, imports)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                a = node.args
                for arg in a.args + a.kwonlyargs + a.posonlyargs:
                    defined.add(arg.arg)
                for extra in (a.vararg, a.kwarg):
                    if extra:
                        defined.add(extra.arg)
                # a def also binds its own name in the enclosing scope
                defined.add(getattr(node, "name", ""))
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)

        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        missing = used - defined
        assert not missing, "%s uses undefined names: %s" % (mod, sorted(missing))
