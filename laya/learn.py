"""Continuous learning, from vegaml: answer typed questions from day one and keep learning from every answer
you confirm, on your own machine.

    from laya import Learner

    learner = Learner(questions, examples=rows, workdir="./state")
    learner.decide("I was charged twice")["answers"]["team"]["answer"]
    learner.feedback("I was charged twice", {"team": "billing"})   # learns every 50, in the background

    desk = Escalator(learner, teacher=my_review_queue)   # what it is unsure of goes to your teacher
    desk.decide("I was charged twice")["answers"]["team"]["source"]   # "teacher" until it has learned

This is vega's learner, from the vegaml-learn package: `pip install vegaml-learn`. It reads with vega (the 0.8B
reader by default), not with laya's own encoder. vegaml-learn is free for noncommercial use (PolyForm
Noncommercial 1.0.0); commercial use needs a licence from Embedding Space, https://embedding.in/licence. laya itself,
including this file, stays Apache 2.0. Full guide: https://github.com/NandhaKishorM/vegaml/blob/main/docs/learn.md
"""

NEEDS = (
    "%s is vega's continuous learning, in the vegaml-learn package: pip install vegaml-learn. "
    "It is free for noncommercial use (PolyForm Noncommercial 1.0.0); commercial use needs a licence from "
    "Embedding Space: https://embedding.in/licence or nandakishor@embedding.in."
)


def _from_vegaml(name, what):
    """`name` from vegaml-learn (through `vegaml.learn`, which vegaml 0.17 to 0.19 also ship), imported on first
    use so `import laya` stays light and works without it."""
    try:
        import vegaml.learn as learn

        return getattr(learn, name)
    except (ImportError, AttributeError) as e:  # not installed, or a vegaml from before `name` shipped
        raise ImportError(NEEDS % what) from e


def _learner():
    return _from_vegaml("Learner", "laya.Learner")


def _escalator():
    return _from_vegaml("Escalator", "laya.Escalator")


class Escalator:
    """A teacher for what the learner is unsure of. `Escalator(learner, teacher=fn)` returns a
    `vegaml_learn.Escalator`: answers the learner is unsure of (abstain, unbound, or a learned answer
    below its confidence line, plus a small audit share) go to `fn(text, questions) -> {question: answer}`,
    whose answer is served and learned from, so the share that needs the teacher falls as it learns.
    The teacher is yours: a person's review queue, a rules engine, or any model you run.
    See `help(vegaml_learn.Escalator)`."""

    def __new__(cls, *args, **kwargs):
        return _escalator()(*args, **kwargs)


def vllm_teacher(*args, **kwargs):
    """A teacher you run yourself: a model served with vLLM (or any OpenAI-compatible server) on your own
    machines, for `Escalator(learner, teacher=vllm_teacher("http://localhost:8000/v1"))`. Nothing leaves
    your network and there is no cost per call. See `help(vegaml_learn.vllm_teacher)`."""
    make = _from_vegaml("vllm_teacher", "laya.vllm_teacher")
    return make(*args, **kwargs)


class Learner:
    """vega's continuous learner (pip install vegaml-learn; free for noncommercial use). `Learner(questions,
    examples=rows, workdir=...)` returns a `vegaml_learn.Learner`; every argument is passed through.
    See `help(vegaml_learn.Learner)`."""

    def __new__(cls, *args, **kwargs):
        return _learner()(*args, **kwargs)


__all__ = ["Escalator", "Learner", "vllm_teacher"]
