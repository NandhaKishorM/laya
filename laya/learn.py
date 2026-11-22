"""Continuous learning, from vegaml: answer typed questions from day one and keep learning from every answer
you confirm, on your own machine.

    from laya import Learner

    learner = Learner(questions, examples=rows, workdir="./state")
    learner.decide("I was charged twice")["answers"]["team"]["answer"]
    learner.feedback("I was charged twice", {"team": "billing"})   # learns every 50, in the background

    desk = Escalator(learner, teacher=my_review_queue)   # what it is unsure of goes to your teacher
    desk.decide("I was charged twice")["answers"]["team"]["source"]   # "teacher" until it has learned

This is vegaml's learner (`vegaml.learn`), re-exported so laya users get it with no other install: it reads
with vega (the 0.8B reader by default), not with laya's own encoder. Full guide:
https://github.com/NandhaKishorM/vegaml/blob/main/docs/learn.md
"""


def _learner():
    """vegaml's Learner class, imported on first use so `import laya` stays light and works without vegaml."""
    try:
        from vegaml.learn import Learner as VegamlLearner
    except ImportError as e:  # vegaml missing, or from before continuous learning shipped
        raise ImportError(
            "laya.Learner is vegaml's continuous learner and needs vegaml 0.17 or newer: "
            "pip install -U laya (or pip install -U 'vegaml>=0.17')"
        ) from e
    return VegamlLearner


def _escalator():
    """vegaml's Escalator class, imported on first use, like the Learner."""
    try:
        from vegaml.learn import Escalator as VegamlEscalator
    except ImportError as e:  # vegaml missing, or from before the teacher shipped
        raise ImportError(
            "laya.Escalator is vegaml's teacher for the learner and needs vegaml 0.18 or newer: "
            "pip install -U laya (or pip install -U 'vegaml>=0.18')"
        ) from e
    return VegamlEscalator


class Escalator:
    """A teacher for what the learner is unsure of. `Escalator(learner, teacher=fn)` returns a
    `vegaml.learn.Escalator`: answers the learner is unsure of (abstain, unbound, or a learned answer
    below its confidence line, plus a small audit share) go to `fn(text, questions) -> {question: answer}`,
    whose answer is served and learned from, so the share that needs the teacher falls as it learns.
    The teacher is yours: a person's review queue, a rules engine, or any model you run.
    See `help(vegaml.learn.Escalator)`."""

    def __new__(cls, *args, **kwargs):
        return _escalator()(*args, **kwargs)


def vllm_teacher(*args, **kwargs):
    """A teacher you run yourself: a model served with vLLM (or any OpenAI-compatible server) on your own
    machines, for `Escalator(learner, teacher=vllm_teacher("http://localhost:8000/v1"))`. Nothing leaves
    your network and there is no cost per call. See `help(vegaml.learn.vllm_teacher)`."""
    try:
        from vegaml.learn import vllm_teacher as make
    except ImportError as e:
        raise ImportError(
            "laya.vllm_teacher is vegaml's and needs vegaml 0.18 or newer: pip install -U laya "
            "(or pip install -U 'vegaml>=0.18')"
        ) from e
    return make(*args, **kwargs)


class Learner:
    """vegaml's continuous learner. `Learner(questions, examples=rows, workdir=...)` returns a
    `vegaml.learn.Learner`; every argument is passed through. See `help(vegaml.learn.Learner)`."""

    def __new__(cls, *args, **kwargs):
        return _learner()(*args, **kwargs)


__all__ = ["Escalator", "Learner", "vllm_teacher"]
