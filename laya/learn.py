"""Continuous learning, from vegaml: answer typed questions from day one and keep learning from every answer
you confirm, on your own machine.

    from laya import Learner

    learner = Learner(questions, examples=rows, workdir="./state")
    learner.decide("I was charged twice")["answers"]["team"]["answer"]
    learner.feedback("I was charged twice", {"team": "billing"})   # learns every 50, in the background

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


class Learner:
    """vegaml's continuous learner. `Learner(questions, examples=rows, workdir=...)` returns a
    `vegaml.learn.Learner`; every argument is passed through. See `help(vegaml.learn.Learner)`."""

    def __new__(cls, *args, **kwargs):
        return _learner()(*args, **kwargs)


__all__ = ["Learner"]
