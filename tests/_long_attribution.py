"""Shared control-flow cases for the two predict_long script suites; no model weights."""
import copy
import math

from laya.hooks import default_hooks, set_default_hooks


def check_long_attribution(factory, check):
    questions = {"dept": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]},
                 "flag": {"type": "noul", "instructions": "?"},
                 "severity": {"type": "score", "instructions": "?", "criteria": ["low", "high"]}}
    state = "".join(chr(65 + (i * i % 23)) for i in range(300))
    winners = {"dept": 8, "flag": 6, "severity": 3}

    def run(transform=None, tier="callback", reverse_start=False, shared=False, ties=False):
        agent = factory()
        agent.cfg = {"max_len": 128, "head_max_len": 20}
        original, shared_answers = [], {}

        def decode(logits, act, items, ids, internal, row, **kwargs):
            index = row // len(ids)
            answers = {}
            for qid, q in questions.items():
                conf = 0.9 if index == winners[qid] or ties else 0.6
                entropy_conf = 1 + (conf * math.log(conf) + (1 - conf) * math.log(1 - conf)) / math.log(2)
                answer = {"type": q["type"], "confidence": entropy_conf, "answer_confidence": conf,
                          "action": {"act_probability": 0.75}}
                if qid == "flag":
                    answer["noul"] = 0.9 if index == winners[qid] else 0.1
                    answer["confidence"] = answer["answer_confidence"] = 0.9
                elif qid == "dept":
                    choice = "b" if index == winners[qid] else "a"
                    answer.update(choice=choice,
                                  probabilities={key: conf if key == choice else 1 - conf for key in ("a", "b")})
                else:
                    score = conf if index == winners[qid] else 1 - conf
                    answer.update(score=score, legend={"0": "low", "1": "high"},
                                  probabilities={"0": 1 - score, "1": score})
                answers[qid] = shared_answers.setdefault(qid, answer) if shared else answer
            return answers

        agent._decode_answers = decode

        def end(ctx):
            original.extend(ctx.results)
            if transform is not None:
                transform(ctx)

        class End:
            def on_predict_end(self, ctx):
                end(ctx)

        async def async_end(ctx):
            end(ctx)

        kwargs = {}
        defaults = default_hooks()
        if tier == "installed":
            agent.hooks = [End()]
        elif tier == "default":
            set_default_hooks([End()])
        elif tier == "per-call":
            kwargs["hooks"] = [End()]
        else:
            kwargs["on_predict_end"] = async_end if tier == "async" else end
        if reverse_start:
            kwargs["on_predict_start"] = lambda ctx: ctx.states.reverse()
        try:
            result = agent.predict_long(state, copy.deepcopy(questions), window=64, stride=32,
                                        batch_size=100, **kwargs)
        finally:
            if tier == "default":
                set_default_hooks(defaults)
        return result, original

    baseline, original = run()
    check("attribution/nine windows were scored", len(original), 9)
    for qid, index in winners.items():
        check("attribution/%s fixture has its own winning span" % qid,
              baseline["answers"][qid]["window"],
              {"index": index, "token_start": index * 32,
               "token_end": min(index * 32 + 64, 300), "count": 9})
        check("attribution/%s does not annotate the hook's result object" % qid,
              "window" in original[index]["answers"][qid], False)

    for tier in ("callback", "per-call", "installed", "default", "async"):
        result, _ = run(lambda ctx: ctx.results.reverse(), tier=tier)
        check("attribution/%s reordering preserves answers and original spans" % tier,
              result, baseline)

    def reorder_wrappers(ctx):
        ctx.results = [dict(row) for row in reversed(ctx.results)]

    result, _ = run(reorder_wrappers)
    check("attribution/new result wrappers retain the original answers' spans", result, baseline)

    def move_answers(ctx):
        ctx.results[0]["answers"], ctx.results[-1]["answers"] = (
            ctx.results[-1]["answers"], ctx.results[0]["answers"])

    result, _ = run(move_answers)
    check("attribution/answers moved between result wrappers retain their spans",
          result["answers"], baseline["answers"])

    result, _ = run(lambda ctx: ctx.results.reverse(), ties=True)
    check("attribution/a tie still selects the first final result, with its original span",
          result["answers"]["dept"]["window"]["index"], 8)

    def annotate(ctx):
        for row in ctx.results:
            for answer in row["answers"].values():
                answer["reliability"] = {"status": "observed"}

    result, _ = run(annotate)
    for qid in questions:
        check("attribution/%s permits reliability annotations" % qid,
              result["answers"][qid], {**baseline["answers"][qid],
                                       "reliability": {"status": "observed"}})

    def clone(ctx):
        ctx.results = copy.deepcopy(ctx.results)
        for row in ctx.results:
            for answer in row["answers"].values():
                answer["window"] = {"index": 999}

    for name, transform, reverse_start, shared in (
            ("cloned answers", clone, False, False),
            ("rewritten start states", None, True, False),
            ("ambiguous shared answers", None, False, True)):
        result, _ = run(transform, reverse_start=reverse_start, shared=shared)
        check("attribution/%s names no span" % name,
              [qid for qid, answer in result["answers"].items() if "window" in answer], [])
        check("attribution/%s still reports scored windows" % name, result["usage"]["windows"], 9)

    for qid, field, value in (("dept", "choice", "edited"),
                              ("dept", "answer_confidence", 0.95),
                              ("dept", "confidence", 0.95),
                              ("flag", "noul", 0.95),
                              ("severity", "score", 0.95),
                              ("severity", "legend", {"0": "edited", "1": "high"})):
        def edit(ctx):
            answer = ctx.results[winners[qid]]["answers"][qid]
            answer[field] = value
            answer["window"] = {"index": 999}

        result, _ = run(edit)
        check("attribution/an edited %s.%s names no span" % (qid, field),
              "window" in result["answers"][qid], False)
        check("attribution/an edited %s.%s is still returned" % (qid, field),
              result["answers"][qid][field], value)
        for other in questions.keys() - {qid}:
            check("attribution/editing %s.%s keeps %s's span" % (qid, field, other),
                  result["answers"][other], baseline["answers"][other])

    def edit_nested(ctx):
        ctx.results[winners["dept"]]["answers"]["dept"]["probabilities"]["b"] = 0.89

    result, _ = run(edit_nested)
    check("attribution/an in-place probability edit names no span",
          "window" in result["answers"]["dept"], False)
