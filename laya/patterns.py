"""Composable decision patterns built on top of :meth:`laya.Agent.system_one`.

These are library-level compositions over the existing primitives -- no model changes.
Every helper takes an agent-like object exposing ``system_one(state, questions)``, so they
work with :class:`laya.Agent`, :class:`laya.AsyncAgent` (via the sync bridge) or a stub in tests.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .errors import InvalidCriteriaError, QuestionError

__all__ = [
    "confidence_gate",
    "route",
    "composite_score",
    "self_consistency",
    "rerank",
    "cascade",
    "fan_out",
    "two_stage_choice",
    "taxonomy_choice",
    "chunk_state",
    "map_reduce",
    "select_tool",
    "check_citation",
    "check_citations",
]


def _answer_value(answer: Any) -> Any:
    """Extract the payload of a typed or raw-dict answer."""
    t = answer["type"] if not hasattr(answer, "type") else answer.type
    if t == "choice":
        return answer["choice"]
    if t == "score":
        return answer["score"]
    return answer["noul"]


def _confidence(answer: Any) -> float:
    return float(answer["confidence"])


def confidence_gate(
    agent,
    state,
    questions: Dict[str, Dict],
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """Split answers into automatic vs. escalated by calibrated confidence.

    Returns ``{"automatic": {...}, "escalate": {...}, "answers": {...}}``. This is the
    selective-automation pattern: act on the confident head, route the tail to review.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1], got %r" % threshold)
    result = agent.system_one(state, questions)
    answers = result["answers"]
    automatic, escalate = {}, {}
    for qid, ans in answers.items():
        (automatic if _confidence(ans) >= threshold else escalate)[qid] = ans
    return {"answers": answers, "automatic": automatic, "escalate": escalate, "result": result}


def route(
    agent,
    state,
    question: Dict,
    routes: Dict[str, Callable[[Any], Any]],
    default: Optional[Callable[[Any], Any]] = None,
    min_confidence: float = 0.0,
    qid: str = "route",
) -> Any:
    """Run one choice question and dispatch to the handler named by the winning option.

    Falls back to ``default`` when the chosen route has no handler, or when confidence is
    below ``min_confidence``. Handlers receive the answer object.
    """
    if question.get("type") != "choice":
        raise QuestionError("route() requires a 'choice' question, got %r" % question.get("type"))
    ans = agent.system_one(state, {qid: question})["answers"][qid]
    choice = ans["choice"]
    handler = routes.get(choice)
    if handler is None or _confidence(ans) < min_confidence:
        if default is None:
            return ans
        return default(ans)
    return handler(ans)


def composite_score(
    agent,
    state,
    questions: Dict[str, Dict],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Combine several questions into one weighted risk/quality score in [0, 1].

    ``score`` answers are normalized by their level count; ``noul`` answers use P(true);
    ``choice`` answers are skipped (no natural ordering). Missing weights default to 1.0.
    """
    result = agent.system_one(state, questions)
    answers = result["answers"]
    weights = weights or {}
    total_w, acc, parts = 0.0, 0.0, {}
    for qid, ans in answers.items():
        t = ans["type"]
        if t == "score":
            n_levels = len(ans["legend"])
            v = float(ans["score"]) / max(1, n_levels - 1)
        elif t == "noul":
            v = float(ans["noul"])
        else:
            continue
        w = float(weights.get(qid, 1.0))
        parts[qid] = {"value": round(v, 4), "weight": w}
        acc += v * w
        total_w += w
    score = acc / total_w if (normalize and total_w) else acc
    return {"score": round(float(score), 4), "components": parts, "answers": answers}


def self_consistency(
    agent,
    state,
    question: Dict,
    variants: Sequence[str],
    qid: str = "q",
    agreement_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Ask the same question several ways; report agreement and flag disagreement.

    ``variants`` are alternative ``instructions`` phrasings. Returns the majority answer,
    the agreement ratio, and ``needs_review`` when agreement falls below the threshold.
    """
    if not variants:
        raise ValueError("self_consistency() needs at least one instruction variant")
    questions = {
        "%s__v%d" % (qid, i): dict(question, instructions=v) for i, v in enumerate(variants)
    }
    result = agent.system_one(state, questions)
    answers = result["answers"]
    values = [_answer_value(a) for a in answers.values()]

    counts: Dict[Any, int] = {}
    for v in values:
        key = round(v, 2) if isinstance(v, float) else v
        counts[key] = counts.get(key, 0) + 1
    majority, n = max(counts.items(), key=lambda kv: kv[1])
    agreement = n / len(values)

    mean_conf = sum(_confidence(a) for a in answers.values()) / len(answers)
    return {
        "value": majority,
        "agreement": round(agreement, 4),
        "needs_review": agreement < agreement_threshold,
        "confidence": round(mean_conf, 4),
        "variants": answers,
    }


def rerank(
    agent,
    candidates: Sequence[Any],
    question: Dict,
    state_fn: Optional[Callable[[Any], Any]] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[Any, float]]:
    """Score each candidate independently and return them sorted best-first.

    ``question`` should be a ``score`` or ``noul`` question expressing relevance/quality.
    ``state_fn`` maps a candidate to the state passed to the model (identity by default).
    """
    if question.get("type") == "choice":
        raise QuestionError("rerank() requires a 'score' or 'noul' question, not 'choice'")
    state_fn = state_fn or (lambda c: c)
    scored: List[Tuple[Any, float]] = []
    for cand in candidates:
        ans = agent.system_one(state_fn(cand), {"rank": question})["answers"]["rank"]
        scored.append((cand, float(_answer_value(ans))))
    scored.sort(key=lambda cs: cs[1], reverse=True)
    return scored[:top_k] if top_k else scored


def cascade(
    agent,
    state,
    stages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run staged questions, stopping early when a stage's gate fails.

    Each stage is ``{"questions": {...}, "gate": callable(answers) -> bool}``. Cheap
    filters run first; later stages only run if every earlier gate passed.
    """
    collected: Dict[str, Any] = {}
    for i, stage in enumerate(stages):
        questions = stage.get("questions")
        if not questions:
            raise QuestionError("cascade stage %d has no 'questions'" % i)
        answers = agent.system_one(state, questions)["answers"]
        collected.update(answers)
        gate = stage.get("gate")
        if gate is not None and not gate(collected):
            return {"answers": collected, "stopped_at": i, "completed": False}
    return {"answers": collected, "stopped_at": None, "completed": True}


def fan_out(
    agent,
    states: Sequence[Any],
    questions: Dict[str, Dict],
) -> List[Dict[str, Any]]:
    """Evaluate the same question set over many states, one call per state."""
    return [agent.system_one(s, questions)["answers"] for s in states]


def two_stage_choice(
    agent,
    state,
    instructions: str,
    options: Dict[str, Any],
    group_size: int = 16,
    group_of: Optional[Callable[[str, Any], str]] = None,
    qid: str = "choice",
) -> Dict[str, Any]:
    """Choose among more options than fit in one head, via coarse-then-fine selection.

    Stage 1 picks a group, stage 2 picks within it, so the per-call option count stays
    near ``sqrt(n)`` instead of ``n``. This lifts the practical ceiling well past what a
    single head can hold (see ``head_max_len``).

    ``group_of(name, criterion) -> group label`` controls grouping; by default options are
    chunked into consecutive blocks of ``group_size``.

    **Pass ``group_of`` whenever you can.** Stage 1 can only be as good as its group
    descriptions, and arbitrary chunks produce labels with nothing to discriminate on. On a
    100-option routing benchmark, semantic grouping scored 0.65 against 0.23 for default
    chunking (single-call, where it still fits, scored 0.75). Default chunking removes the
    ceiling but costs real accuracy; semantic grouping removes it for much less.

    Returns the winning option plus both stages' answers. Confidence is the product of the
    two stages -- a fine-grained pick is only as trustworthy as the group that contained it.
    """
    if not options:
        raise InvalidCriteriaError("two_stage_choice() needs at least one option")
    if group_size < 2:
        raise ValueError("group_size must be >= 2, got %r" % group_size)

    items = list(options.items())
    groups: Dict[str, Dict[str, Any]] = {}
    if group_of is not None:
        for name, crit in items:
            groups.setdefault(str(group_of(name, crit)), {})[name] = crit
    else:
        for i in range(0, len(items), group_size):
            block = items[i : i + group_size]
            label = "group_%d" % (i // group_size)
            groups[label] = dict(block)

    # A single group means there is nothing to narrow -- ask directly.
    if len(groups) == 1:
        only = next(iter(groups.values()))
        ans = agent.system_one(
            state, {qid: {"type": "choice", "instructions": instructions, "criteria": only}}
        )["answers"][qid]
        return {
            "choice": ans["choice"],
            "confidence": _confidence(ans),
            "group": next(iter(groups)),
            "stage1": None,
            "stage2": ans,
        }

    # Group descriptions must stay short: they share head_max_len with the state, and a
    # description listing every member crowds the state out of the window entirely.
    def _summarize(members: Dict[str, Any]) -> str:
        names = list(members)
        shown, budget = [], 96
        for nm in names:
            if sum(len(s) + 2 for s in shown) + len(nm) > budget:
                break
            shown.append(nm)
        text = ", ".join(shown)
        if len(shown) < len(names):
            text += ", and %d more" % (len(names) - len(shown))
        return text

    group_criteria = {label: _summarize(members) for label, members in groups.items()}

    # Too many groups to ask about at once: recurse so the group question itself is staged.
    if len(group_criteria) > group_size:
        outer = two_stage_choice(
            agent, state, instructions, group_criteria, group_size=group_size, qid=qid
        )
        chosen_group = outer["choice"]
        stage1 = outer.get("stage2") or outer.get("stage1")
        stage1_conf = outer["confidence"]
    else:
        stage1 = agent.system_one(
            state,
            {qid: {"type": "choice", "instructions": instructions, "criteria": group_criteria}},
        )["answers"][qid]
        chosen_group = stage1["choice"]
        stage1_conf = _confidence(stage1)

    members = groups[chosen_group]
    if len(members) == 1:
        only_name = next(iter(members))
        return {
            "choice": only_name,
            "confidence": round(stage1_conf, 4),
            "group": chosen_group,
            "stage1": stage1,
            "stage2": None,
        }

    stage2 = agent.system_one(
        state, {qid: {"type": "choice", "instructions": instructions, "criteria": members}}
    )["answers"][qid]
    return {
        "choice": stage2["choice"],
        "confidence": round(stage1_conf * _confidence(stage2), 4),
        "group": chosen_group,
        "stage1": stage1,
        "stage2": stage2,
    }


def taxonomy_choice(
    agent,
    state,
    instructions: str,
    tree: Dict[str, Any],
    qid: str = "node",
    max_depth: int = 8,
) -> Dict[str, Any]:
    """Walk a nested taxonomy, choosing one child per level until reaching a leaf.

    ``tree`` maps option name -> either a description (leaf) or a nested dict (branch).
    Returns the full ``path`` plus the joint confidence across levels.
    """
    if not tree:
        raise InvalidCriteriaError("taxonomy_choice() needs a non-empty tree")

    path: List[str] = []
    steps: List[Any] = []
    joint = 1.0
    node = tree

    for _ in range(max_depth):
        criteria = {
            k: (v if isinstance(v, str) else "category containing: " + ", ".join(list(v)[:6]))
            for k, v in node.items()
        }
        ans = agent.system_one(
            state, {qid: {"type": "choice", "instructions": instructions, "criteria": criteria}}
        )["answers"][qid]
        choice_name = ans["choice"]
        path.append(choice_name)
        steps.append(ans)
        joint *= _confidence(ans)

        nxt = node.get(choice_name)
        if not isinstance(nxt, dict) or not nxt:
            break
        node = nxt

    return {
        "choice": path[-1] if path else None,
        "path": path,
        "confidence": round(joint, 4),
        "steps": steps,
    }


def chunk_state(text: str, chunk_chars: int = 1200, overlap: int = 100) -> List[str]:
    """Split long text into overlapping chunks that each fit the context window.

    A stopgap for state longer than ``max_len``: pair with :func:`map_reduce` to cover a
    whole document instead of silently truncating it.
    """
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be >= 1, got %r" % chunk_chars)
    if overlap < 0:
        raise ValueError("overlap must be non-negative, got %r" % overlap)
    text = text or ""
    if len(text) <= chunk_chars:
        return [text]
    if overlap >= chunk_chars:
        raise ValueError(
            "overlap (%r) must be smaller than chunk_chars (%r), otherwise chunking cannot advance"
            % (overlap, chunk_chars)
        )
    step = chunk_chars - overlap
    return [text[i : i + chunk_chars] for i in range(0, len(text), step) if text[i : i + chunk_chars]]


def map_reduce(
    agent,
    text: str,
    questions: Dict[str, Dict],
    chunk_chars: int = 1200,
    overlap: int = 100,
    reduce: str = "max",
) -> Dict[str, Any]:
    """Run questions over every chunk of a long document and combine the results.

    ``reduce`` is ``"max"`` (strongest signal anywhere -- the default, right for detection),
    ``"mean"`` (average), or ``"first"``. Choice answers reduce by highest confidence.
    Use this when the document exceeds the model's trained context window.
    """
    if reduce not in ("max", "mean", "first"):
        raise ValueError("reduce must be 'max', 'mean' or 'first', got %r" % reduce)

    chunks = chunk_state(text, chunk_chars, overlap)
    per_chunk = [agent.system_one(c, questions)["answers"] for c in chunks]

    combined: Dict[str, Any] = {}
    for qid in questions:
        answers = [c[qid] for c in per_chunk if qid in c]
        if not answers:
            continue
        t = answers[0]["type"]
        if t == "choice" or reduce == "first":
            best = answers[0] if reduce == "first" else max(answers, key=_confidence)
            combined[qid] = best
        else:
            values = [float(_answer_value(a)) for a in answers]
            value = max(values) if reduce == "max" else sum(values) / len(values)
            pick = (
                max(answers, key=lambda a: float(_answer_value(a)))
                if reduce == "max"
                else max(answers, key=_confidence)
            )
            combined[qid] = {
                "type": t,
                t: round(value, 4),
                "confidence": _confidence(pick),
                "action": pick["action"],
                **({"legend": pick["legend"]} if t == "score" else {}),
            }
    return {"answers": combined, "chunks": len(chunks), "per_chunk": per_chunk}


def select_tool(
    agent,
    state,
    tools: Dict[str, Any],
    instructions: str = "Which tool should be used to handle this request?",
    none_option: Optional[str] = "no_tool",
    min_confidence: float = 0.0,
    group_of: Optional[Callable[[str, Any], str]] = None,
    qid: str = "tool",
) -> Dict[str, Any]:
    """Pick the tool that should handle a request (function calling, without generation).

    ``tools`` maps tool name -> description, or an OpenAI-style schema dict
    (``{"description": ..., "parameters": {...}}``), whose description is used and whose
    schema is returned untouched for the caller to fill in.

    A ``none_option`` entry is added so the model can decline rather than being forced to
    pick; set it to ``None`` to require a choice. When confidence falls below
    ``min_confidence`` the result is marked ``should_call=False``, so a caller can escalate
    instead of invoking something on a coin flip.

    Above roughly a hundred tools the options stop fitting one head; this then routes
    through :func:`two_stage_choice`, and ``group_of`` is forwarded (pass it -- see that
    function's note on why grouping quality dominates accuracy).
    """
    if not tools:
        raise InvalidCriteriaError("select_tool() needs at least one tool")
    if none_option is not None and none_option in tools:
        raise InvalidCriteriaError(
            "none_option %r collides with a tool of the same name; pass a different "
            "none_option or rename the tool" % none_option
        )

    def _describe(spec: Any) -> str:
        if isinstance(spec, dict):
            return str(spec.get("description") or spec.get("what") or "")
        return str(spec)

    criteria = {name: _describe(spec) for name, spec in tools.items()}
    if none_option is not None:
        criteria[none_option] = "none of these tools applies to this request"

    # Enough tools to overflow the head: narrow in two stages instead of failing.
    if len(criteria) > 96:
        out = two_stage_choice(
            agent, state, instructions, criteria, group_of=group_of, qid=qid
        )
        chosen, conf = out["choice"], out["confidence"]
    else:
        ans = agent.system_one(
            state, {qid: {"type": "choice", "instructions": instructions, "criteria": criteria}}
        )["answers"][qid]
        chosen, conf = ans["choice"], _confidence(ans)

    declined = none_option is not None and chosen == none_option
    return {
        "tool": None if declined else chosen,
        "confidence": round(conf, 4),
        "should_call": bool(not declined and conf >= min_confidence),
        "schema": tools.get(chosen) if not declined else None,
        "declined": declined,
    }


def check_citation(
    agent,
    claim: str,
    source: str,
    threshold: float = 0.5,
    strong_contradiction: float = 0.95,
    qid: str = "supported",
) -> Dict[str, Any]:
    """Check whether ``source`` actually supports ``claim``.

    Asks three questions: whether the source supports the claim, whether it contradicts it,
    and whether it is even on-topic. Returns ``supported`` / ``contradicted`` / ``relevant``
    probabilities and a ``verdict`` of ``"supported"``, ``"contradicted"`` or
    ``"unsupported"``.

    **Trust ``supported`` more than ``contradicted``.** Measured on the published checkpoint
    over a small hand-built set, the support signal is reliable (0.94 and 0.99 on genuinely
    supported pairs, <0.03 otherwise). The contradiction signal is not: the model reads "does
    not support" as "contradicts", scoring an *unrelated* source at 0.996 -- indistinguishable
    from a true contradiction at 0.965. The relevance question separates most such cases and
    is used to downgrade borderline contradictions, but it cannot separate all of them.

    In practice: a ``"supported"`` verdict is dependable, and anything else means "this
    citation needs a human", without relying on the contradicted/unsupported split. Use
    :func:`check_citations` to get exactly that list.
    """
    if not claim or not str(claim).strip():
        raise QuestionError("check_citation() needs a non-empty claim")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1], got %r" % threshold)

    # Inline premise/hypothesis text, NOT a {"claim":..., "source":...} dict. The checkpoint
    # scores the same supported pair at 0.87 inline vs 0.05 as JSON fields -- it was not
    # trained to resolve backtick field references into a JSON state.
    state = "Premise: %s\n\nHypothesis: %s" % (source, claim)
    answers = agent.system_one(
        state,
        {
            qid: {
                "type": "noul",
                "instructions": "Does the premise entail the hypothesis?",
                "criteria": {
                    "true": "the premise states or directly implies the hypothesis",
                    "false": "the premise does not establish the hypothesis",
                },
            },
            "contradicted": {
                "type": "noul",
                "instructions": "Does the premise contradict the hypothesis?",
                "criteria": {
                    "true": "the premise asserts something incompatible with the hypothesis",
                    "false": "the premise does not contradict the hypothesis",
                },
            },
            "relevant": {
                "type": "noul",
                "instructions": "Is the premise about the same topic as the hypothesis?",
                "criteria": {
                    "true": "same subject matter",
                    "false": "unrelated subject matter",
                },
            },
        },
    )["answers"]

    support = float(answers[qid]["noul"])
    contradict = float(answers["contradicted"]["noul"])
    relevant = float(answers["relevant"]["noul"])

    # The checkpoint reads "does not support" as "contradicts", scoring an unrelated source at
    # 0.99 contradiction, so a bare contradiction signal cannot be trusted on its own. The
    # relevance question helps but is itself weak (a true contradiction scored 0.015), so it
    # only downgrades borderline contradictions -- a decisive one still stands on its own.
    if relevant < threshold and support < threshold and contradict < strong_contradiction:
        verdict = "unsupported"
    elif contradict >= threshold and contradict > support:
        verdict = "contradicted"
    elif support >= threshold:
        verdict = "supported"
    else:
        verdict = "unsupported"

    return {
        "verdict": verdict,
        "supported": round(support, 4),
        "contradicted": round(contradict, 4),
        "relevant": round(relevant, 4),
        "confidence": round(_confidence(answers[qid]), 4),
        "answers": answers,
    }


def check_citations(
    agent,
    pairs: Sequence[Tuple[str, str]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Check many (claim, source) pairs and summarize which ones fail.

    Returns every per-pair result plus the indices of the pairs that are not supported, so
    a caller can surface exactly the citations that need review.
    """
    results = [check_citation(agent, c, s, threshold=threshold) for c, s in pairs]
    unsupported = [i for i, r in enumerate(results) if r["verdict"] != "supported"]
    return {
        "results": results,
        "unsupported": unsupported,
        "all_supported": not unsupported,
        "n_checked": len(results),
    }
