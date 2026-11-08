"""vega inside laya: every vega feature, under laya's own names.

    import laya

    agent = laya.Agent(vegaml=True)                 # the 0.8B universal reader + the engine
    agent = laya.load(vegaml=True)                  # the same
    agent = laya.Agent(vegaml="4b")                 # the 4B reader; or a Hub id, or a local dir

    out = agent.predict(state, questions)         # laya's shape, plus vega's fields
    out["answers"]["dept"]["choice"]              # as always
    out["answers"]["dept"]["prediction_set"]      # the conformal set at alpha
    out["answers"]["dept"]["abstain"]             # the engine's own refusal
    out["answers"]["dept"]["unbound"]             # far from every well: likely off-distribution
    out["answers"]["dept"]["route"]               # "auto" or "review", decided for you

    agent.fit(rows, questions)                    # heads + temperature + conformal + band, one call
    agent.predict(image=img, questions=questions) # pictures, through the reader's vision tower
    agent.predict(audio=wav, questions=questions) # speech, transcribed then decided
    agent.predict(video=clip, questions=questions) # the sound track if it carries speech, frames if not
    agent.predict_long(very_long_state, questions)   # 73k tokens, no windowing
    agent.save_calibration("fit/")               # heads and calibration, reloadable
    laya.vega.Router(fast=laya.Agent(), ...)     # model switching: laya for short English text,
                                                  # vega for everything else

**What the argument buys.** laya's own checkpoints are ModernBERT encoders: fast, small, text
only, with a temperature-scaled confidence and no notion of "I have not seen anything like
this". ``vegaml=True`` swaps the engine underneath for vegaml's: a frozen decoder reader with a
small physics engine on top, which adds a conformal prediction set, an abstain flag, an
out-of-distribution flag (``unbound``), heads fitted on your labels over the frozen features,
a calibration that is refit on your rows rather than shipped, 73k tokens of context, and the
same typed interface over images and speech. Everything is reachable through the methods laya
already has, so a caller switches by changing one argument.

**What it costs.** A 0.8B reader against a ~150M encoder: roughly five times the latency on
plain text (measured 72 ms a decision against 16 ms batched on a T4), two to four gigabytes
of memory. vegaml ships with laya since 0.4.23 (``pip install -U laya``). Core laya
is untouched when the argument is not given, which is why this lives in its own module and is
imported only on demand.

**Automatic thresholding, which is the question this module exists to answer.** A caller never
sets a confidence number. ``route`` is decided per answer, in this order:

1. If a band has been fitted (``fit`` or ``calibrate`` ran on labelled rows), the band's own
   line for that question decides: it was placed where precision holds at
   ``target_precision`` (default 0.95) on your rows, with a 95% Wilson bound to spare.
2. Otherwise the engine's own machinery decides, no labels required: ``review`` when it
   abstains at the configured risk, or when the conformal set at ``alpha`` (default 0.10) holds
   more than one option; ``auto`` only when both agree. ``unbound`` is still reported on every
   answer but no longer routes: measured with the published threshold (Embedding Space study,
   2026-11-08) it fired on 71% of in-domain banking77 rows and 26% of typed-decisions rows,
   against 93% of off-topic ones, so it sent most good answers to review.

Two more reasons apply to a policy loaded with :meth:`load_calibration` (0.4.29+):

* **outbound**: the policy's routing head measures how much of a record lies outside the space
  it was trained on; past the limit Studio measured on the calibration rows, every answer of
  that record goes to ``review``. Policies exported before Studio wrote the limit carry none,
  and then this check is off.
* **options changed**: a question asked with an option the policy was never trained on is not
  calibrated (``calibrated: False`` on that answer) and goes to ``review``; the policy's
  calibration is not applied to it. It used to raise inside the calibration instead.

Both are targets, not thresholds: ``alpha`` is the miscoverage the sets are allowed, and
``target_precision`` is the precision the band is held to. Neither is a confidence a human had
to guess at, and both have defaults. The house rule behind this is measured, not stylistic: a
fitted head once returned confidence 1.000 on every one of sixteen held-out records including
both of its mistakes, while ``route`` on the same records was still right, because route comes
from the conformal set and the band rather than from the top-1 probability. **Gate on
``route``, never on ``confidence``.** laya's ``min_confidence`` is still accepted for
compatibility and applied on top when given, but nothing here needs it.

Two vegaml facts this module is built around, both of which pass every test until they
matter: a head-served answer carries no ``abstain`` and no ``prediction_set`` of its own, so
the fitted path gets its route from the band; and ``vegaml.calibrate.calibrate`` returns a
Calibration that nothing applies unless the caller does, so :meth:`predict` applies it on
every call itself.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .agent import Agent
from .hooks import PredictContext, compose_hooks, dispatch, validate_timeout

__all__ = ["VegaAgent", "Router", "is_available", "HEAD_ARRAYS"]

#: what a fitted TTT head is, in full: standardisation, projection and the linear readout.
#: vegaml gives the heads no save of their own, so this module writes these arrays itself.
HEAD_ARRAYS = ("mu", "sd", "c", "P", "s", "W", "b")

_INSTALL = ("vegaml=True needs the vegaml package, which needs Python 3.10 or newer: "
            "pip install -U laya  (vegaml ships with laya 0.4.23 and newer; or: pip install vegaml)")


def is_available() -> bool:
    """Whether the vega engine can be imported in this interpreter."""
    try:
        import vegaml  # noqa: F401
    except ImportError:
        return False
    return True


# ----------------------------------------------------------------------------- translation
def _to_vega_questions(questions: Mapping[str, Mapping]) -> Dict[str, Dict]:
    """laya's question map in vegaml's vocabulary.

    The two schemas are the same typed-decision design and differ in one word: laya's yes/no
    is ``noul``, vegaml's is ``boolean``. laya's noul-only ``labels`` (the two sides named)
    becomes vegaml's boolean ``criteria``, which is the same information under the other key.

    Each question goes through :meth:`laya.agent.Agent._check_question` first. The translation
    reads ``q["type"]`` / ``q["instructions"]`` / ``q["criteria"]`` directly, so a malformed
    definition used to surface as a bare ``KeyError`` or ``TypeError: string indices must be
    integers`` that names neither the question nor what to fix, on ``fit`` as well as on
    ``predict``.
    """
    out = {}
    for qid, q in questions.items():
        Agent._check_question(qid, q)
        t = q["type"]
        vq: Dict[str, Any] = {"instructions": q["instructions"]}
        if t == "noul":
            vq["type"] = "boolean"
            crit = q.get("criteria")
            lab = q.get("labels")
            if isinstance(crit, dict):
                vq["criteria"] = crit
            elif isinstance(lab, dict):
                vq["criteria"] = {"false": lab.get("false", "no"), "true": lab.get("true", "yes")}
        elif t == "score":
            vq["type"] = "score"
            vq["criteria"] = list(q["criteria"])
        elif t == "choice":
            vq["type"] = "choice"
            crit = q["criteria"]
            vq["criteria"] = (dict(crit) if isinstance(crit, dict)
                              else {str(c): None for c in crit})
        else:
            raise ValueError("question %r has unknown type %r (choice, score or noul)" % (qid, t))
        out[qid] = vq
    return out


def _labels_for(q: Mapping) -> List[str]:
    """The option ids a question's answer spans, in laya's own spelling."""
    t = q["type"]
    if t == "noul":
        return ["false", "true"]
    if t == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    crit = q["criteria"]
    return [str(k) for k in (crit if isinstance(crit, dict) else crit)]


def _engine_number(value: Any, where: str) -> float:
    """One number off the engine's answer, refused rather than rounded when it is not one.

    ``float()`` reads ``True`` as 1.0 and waves ``nan`` and ``inf`` through, and every comparison
    against a NaN answers False -- so a NaN ``answer_confidence`` clears ``min_confidence``
    untouched (``nan < 0.9`` is False) and the answer leaves with ``route: "auto"``, which is the
    one field this module tells callers to gate on. The same reasoning is already the rule twice
    over: :func:`laya.train.target_from_gold` raises "probabilities must be finite and
    non-negative" for a training row, and
    :func:`laya.integrations._guard.answer_number` refuses a bool and a non-finite for the same
    gate on the remote-body path. A corrupt distribution is not a decision, and rounding it is
    how it becomes one.
    """
    if isinstance(value, bool):
        raise ValueError("%s is %r: the engine's numbers must be finite numbers, not a bool"
                         % (where, value))
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("%s is %r: the engine's numbers must be finite numbers"
                         % (where, value)) from exc
    if not math.isfinite(number):
        raise ValueError("%s is %r: the engine's numbers must be finite numbers" % (where, value))
    return number


def _entropy_confidence(probs: Mapping[str, float]) -> float:
    """``1 - H(p)/log(k)``: what laya reports as ``confidence``. Not an error rate.

    Kept so a laya caller reading ``confidence`` sees the quantity laya has always put there;
    the calibrated one is ``answer_confidence`` (max p), on vega exactly as on laya.
    """
    vals = [v for v in probs.values() if v > 0]
    k = len(probs)
    if k < 2:
        return 1.0
    h = -sum(v * math.log(v) for v in vals)
    return round(max(0.0, min(1.0, 1.0 - h / math.log(k))), 4)


def _gold_label(q: Mapping, gold: Any) -> str:
    """A training row's answer as the label vegaml's fit wants.

    Accepts laya's three shapes for an answer: a bare value, ``{"label": value}`` (the
    typed-decisions dataset), or ``{"probabilities": {...}}`` where the argmax is taken.

    A value that cannot be placed on this question is refused, as
    :func:`laya.train.target_from_expected` refuses it for the same rows on laya's own fine-tune
    path. The quiet one is a yes/no written as a float: ``str(1.0)`` is ``"1.0"``, which is not
    one of the true-words, so the row landed on the **false** side of its own label and the head
    learned the reverse of what was labelled. A score level outside the question or between two
    levels, and a choice label the question never offered, are the same class of mistake: the
    fit takes the string, so nothing downstream can tell it from a real answer.
    """
    if isinstance(gold, dict):
        if "label" in gold:
            gold = gold["label"]
        elif "probabilities" in gold and gold["probabilities"]:
            gold = max(gold["probabilities"], key=gold["probabilities"].get)
        elif "noul" in gold:
            gold = bool(float(gold["noul"]) >= 0.5)
    t = q["type"]
    if t == "noul":
        if isinstance(gold, bool):
            return "true" if gold else "false"
        if isinstance(gold, (int, float)):
            if gold in (0, 0.0):
                return "false"
            if gold in (1, 1.0):
                return "true"
        elif isinstance(gold, str):
            s = gold.strip().lower()
            if s in ("true", "yes", "1", "t", "y"):
                return "true"
            if s in ("false", "no", "0", "f", "n"):
                return "false"
        raise ValueError("a noul gold must be true/false, yes/no or 0/1, got %r" % (gold,))
    if t == "score":
        crit = list(q["criteria"])
        if isinstance(gold, bool):
            raise ValueError("a score gold is a level index or a level description, not %r" % (gold,))
        if isinstance(gold, str) and not gold.isdigit():
            if gold not in crit:
                raise ValueError("a score gold must be a level index in [0, %d] or one of %r, got %r"
                                 % (len(crit) - 1, crit, gold))
            return str(crit.index(gold))
        try:
            level = float(gold)
        except (TypeError, ValueError, OverflowError):
            level = None
        if level is None or not level.is_integer():
            raise ValueError("a score gold must name one whole level of %r, got %r" % (crit, gold))
        if not 0 <= level <= len(crit) - 1:
            raise ValueError("a score gold must be a level index in [0, %d], got %r"
                             % (len(crit) - 1, gold))
        return str(int(level))
    labels = _labels_for(q)
    if str(gold) in labels:
        return str(gold)
    raise ValueError("a choice gold must be one of the question's options %r, got %r" % (labels, gold))


def _fold(answer: Mapping, q: Mapping) -> Dict[str, Any]:
    """One vegaml answer in laya's shape, with vega's extra fields kept.

    The engine keys a boolean's probabilities ``"False"``/``"True"`` and a head keys them
    ``"false"``/``"true"``; laya spells them lower case, so the keys are folded here once and
    scoring code downstream never sees two spellings. Not folding this is how a benchmark read
    every yes/no task at 0.000 for a day.
    """
    t = q["type"]
    probs = {str(k): _engine_number(v, "the probability of %r" % (k,))
             for k, v in (answer.get("probs") or {}).items()}
    if t == "noul":
        low = {k.lower(): v for k, v in probs.items()}
        probs = {"false": low.get("false", 0.0), "true": low.get("true", 0.0)}
    pset = answer.get("prediction_set")
    if pset is not None:
        pset = [str(p).lower() if t == "noul" else str(p) for p in pset]
    out: Dict[str, Any] = {
        "type": t,
        "probabilities": {k: round(v, 6) for k, v in probs.items()},
        "answer_confidence": (round(_engine_number(answer["confidence"], "the engine's confidence"), 6)
                              if answer.get("confidence") is not None
                              else (round(max(probs.values()), 6) if probs else None)),
        "confidence": _entropy_confidence(probs) if probs else None,
        "prediction_set": pset,
        "abstain": answer.get("abstain"),
        "unbound": answer.get("unbound"),
        "readout": answer.get("readout", "engine"),
    }
    if t == "noul":
        out["noul"] = round(probs.get("true", 0.0), 6)
    elif t == "score":
        best = max(probs, key=probs.get) if probs else None
        level = answer.get("score")
        out["score"] = (int(_engine_number(level, "the engine's score")) if level is not None
                        else (int(best) if best else None))
        if answer.get("expected_score") is not None:
            out["expected_score"] = round(_engine_number(answer["expected_score"],
                                                         "the engine's expected score"), 4)
        out["legend"] = {str(i): c for i, c in enumerate(q["criteria"])}
    else:
        ch = answer.get("choice")
        out["choice"] = str(ch) if ch is not None else (max(probs, key=probs.get) if probs else None)
    for k in ("settle_steps", "energy", "adapter", "gate", "view_agreement", "cv_accuracy"):
        if k in answer:
            out[k] = answer[k]
    return out


# ----------------------------------------------------------------------------- the agent
class VegaAgent(Agent):
    """``laya.Agent`` with vegaml's engine underneath. Built by ``laya.Agent(vegaml=...)``.

    Subclasses :class:`laya.Agent` so ``isinstance`` checks, hooks and callers that duck-type
    a laya agent keep working, but it does **not** run ``Agent.__init__``: there is no
    ModernBERT checkpoint to load. Everything a laya caller touches is reimplemented here on
    top of ``vegaml.Vega``.
    """

    _backend = "vega"

    def __new__(cls, *args, **kwargs):
        # Agent.__new__ dispatches here on vegaml=...; coming back through it would recurse.
        return object.__new__(cls)

    def __init__(self, model_id_or_path: Optional[str] = None, device: Optional[str] = None,
                 token: Optional[str] = None, *, vegaml: Union[bool, str, None] = True,
                 backbone: Optional[str] = None, max_len: Optional[int] = None,
                 alpha: float = 0.1, risk: Union[str, float] = "0.1",
                 target_precision: float = 0.95, folds: int = 5,
                 mode: str = "auto", revision: Optional[str] = None,
                 hooks=None, on_predict_start=None, on_predict_end=None,
                 hooks_raise: bool = True, hooks_timeout: Optional[float] = None, **ignored):
        """Load vega.

        ``vegaml`` is the argument that got us here: ``True`` loads the default 0.8B reader,
        a string is a size (``"4b"``), a Hub repository or a local directory, exactly what
        ``vegaml.load`` takes. ``model_id_or_path`` is accepted for positional compatibility
        with ``laya.Agent`` and used when ``vega`` is just ``True``. ``backbone`` names the
        reader inside a universal checkpoint. ``alpha`` is the conformal miscoverage,
        ``risk`` the abstention risk level the checkpoint was calibrated at, and
        ``target_precision`` what :meth:`fit` holds the band to.
        """
        try:
            import vegaml as _vegaml_pkg
        except ImportError as exc:
            raise ImportError(_INSTALL) from exc
        from .hooks import normalise_hooks, validate_timeout

        which = vegaml if isinstance(vegaml, str) and vegaml else model_id_or_path
        self.vega = _vegaml_pkg.load(which, device=device, backbone=backbone, max_len=max_len,
                                mode=mode, revision=revision, token=token)
        self.model_id = self.model_id_or_path = which or "nandakishorm/vega-U"
        self.subfolder = backbone
        self.alpha = str(alpha)
        self.risk = str(risk)
        self.target_precision = float(target_precision)
        self.folds = int(folds)
        self.calibration = None          # a vegaml Calibration, applied on every predict
        self._novelty = None             # (routing question, limit) from a loaded policy
        self.fit_report: Dict[str, Any] = {}
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
        self.hooks_raise = bool(hooks_raise)
        self.hooks_timeout = None if hooks_timeout is None else validate_timeout(hooks_timeout)
        self._lock = threading.RLock()
        self.device = getattr(self.vega.engine, "device", device)
        cfg = getattr(self.vega.engine, "cfg", {}) or {}
        self.max_len = int(cfg.get("max_len", max_len or 0) or 0)

    # ------------------------------------------------------------------ introspection
    def __repr__(self) -> str:
        heads = sorted(self.vega.heads)
        return "VegaAgent(%s, max_len=%s, heads=%d, calibrated=%s)" % (
            self.model_id, self.max_len, len(heads), self.calibration is not None)

    @property
    def context_tokens(self) -> int:
        """How many tokens one :meth:`predict` reads before truncating. 73k on the shipped
        universal reader; laya's own checkpoints stop at 512 or 1,024 and window past that."""
        return self.max_len

    # ------------------------------------------------------------------ predicting
    def predict(self, state: Any = None, questions: Optional[Mapping] = None,
                lang: Optional[str] = None, *, image: Any = None, audio: Any = None,
                video: Any = None, frames: int = 4,
                mode: Optional[str] = None, views: int = 1, max_len: Optional[int] = None,
                head_max_len: Optional[int] = None, min_confidence: Optional[float] = None,
                asr: Optional[str] = None, language: Optional[str] = None,
                sampling_rate: Optional[int] = None, hooks=None, on_predict_start=None,
                on_predict_end=None, hooks_raise: Optional[bool] = None,
                hooks_timeout: Optional[float] = None, **ignored) -> Dict[str, Any]:
        """Answer typed questions about text, an image, a recording or a clip, in laya's shape.

        One of ``state``, ``image``, ``audio`` or ``video``. The modality is chosen by which you passed,
        which is the whole of "modal switching": the same questions, the same answer shape,
        the same route. ``lang`` is accepted for compatibility and recorded; the universal
        reader is multilingual and needs no checkpoint switch.

        ``max_len`` and ``head_max_len`` are laya's option-budget knobs; vega truncates
        options at 32 tokens and instructions at 256 by itself, so they are recorded and not
        applied. ``min_confidence`` is applied on top of the automatic route when given, the
        way laya's gate applies it, and nothing here needs it.

        ``video`` is a path or raw file bytes. Its sound track is transcribed and, when the
        transcript holds real words, decided through the text path with everything that path
        has (heads, calibration, set, route); otherwise ``frames`` evenly spaced frames are
        tiled into one image and decided through the image path. ``out["video"]`` says which,
        and why. See :mod:`laya.video`.

        The hook lifecycle is the one :meth:`laya.Agent.predict_batch` runs: ``on_predict_start``
        fires before the decision (``ctx.skip(...)`` short-circuits it, and a hook may rewrite
        ``ctx.questions``), ``on_error`` on a failure, ``on_predict_end`` on the finished
        answer and may rewrite ``ctx.results``. The video path decides recursively through
        this method, so its hooks fire there too. ``hooks_raise`` and ``hooks_timeout``
        override the agent's for this call, as on every laya surface.
        """
        if questions is None:
            raise ValueError("predict needs `questions`")
        given = sum(x is not None for x in (state, image, audio, video))
        if given != 1:
            raise ValueError("predict takes exactly one of state=, image=, audio=, video=; got %d"
                             % given)
        active = compose_hooks(self.hooks, hooks, on_predict_start, on_predict_end)
        raise_errors = self.hooks_raise if hooks_raise is None else bool(hooks_raise)
        timeout = self.hooks_timeout if hooks_timeout is None else validate_timeout(hooks_timeout)
        ctx = PredictContext(states=state, questions=questions, model=self.model_id, agent=self,
                             max_len=max_len, head_max_len=head_max_len)
        try:
            dispatch(active, "on_predict_start", ctx, raise_errors=raise_errors,
                     lock=self._hooks_lock, timeout=timeout)
            if ctx.results is not None:
                out = ctx.results
                if isinstance(out, list) and len(out) == 1:
                    out = out[0]        # ctx.skip()'s whole-call list, as predict_long takes it
                if isinstance(out, dict):
                    ctx.usage = out.get("usage")
            else:
                out = self._decide(ctx.states, ctx.questions, lang=lang, image=image,
                                   audio=audio, video=video, frames=frames, mode=mode,
                                   views=views, min_confidence=min_confidence, asr=asr,
                                   language=language, sampling_rate=sampling_rate)
                ctx.results = out
                ctx.usage = out.get("usage")
        except BaseException as exc:
            ctx.error = exc
            try:
                dispatch(active, "on_error", ctx, raise_errors=raise_errors,
                         lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                # A failing on_error hook must not hide the failure that triggered it.
                exc.__context__ = hook_exc
            raise
        finally:
            ctx.elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
            try:
                dispatch(active, "on_predict_end", ctx, raise_errors=raise_errors,
                         lock=self._hooks_lock, timeout=timeout)
            except BaseException as hook_exc:
                # End hooks run on the failure path too; do not let one mask the real error.
                if ctx.error is not None:
                    ctx.error.__context__ = hook_exc
                else:
                    raise
        return ctx.results

    def _decide(self, state: Any, questions: Mapping, *, lang: Optional[str], image: Any,
                audio: Any, video: Any, frames: int, mode: Optional[str], views: int,
                min_confidence: Optional[float], asr: Optional[str],
                language: Optional[str], sampling_rate: Optional[int]) -> Dict[str, Any]:
        """The decision itself, without the hook lifecycle around it."""
        if video is not None:
            return self._predict_video(video, questions, frames=frames, asr=asr, language=language,
                                       mode=mode, views=views, min_confidence=min_confidence,
                                       lang=lang)
        if not questions:
            return {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0},
                    "latency_ms": 0.0, "modality": "text"}
        vq = _to_vega_questions(questions)
        started = time.perf_counter()
        with self._lock:
            novelty = getattr(self, "_novelty", None)
            if novelty and novelty[0] in self.vega.heads:
                self.vega.heads[novelty[0]][0].last_residual = None
            if image is not None:
                try:
                    raw = self.vega.decide_image(image, vq, mode=mode, alpha=self.alpha, risk=self.risk)
                except ImportError as exc:
                    # the reader's image processor wants Pillow and torchvision, which core laya
                    # does not carry; say which extra, not which transformers backend
                    raise ImportError("images and video need Pillow and torchvision: "
                                      "pip install 'laya[vision]'  (%s)" % exc) from exc
                modality = "image"
            elif audio is not None:
                try:
                    raw = self.vega.decide_audio(audio, vq, asr=asr, language=language,
                                                 sampling_rate=sampling_rate, mode=mode,
                                                 alpha=self.alpha, views=views)
                except ImportError as exc:
                    raise ImportError("an audio file needs soundfile to decode: pip install 'laya[audio]'; "
                                      "a numpy waveform with sampling_rate= needs nothing extra  (%s)" % exc) from exc
                modality = "audio"
            else:
                raw = self.vega.decide(state, vq, mode=mode, alpha=self.alpha, views=views)
                modality = "text"
            stale: Dict[str, List[str]] = {}
            if self.calibration is not None:
                stale = _stale_for_calibration(raw, self.calibration)
                raw = _apply_calibration(raw, self.calibration, stale)
            outbound = self._read_novelty()
        answers = {}
        for qid, q in questions.items():
            a = _fold(raw["answers"][qid], q)
            a.update(self._route(dict(raw["answers"][qid], _qid=qid), a, min_confidence))
            if self.calibration is not None:
                a["calibrated"] = qid in self.calibration.questions and qid not in stale
            if qid in stale:
                a["route"] = "review"
                a["route_reason"] = ("options changed since this policy was trained (%s not in it), "
                                     "so it is not calibrated; train it again"
                                     % ", ".join(stale[qid]))
            if outbound is not None:
                a["outbound"] = outbound
                a["route"] = "review"
                a["route_reason"] = ("outbound: %.3f of this record lies outside the policy's space, "
                                     "over the limit %.3f; %s" % (outbound["residual"], outbound["limit"],
                                                                  a["route_reason"]))
            answers[qid] = a
        out = {
            "answers": answers,
            "usage": {"input_tokens": int(raw.get("n_tokens") or 0), "output_tokens": 0},
            "latency_ms": round(1000 * (time.perf_counter() - started), 1),
            "modality": modality,
            "mode": raw.get("mode", mode or self.vega.mode),
            "views": raw.get("views", views),
            "calibrated": self.calibration is not None,
        }
        if lang is not None:
            out["lang"] = lang
        for k in ("transcript", "asr"):
            if k in raw:
                out[k] = raw[k]
        return out

    system_one = predict

    def predict_image(self, image: Any, questions: Mapping, **kw) -> Dict[str, Any]:
        return self.predict(image=image, questions=questions, **kw)

    def predict_video(self, video: Any, questions: Mapping, **kw) -> Dict[str, Any]:
        return self.predict(video=video, questions=questions, **kw)

    def _predict_video(self, video: Any, questions: Mapping, *, frames: int = 4,
                       asr: Optional[str] = None, language: Optional[str] = None,
                       **kw) -> Dict[str, Any]:
        """The clip split into what the engine can read, and the path chosen by its content.

        The speech test is made on the transcript the audio path produced, not before it, so a
        sound track that is loud but wordless (music, noise) costs one audio decision and then
        falls through to the frames. That is cheaper than a second transcriber and keeps one
        definition of "speech" for the whole library: :func:`laya.video.has_speech`.
        """
        from . import video as _video

        started = time.perf_counter()
        path, tmp = _video.as_path(video)
        try:
            info = _video.probe(path)
            wave = _video.extract_audio(path)
            out = None
            detail: Dict[str, Any] = {"duration_s": round(info["duration_s"], 3),
                                      "has_audio": bool(info["has_audio"])}
            if _video.has_speech(wave):
                heard = self.predict(audio=wave, questions=questions, sampling_rate=_video.TARGET_SR,
                                     asr=asr, language=language, **kw)
                transcript = heard.get("transcript") or ""
                if _video.has_speech(wave, transcript):
                    out = heard
                    detail.update(path="speech", transcript=transcript, asr=heard.get("asr"))
                else:
                    detail["transcript_rejected"] = transcript
            if out is None:
                if not info.get("has_video", True):
                    raise ValueError("the clip has no video stream and no speech on its sound track; "
                                     "there is nothing to decide on")
                sampled, times = _video.sample_frames(path, frames, info["duration_s"])
                sheet = _video.contact_sheet(sampled)
                out = self.predict(image=sheet, questions=questions,
                                   **{k: v for k, v in kw.items() if k != "views"})
                out.pop("transcript", None); out.pop("asr", None)
                detail.update(path="frames", frame_times=[round(t, 3) for t in times],
                              frames=len(sampled))
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        out["modality"] = "video"
        out["video"] = detail
        out["latency_ms"] = round(1000 * (time.perf_counter() - started), 1)
        return out

    def predict_image_text(self, image: Any, questions: Mapping, **kw) -> Dict[str, Any]:
        """Parity check: the vision path with the image withheld, to see what the picture
        itself contributed."""
        raw = self.vega.decide_image_text(image, _to_vega_questions(questions), **kw)
        return {"answers": {qid: _fold(raw["answers"][qid], q) for qid, q in questions.items()},
                "modality": "image_withheld"}

    def predict_audio(self, audio: Any, questions: Mapping, **kw) -> Dict[str, Any]:
        return self.predict(audio=audio, questions=questions, **kw)

    def predict_batch(self, states: Sequence[Any], questions: Mapping, batch_size: Optional[int] = None,
                      lang: Optional[str] = None, **kw) -> List[Dict[str, Any]]:
        """Every state, one decide call each. vega packs a state's questions into one pass,
        not many states, so this is a loop with laya's signature rather than a faster path.
        ``batch_size`` is accepted and ignored."""
        return [self.predict(s, questions, lang=lang, **kw) for s in states]

    def predict_long(self, state: Any, questions: Mapping, window: Optional[int] = None,
                     stride: Optional[int] = None, aggregate: str = "auto", **kw) -> Dict[str, Any]:
        """A long state, read whole.

        laya's own checkpoints stop at 512 or 1,024 tokens and this method windows past that.
        The universal reader holds 73k tokens, so the state is read in one pass up to that and
        the per-window aggregation never happens; ``window``, ``stride`` and ``aggregate`` are
        accepted for compatibility and recorded. Past the reader's own limit the engine
        truncates, and ``usage.truncated`` says so, which is the one case where the caller may
        still want to window and vote.
        """
        out = self.predict(state, questions, **kw)
        n = int(out["usage"].get("input_tokens") or 0)
        out["usage"]["windows"] = 1
        out["usage"]["context_tokens"] = self.max_len
        out["usage"]["truncated"] = bool(self.max_len and n >= self.max_len)
        out["long"] = {"window": window, "stride": stride, "aggregate": aggregate,
                       "applied": False, "why": "read whole, within %d tokens" % self.max_len}
        return out

    def _read_novelty(self) -> Optional[Dict[str, Any]]:
        """The routing head's residual for the record just read, when it passed the limit."""
        novelty = getattr(self, "_novelty", None)
        if not novelty or novelty[0] not in self.vega.heads:
            return None
        r = getattr(self.vega.heads[novelty[0]][0], "last_residual", None)
        if r is None or r <= novelty[1]:
            return None
        return {"residual": round(float(r), 4), "limit": round(float(novelty[1]), 4),
                "question": novelty[0]}

    # ------------------------------------------------------------------ the route
    def _route(self, raw_answer: Mapping, folded: Mapping, min_confidence: Optional[float]) -> Dict:
        """``route`` and why, decided without a user-set number. See the module docstring."""
        band_q = None
        cal = getattr(self, "calibration", None)
        if cal is not None and getattr(cal, "band", None) is not None:
            band_q = cal.band.questions.get(raw_answer.get("_qid", ""))
        unreachable = bool(band_q) and band_q.get("reachable") is False
        if "route" in raw_answer and raw_answer["route"] in ("auto", "review") and not unreachable:
            route, why = raw_answer["route"], "band: " + str(raw_answer.get("route_reason", ""))
        else:
            reasons = []
            if folded.get("abstain"):
                reasons.append("the engine abstains at risk %s" % self.risk)
            pset = folded.get("prediction_set")
            if pset is not None and len(pset) > 1:
                reasons.append("conformal set at alpha %s holds %d options" % (self.alpha, len(pset)))
            if pset is None and folded.get("abstain") is None:
                # a head-served answer with no band fitted: nothing here can vouch for it
                reasons.append("head answer with no fitted band; fit() or calibrate() first")
            route = "review" if reasons else "auto"
            why = "; ".join(reasons) if reasons else "conformal singleton, not abstaining"
            if unreachable:
                # The line was asked to hold target_precision with a 95% lower bound and the
                # calibration rows could not certify it (12 for 12 certifies nothing). Sending
                # every answer to review would make the fitted path useless, so the conformal
                # rule decides and the reason says what the band could not do.
                why = ("band could not hold precision %s on %s calibration rows, so the conformal "
                       "rule decides: %s" % (band_q.get("target_precision"), band_q.get("examples"), why))
        if min_confidence is not None and route == "auto":
            c = folded.get("answer_confidence")
            if c is None or c < float(min_confidence):
                route, why = "review", "min_confidence %.3f not met (%s)" % (min_confidence, c)
        return {"route": route, "route_reason": why}

    def thresholds(self) -> Dict[str, Any]:
        """What decides ``route`` right now, per question, so nothing about it is hidden."""
        out: Dict[str, Any] = {"alpha": self.alpha, "risk": self.risk,
                               "target_precision": self.target_precision,
                               "policy": ("band" if self.calibration is not None
                                          else "conformal set + abstain")}
        novelty = getattr(self, "_novelty", None)
        if novelty:
            out["outbound"] = {"question": novelty[0], "limit": novelty[1]}
        if self.calibration is not None:
            band = self.calibration.band
            out["band"] = ({"default": band.default, "questions": band.questions}
                           if band is not None else None)
            out["calibration"] = {q: {k: v for k, v in c.items() if k != "labels"}
                                  for q, c in self.calibration.questions.items()}
        out["heads"] = {q: {"cv_accuracy": cv, "serving": self.vega._head_worth_using(q, {
            "type": "choice", "criteria": {l: None for l in h.labels}})}
            for q, (h, cv) in self.vega.heads.items()}
        return out

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def load_rows(path: Any, questions: Optional[Mapping] = None) -> List[Dict[str, Any]]:
        """Labelled rows from a file: ``.csv`` or ``.jsonl``.

        A CSV has a ``state`` column and one column per question id holding that row's gold
        label (an option id for ``choice``, a level index for ``score``, ``yes``/``no`` or
        ``true``/``false`` for ``noul``); an empty cell means the row carries no label for that
        question. ``examples/data/routing_labels.csv`` in the repository is the shape. JSONL
        rows are the ``{"state", "questions"?, "gold"}`` dicts :meth:`fit` takes directly.
        """
        import csv

        path = os.fspath(path)
        if path.lower().endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "state" not in reader.fieldnames:
                raise ValueError("%s needs a `state` column; the other columns are question ids" % path)
            label_cols = [c for c in reader.fieldnames if c != "state"]
            if questions:
                unknown = [c for c in label_cols if c not in questions]
                if unknown:
                    raise ValueError("column(s) %s in %s match no question id in %s"
                                     % (unknown, path, sorted(questions)))
            rows = []
            for rec in reader:
                gold = {c: rec[c].strip() for c in label_cols if (rec.get(c) or "").strip()}
                if rec.get("state", "").strip() and gold:
                    rows.append({"state": rec["state"], "gold": gold})
        return rows

    def _examples(self, rows: Any, questions: Optional[Mapping]) -> Tuple[list, Dict]:
        """laya training rows as vegaml fit examples: ``(state, {qid: label}, {qid: question})``.

        Rows are ``{"state", "questions"?, "gold"|"expected"}``: the shape ``items_from_rows``
        reads, so a corpus prepared for laya's fine-tune fits vega's heads unchanged. A row
        without its own ``questions`` uses ``questions``. A path to a ``.csv`` or ``.jsonl``
        is read with :meth:`load_rows` first.

        A gold keyed to no question id of that row is refused, as ``load_rows`` refuses an
        unknown CSV column: dropped, it made ``fit`` report success on a corpus whose labels
        never trained anything. Naming only some of the questions is fine.
        """
        if isinstance(rows, (str, os.PathLike)):
            rows = self.load_rows(rows, questions)
        examples, sheet = [], {}
        for r in rows:
            qs = r.get("questions") or questions
            if not qs:
                raise ValueError("a row has no questions and none were given")
            vq = _to_vega_questions(qs)
            sheet.update(vq)
            gold = r.get("gold") if r.get("gold") is not None else r.get("expected")
            if not isinstance(gold, Mapping):
                raise ValueError("each row needs `gold` or `expected` as {question_id: answer}")
            unknown = [qid for qid in gold if qid not in qs]
            if unknown:
                raise ValueError("gold key(s) %s match no question id in %s"
                                 % (unknown, sorted(qs)))
            labels = {qid: _gold_label(qs[qid], g) for qid, g in gold.items()}
            if labels:
                examples.append((r["state"], labels, vq))
        return examples, sheet

    #: Rows kept back from the heads so the calibration is measured on rows they never saw.
    CALIBRATION_SHARE = 0.25
    #: What vegaml needs to say anything: ten rows to calibrate, and for the heads at least
    #: ``folds`` rows of every option so stratified cross-validation can score each one.
    MIN_CALIBRATION_ROWS = 10

    @classmethod
    def min_rows(cls, questions: Mapping, folds: int = 5, calibrate: bool = True) -> int:
        """The fewest labelled rows :meth:`fit` accepts for these questions.

        Heads need ``folds`` rows per option (the widest question decides) and never fewer
        than ``2 * folds``; with ``calibrate`` another quarter is held out for the
        calibration, which needs ten rows of its own. A five-way choice therefore needs 35,
        a yes/no question 24. These are floors: the example in the README fits twelve rows
        per option, and more rows move every number in the right direction.
        """
        widest = max((len(_labels_for(q)) for q in questions.values()), default=2)
        heads = max(2 * folds, folds * widest)
        if not calibrate:
            return heads
        cal = max(cls.MIN_CALIBRATION_ROWS, math.ceil(heads * cls.CALIBRATION_SHARE / (1 - cls.CALIBRATION_SHARE)))
        return heads + cal

    @classmethod
    def _split(cls, examples: Sequence[Tuple], share: float) -> Tuple[list, list]:
        """A calibration share taken across the first question's labels in proportion, never
        fewer than ``MIN_CALIBRATION_ROWS`` in all and at least one row from every label that
        has two; deterministic, so a refit is the same split."""
        by_label: Dict[str, list] = {}
        for ex in examples:
            first = next(iter(ex[1].values()), "")
            by_label.setdefault(str(first), []).append(ex)
        target = min(len(examples) // 2, max(cls.MIN_CALIBRATION_ROWS, round(len(examples) * share)))
        take = {lab: (1 if len(m) >= 2 else 0) for lab, m in by_label.items()}
        # the rest of the target by largest remainder of each label's proportional share
        remaining = target - sum(take.values())
        order = sorted(by_label, key=lambda lab: -(len(by_label[lab]) * share - take[lab]))
        while remaining > 0:
            moved = False
            for lab in order:
                if remaining <= 0:
                    break
                if take[lab] < len(by_label[lab]) // 2:
                    take[lab] += 1; remaining -= 1; moved = True
            if not moved:
                break
        fit_part, cal_part = [], []
        for lab, members in by_label.items():
            k = take[lab]
            cal_part.extend(members[len(members) - k:])
            fit_part.extend(members[:len(members) - k])
        return fit_part, cal_part

    def fit(self, rows: Any, questions: Optional[Mapping] = None,
            folds: Optional[int] = None, calibrate: bool = True,
            target_precision: Optional[float] = None,
            calibration_rows: Any = None) -> Dict[str, Any]:
        """Heads on your labels, then the calibration and the band, in one call.

        What laya's fine-tune does with gradient steps, this does with a small head per
        question over the frozen features: seconds on a GPU for a few hundred rows, and the
        reader is never touched. A head is kept only if it beats chance under cross-validation
        and SERVED only if it also beats the engine on the same rows, so ``report[qid]
        ["used_by_auto"]`` is the fact to read, not ``fitted``. With ``calibrate`` (the
        default) the same rows then refit a temperature and a conformal threshold per
        question and place the band, which is where the automatic ``route`` comes from.
        Nothing is scored on a fitted row by this method; held-out evaluation is yours.

        ``rows`` may also be a path to a ``.csv`` (a ``state`` column plus one column per
        question id) or a ``.jsonl``; see :meth:`load_rows`.

        The calibration is measured on rows the heads never saw. By default a quarter of the
        rows, taken evenly across the first question's labels, is held out for it; pass
        ``calibration_rows`` to use a separate set instead and fit the heads on all of
        ``rows``. Calibrating on the fitting rows is the one thing this method refuses to do:
        a head is right on its own rows, so the temperature rails to its floor, every
        confidence becomes 1.000 and the review line cannot be placed. Fewer rows than
        :meth:`min_rows` is a ``ValueError`` naming the number.
        """
        examples, sheet = self._examples(rows, questions)
        if not examples:
            raise ValueError("no trainable rows")
        qs = questions or {}
        for ex in examples:
            qs = dict(qs, **{k: v for k, v in ex[2].items() if k not in qs}) if ex[2] else qs
        folds = folds or self.folds
        cal_examples = None
        if calibrate and calibration_rows is not None:
            cal_examples, cal_sheet = self._examples(calibration_rows, questions)
            sheet = dict(sheet, **cal_sheet)
            need = self.min_rows(qs, folds, calibrate=False)
            if len(examples) < need or len(cal_examples) < self.MIN_CALIBRATION_ROWS:
                raise ValueError("fit needs at least %d rows for the heads and %d to calibrate on for "
                                 "these questions; got %d and %d"
                                 % (need, self.MIN_CALIBRATION_ROWS, len(examples), len(cal_examples)))
            fit_examples = examples
        elif calibrate:
            need = self.min_rows(qs, folds, calibrate=True)
            if len(examples) < need:
                raise ValueError("fit needs at least %d labelled rows for these questions (%d for the "
                                 "heads, the rest held out to calibrate on); got %d. Pass "
                                 "calibrate=False to fit heads alone, or label more rows."
                                 % (need, self.min_rows(qs, folds, calibrate=False), len(examples)))
            fit_examples, cal_examples = self._split(examples, self.CALIBRATION_SHARE)
        else:
            need = self.min_rows(qs, folds, calibrate=False)
            if len(examples) < need:
                raise ValueError("fit needs at least %d labelled rows for these questions; got %d"
                                 % (need, len(examples)))
            fit_examples = examples
        with self._lock:
            report = self.vega.fit(fit_examples, sheet, folds=folds)
        self.fit_report = {"heads": report, "n_rows": len(examples), "n_fit": len(fit_examples),
                           "n_calibrate": len(cal_examples) if cal_examples else 0,
                           "serving": sorted(q for q, r in report.items() if r.get("used_by_auto")),
                           "kept": sorted(q for q, r in report.items() if r.get("fitted"))}
        if calibrate:
            self.fit_report["calibration"] = self.calibrate(
                None, questions, target_precision=target_precision, _examples=(cal_examples, sheet))
        return self.fit_report

    def calibrate(self, rows: Any, questions: Optional[Mapping] = None,
                  target_precision: Optional[float] = None, alpha: Optional[float] = None,
                  _examples=None) -> Dict[str, Any]:
        """Refit the probabilities and place the band on your labelled rows.

        A temperature per question, a conformal threshold on the corrected probabilities so
        the sets cover at ``1 - alpha`` on rows like yours, and the escalation band at
        ``target_precision``. The argmax is invariant under temperature, so this moves
        confidence, sets, ``abstain`` and ``route`` and never which option wins. Sharper
        answers need :meth:`fit`. Applied on every :meth:`predict` from now on.
        """
        from vegaml.calibrate import calibrate as _calibrate

        examples, sheet = _examples or self._examples(rows, questions)
        tp = self.target_precision if target_precision is None else float(target_precision)
        a = float(self.alpha) if alpha is None else float(alpha)
        with self._lock:
            self.calibration = _calibrate(self.vega, examples, sheet, alpha=a,
                                          target_precision=tp, readout="auto")
        rep = {q: {k: r.get(k) for k in ("temperature", "qhat", "ece_before", "ece_after",
                                         "examples") if k in r}
               for q, r in self.calibration.report.items()}
        band = self.calibration.band
        return {"questions": rep, "target_precision": tp, "alpha": a,
                "n_rows": len(examples),
                "band": None if band is None else
                {q: (round(float(v["threshold"]), 4) if v.get("reachable", True) else None)
                 for q, v in band.questions.items()},
                "band_unreachable": [] if band is None else
                sorted(q for q, v in band.questions.items() if v.get("reachable") is False)}

    fit_temperatures = calibrate

    # ------------------------------------------------------------------ persistence
    def save_calibration(self, path: str) -> Dict[str, Any]:
        """Heads and calibration to ``path`` (a directory), reloadable with
        :meth:`load_calibration`. vegaml's heads have no save of their own, so the arrays are
        written here; the Calibration is vegaml's own JSON."""
        import numpy as np

        os.makedirs(path, exist_ok=True)
        arrays, meta = {}, {"heads": {}, "model": self.model_id, "alpha": self.alpha,
                            "risk": self.risk, "target_precision": self.target_precision}
        for qid, (head, cv) in self.vega.heads.items():
            if not getattr(head, "ready", False):
                continue
            for name in HEAD_ARRAYS:
                arrays["%s::%s" % (qid, name)] = np.asarray(getattr(head, name))
            meta["heads"][qid] = {"labels": [str(x) for x in head.labels],
                                  "k": int(head.k), "l2": float(head.l2),
                                  "cv": None if cv is None else float(cv),
                                  "engine_accuracy": head.engine_accuracy}
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(os.path.join(path, "heads.npz"), **arrays)
        if self.calibration is not None:
            self.calibration.save(os.path.join(path, "calibration.json"))
        # A manifest of hashes, unsigned: load_calibration checks the folder was not changed.
        from . import policy

        files = {n: open(os.path.join(path, n), "rb").read()
                 for n in ("heads.npz", "calibration.json") if os.path.exists(os.path.join(path, n))}
        try:
            engine = self.vega.engine.base_sha256()
        except Exception:
            engine = None
        with open(os.path.join(path, "manifest.json"), "wb") as f:
            f.write(policy.manifest_for(files, format="laya-policy/1", engine_sha256=engine,
                                        model=self.model_id))
        return {"path": path, "heads": sorted(meta["heads"]),
                "calibrated": self.calibration is not None}

    def load_calibration(self, path: str, verify: Union[bool, str] = True) -> Dict[str, Any]:
        """Install a trained policy from a folder or straight from Studio's downloaded ``.zip``.

        ``verify=True`` (the default) checks before anything is installed: every file has the
        SHA-256 its manifest states, a signed policy's signature verifies against a key laya
        trusts, and this agent's engine is the one the policy was trained on. Any failure raises
        :class:`laya.policy.PolicyError` naming what failed. ``verify="signed"`` also refuses a
        policy that is not signed. ``verify=False`` skips all of it.

        Returns the questions (laya's spelling, ready for :meth:`predict`) and what was verified.
        """
        from vegaml.calibrate import Calibration
        from vegaml.ttt import TTTHead

        from . import policy

        files = policy.read(path)
        report: Dict[str, Any] = {"status": "skipped"}
        if verify:
            report = policy.verify(files, require_signature=(verify == "signed"))
            try:
                actual = self.vega.engine.base_sha256()
            except Exception:  # an engine that cannot hash itself is reported, not guessed
                actual = None
            expected = policy.expected_engine(files, report)
            policy.check_engine(expected, actual)
            report = dict(report, engine="matches" if (expected and actual) else "not checked")
            report.pop("manifest", None)
        z = policy.npz(files)
        meta = json.loads(str(z["meta"]))
        guarded = _guarded_head_class(TTTHead)
        loaded, limits = [], {}
        for qid, m in meta["heads"].items():
            head = guarded(m["labels"], k=m["k"], l2=m["l2"])
            for name in HEAD_ARRAYS:
                setattr(head, name, z["%s::%s" % (qid, name)])
            head.ready = True
            head.engine_accuracy = m.get("engine_accuracy")
            self.vega.heads[qid] = (head, m.get("cv"))
            loaded.append(qid)
            if m.get("novelty_max") is not None:
                limits[qid] = float(m["novelty_max"])
        if "calibration.json" in files:
            self.calibration = Calibration.from_payload(json.loads(files["calibration.json"]))
        out: Dict[str, Any] = {"heads": loaded, "calibrated": self.calibration is not None,
                               "verified": report}
        if "questions.json" in files:
            out["questions"] = json.loads(files["questions.json"])
            self.policy_questions = out["questions"]
        # The routing question is the policy's first; Studio measures the limit on its head.
        order = list(out.get("questions") or {}) or loaded
        routing = next((q for q in order if q in limits), None)
        self._novelty = (routing, limits[routing]) if routing else None
        out["outbound"] = ({"question": routing, "limit": limits[routing]} if routing
                           else None)
        return out

    # ------------------------------------------------------------------ readers
    @staticmethod
    def attach(backbone: str, texts: Sequence[str], out_dir: str, **kw) -> Dict[str, Any]:
        """Attach your own frozen reader to the universal engine, from your own text, with no
        labels: writes a model directory that ``laya.Agent(vegaml=out_dir)`` then loads."""
        import vegaml
        return vegaml.attach(backbone, texts, out_dir, **kw)

    # laya's own calibration file is a different artefact; say so rather than mis-save one
    def fit_binning(self, *a, **kw):
        raise NotImplementedError("histogram binning is laya's own calibration; on vega the "
                                  "calibrated probabilities come from calibrate()")


def _residual(head, x) -> float:
    """The share of a record's standardised signal the head's projection does not hold: what
    Studio's outbound check measures (studio/heads.py ``residual``)."""
    import numpy as np

    Z = (np.asarray(x, dtype=np.float32).reshape(1, -1) - head.mu) / head.sd
    Zc = Z - head.c
    total = float((Zc ** 2).sum())
    if total <= 0:
        return 0.0
    inside = float(((Zc @ head.P) ** 2).sum())
    return max(0.0, 1.0 - inside / total)


def _guarded_head_class(base):
    """A head that remembers how far outside its space the last record it read was."""
    class GuardedHead(base):
        last_residual = None

        def predict(self, x, allowed=None):
            self.last_residual = _residual(self, x)
            return super().predict(x, allowed)

    return GuardedHead


def _stale_for_calibration(result: Mapping, cal) -> Dict[str, List[str]]:
    """Questions whose answer names an option the calibration was never fitted on, with those
    options. Applying the calibration to them raised "probability key names no option"."""
    out: Dict[str, List[str]] = {}
    for qid, ans in (result.get("answers") or {}).items():
        fitted = cal.questions.get(qid)
        if fitted is None or not ans.get("probs"):
            continue
        known = {str(o).lower() for o in fitted["labels"]}
        new = [str(k) for k in ans["probs"] if str(k).lower() not in known]
        if new:
            out[qid] = new
    return out


def _apply_calibration(result: Mapping, cal, stale: Mapping) -> Dict:
    """The calibration on every answer it covers; stale answers pass through untouched."""
    if not stale:
        return cal.apply(_pad_for_calibration(result, cal))
    answers = result["answers"]
    sub = dict(result)
    sub["answers"] = {q: a for q, a in answers.items() if q not in stale}
    done = cal.apply(_pad_for_calibration(sub, cal))
    merged = dict(done)
    merged["answers"] = {q: (answers[q] if q in stale else done["answers"][q]) for q in answers}
    return merged


def _pad_for_calibration(result: Mapping, cal) -> Dict:
    """Zero-fill any fitted option an answer never mentioned.

    ``Calibration.apply`` indexes every label it was fitted on and KeyErrors on an answer over
    a strict subset, after the fit and the calibration have both succeeded, which is the worst
    place for it. Zero is the correct value: the row did not offer that option.
    """
    out = dict(result)
    out["answers"] = {}
    for qid, ans in result["answers"].items():
        fitted = cal.questions.get(qid)
        if fitted is None or not ans.get("probs"):
            out["answers"][qid] = ans
            continue
        have = {str(k).lower() for k in ans["probs"]}
        missing = [o for o in fitted["labels"] if str(o).lower() not in have]
        if missing:
            ans = dict(ans)
            ans["probs"] = dict(ans["probs"])
            for o in missing:
                ans["probs"][o] = 0.0
        out["answers"][qid] = ans
    return out


# ----------------------------------------------------------------------------- switching
class Router:
    """Model switching: laya's fast encoder where it is enough, vega where it is not.

        router = from laya.vega import Router
        Router(fast=laya.Agent(), vega=laya.Agent(vegaml=True))
        router.predict(state, questions)            # -> picks, and says which under "routing"

    The rule, in order: an ``image``, ``audio`` or ``video`` goes to vega (the encoder has no eyes or
    ears); a state longer than the fast model's context goes to vega (73k against 512 or
    1,024); non-English or non-Latin text goes to vega (the universal reader is multilingual,
    and laya's own English checkpoint collapses on non-Latin scripts); a question with a
    fitted head on vega goes to vega (that is what the labels bought); everything else goes to
    the fast model. ``model=`` forces one. The result carries ``routing = {"model", "reason"}``
    in the shape ``laya.Router`` already returns, so a caller switching from that class reads
    nothing new.
    """

    def __init__(self, fast: Optional[Agent] = None, vega: Optional[VegaAgent] = None,
                 fast_context_tokens: int = 512, prefer_vega_when_fitted: bool = True):
        self.fast = fast
        self._vega = vega
        self.fast_context_tokens = int(fast_context_tokens)
        self.prefer_vega_when_fitted = prefer_vega_when_fitted

    @property
    def vega(self) -> VegaAgent:
        if self._vega is None:
            self._vega = VegaAgent()
        return self._vega

    def choose(self, state: Any = None, questions: Optional[Mapping] = None, *, image=None,
               audio=None, video=None, lang: Optional[str] = None) -> Tuple[str, str]:
        from .lang import is_english

        if image is not None or audio is not None or video is not None:
            return "vega", "image, audio or video: the encoder has no vision or speech path"
        if self.fast is None:
            return "vega", "no fast model attached"
        text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        if len(text) > self.fast_context_tokens * 4:          # ~4 characters a token
            return "vega", "state is longer than the fast model's %d-token context" % \
                self.fast_context_tokens
        if lang is not None:
            english = str(lang).lower() in ("en", "eng", "english")
        else:
            english = bool(is_english(text))
        if not english:
            return "vega", "non-English or non-Latin text; the universal reader is multilingual"
        if self.prefer_vega_when_fitted and questions and self._vega is not None:
            fitted = [q for q in questions if q in self._vega.vega.heads]
            if fitted:
                return "vega", "fitted heads exist for %s" % ", ".join(fitted)
        return "fast", "short English text, no fitted head: the fast encoder is enough"

    def predict(self, state: Any = None, questions: Optional[Mapping] = None, *,
                model: Optional[str] = None, image=None, audio=None, video=None,
                lang: Optional[str] = None, **kw) -> Dict[str, Any]:
        if model is None:
            model, reason = self.choose(state, questions, image=image, audio=audio, video=video,
                                        lang=lang)
        else:
            reason = "explicit model=%s" % model
        if model == "vega":
            out = self.vega.predict(state, questions, lang=lang, image=image, audio=audio,
                                    video=video, **kw)
        elif model == "fast":
            if image is not None or audio is not None or video is not None:
                raise ValueError("the fast laya model takes text only; drop model= or use vega")
            out = self.fast.predict(state, questions, lang=lang, **kw)
        else:
            raise ValueError("model must be 'fast' or 'vega', got %r" % model)
        out["routing"] = {"model": model, "reason": reason}
        return out
