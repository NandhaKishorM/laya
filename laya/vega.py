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
   abstains at the configured risk, when the answer is ``unbound``, or when the conformal set
   at ``alpha`` (default 0.10) holds more than one option; ``auto`` only when all three agree.

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
    """
    out = {}
    for qid, q in questions.items():
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
        s = str(gold).strip().lower()
        return "true" if s in ("true", "yes", "1", "t", "y") else "false"
    if t == "score":
        if isinstance(gold, str) and not gold.isdigit():
            crit = list(q["criteria"])
            return str(crit.index(gold))
        return str(int(gold))
    return str(gold)


def _fold(answer: Mapping, q: Mapping) -> Dict[str, Any]:
    """One vegaml answer in laya's shape, with vega's extra fields kept.

    The engine keys a boolean's probabilities ``"False"``/``"True"`` and a head keys them
    ``"false"``/``"true"``; laya spells them lower case, so the keys are folded here once and
    scoring code downstream never sees two spellings. Not folding this is how a benchmark read
    every yes/no task at 0.000 for a day.
    """
    t = q["type"]
    probs = {str(k): float(v) for k, v in (answer.get("probs") or {}).items()}
    if t == "noul":
        low = {k.lower(): v for k, v in probs.items()}
        probs = {"false": low.get("false", 0.0), "true": low.get("true", 0.0)}
    pset = answer.get("prediction_set")
    if pset is not None:
        pset = [str(p).lower() if t == "noul" else str(p) for p in pset]
    out: Dict[str, Any] = {
        "type": t,
        "probabilities": {k: round(v, 6) for k, v in probs.items()},
        "answer_confidence": (round(float(answer["confidence"]), 6)
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
        out["score"] = int(level) if level is not None else (int(best) if best else None)
        if answer.get("expected_score") is not None:
            out["expected_score"] = round(float(answer["expected_score"]), 4)
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
                 hooks=None, on_predict_start=None, on_predict_end=None, **ignored):
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
        from .hooks import normalise_hooks

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
        self.fit_report: Dict[str, Any] = {}
        self.hooks = normalise_hooks(hooks, on_predict_start, on_predict_end)
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
                on_predict_end=None, **ignored) -> Dict[str, Any]:
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
        """
        if questions is None:
            raise ValueError("predict needs `questions`")
        given = sum(x is not None for x in (state, image, audio, video))
        if given != 1:
            raise ValueError("predict takes exactly one of state=, image=, audio=, video=; got %d"
                             % given)
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
            if image is not None:
                raw = self.vega.decide_image(image, vq, mode=mode, alpha=self.alpha, risk=self.risk)
                modality = "image"
            elif audio is not None:
                raw = self.vega.decide_audio(audio, vq, asr=asr, language=language,
                                             sampling_rate=sampling_rate, mode=mode,
                                             alpha=self.alpha, views=views)
                modality = "audio"
            else:
                raw = self.vega.decide(state, vq, mode=mode, alpha=self.alpha, views=views)
                modality = "text"
            if self.calibration is not None:
                raw = self.calibration.apply(_pad_for_calibration(raw, self.calibration))
        answers = {}
        for qid, q in questions.items():
            a = _fold(raw["answers"][qid], q)
            a.update(self._route(raw["answers"][qid], a, min_confidence))
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

    # ------------------------------------------------------------------ the route
    def _route(self, raw_answer: Mapping, folded: Mapping, min_confidence: Optional[float]) -> Dict:
        """``route`` and why, decided without a user-set number. See the module docstring."""
        if "route" in raw_answer and raw_answer["route"] in ("auto", "review"):
            route, why = raw_answer["route"], "band: " + str(raw_answer.get("route_reason", ""))
        else:
            reasons = []
            if folded.get("abstain"):
                reasons.append("the engine abstains at risk %s" % self.risk)
            if folded.get("unbound"):
                reasons.append("unbound: far from every well, likely off-distribution")
            pset = folded.get("prediction_set")
            if pset is not None and len(pset) > 1:
                reasons.append("conformal set at alpha %s holds %d options" % (self.alpha, len(pset)))
            if pset is None and folded.get("abstain") is None:
                # a head-served answer with no band fitted: nothing here can vouch for it
                reasons.append("head answer with no fitted band; fit() or calibrate() first")
            route = "review" if reasons else "auto"
            why = "; ".join(reasons) if reasons else (
                "conformal singleton, not abstaining, in distribution")
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
                                          else "conformal set + abstain + unbound")}
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
    def _examples(self, rows: Sequence[Mapping], questions: Optional[Mapping]) -> Tuple[list, Dict]:
        """laya training rows as vegaml fit examples: ``(state, {qid: label}, {qid: question})``.

        Rows are ``{"state", "questions"?, "gold"|"expected"}``: the shape ``items_from_rows``
        reads, so a corpus prepared for laya's fine-tune fits vega's heads unchanged. A row
        without its own ``questions`` uses ``questions``.
        """
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
            labels = {qid: _gold_label(qs[qid], g) for qid, g in gold.items() if qid in qs}
            if labels:
                examples.append((r["state"], labels, vq))
        return examples, sheet

    def fit(self, rows: Sequence[Mapping], questions: Optional[Mapping] = None,
            folds: Optional[int] = None, calibrate: bool = True,
            target_precision: Optional[float] = None) -> Dict[str, Any]:
        """Heads on your labels, then the calibration and the band, in one call.

        What laya's fine-tune does with gradient steps, this does with a small head per
        question over the frozen features: seconds on a GPU for a few hundred rows, and the
        reader is never touched. A head is kept only if it beats chance under cross-validation
        and SERVED only if it also beats the engine on the same rows, so ``report[qid]
        ["used_by_auto"]`` is the fact to read, not ``fitted``. With ``calibrate`` (the
        default) the same rows then refit a temperature and a conformal threshold per
        question and place the band, which is where the automatic ``route`` comes from.
        Nothing is scored on a fitted row by this method; held-out evaluation is yours.
        """
        examples, sheet = self._examples(rows, questions)
        if not examples:
            raise ValueError("no trainable rows")
        with self._lock:
            report = self.vega.fit(examples, sheet, folds=folds or self.folds)
        self.fit_report = {"heads": report, "n_rows": len(examples),
                           "serving": sorted(q for q, r in report.items() if r.get("used_by_auto")),
                           "kept": sorted(q for q, r in report.items() if r.get("fitted"))}
        if calibrate:
            self.fit_report["calibration"] = self.calibrate(
                rows, questions, target_precision=target_precision, _examples=(examples, sheet))
        return self.fit_report

    def calibrate(self, rows: Sequence[Mapping], questions: Optional[Mapping] = None,
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
                "band": None if band is None else
                {q: round(float(v["threshold"]), 4) for q, v in band.questions.items()}}

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
        return {"path": path, "heads": sorted(meta["heads"]),
                "calibrated": self.calibration is not None}

    def load_calibration(self, path: str) -> Dict[str, Any]:
        import numpy as np
        from vegaml.calibrate import Calibration
        from vegaml.ttt import TTTHead

        z = np.load(os.path.join(path, "heads.npz"), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        loaded = []
        for qid, m in meta["heads"].items():
            head = TTTHead(m["labels"], k=m["k"], l2=m["l2"])
            for name in HEAD_ARRAYS:
                setattr(head, name, z["%s::%s" % (qid, name)])
            head.ready = True
            head.engine_accuracy = m.get("engine_accuracy")
            self.vega.heads[qid] = (head, m.get("cv"))
            loaded.append(qid)
        cal_path = os.path.join(path, "calibration.json")
        if os.path.exists(cal_path):
            self.calibration = Calibration.load(cal_path)
        return {"heads": loaded, "calibrated": self.calibration is not None}

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
