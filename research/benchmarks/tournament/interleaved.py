"""Row-interleaved online timing, separate from the frozen v1 measurements."""
from dataclasses import asdict, dataclass
import time
from types import MappingProxyType

import numpy as np

from laya.common import render_options
from laya.shortlist import embed_fn_from_agent
from research.benchmarks.tournament.compare_embeddings import (
    Predictor, STRATEGIES, infeasible_budget, memory_snapshot, predict_strategy, synchronize,
)
from research.benchmarks.tournament.diagnostics import fingerprint


DEDICATED = ("top1", "shortlist", "combined", "shortlist-wide")


@dataclass(frozen=True)
class Arm:
    id: str
    strategy: str
    k: int = 20
    embedder: str = None
    group_size: int = 16


class LatencyFailure(RuntimeError):
    """An incomplete report is evidence of failure, never a partial timing score."""

    def __init__(self, report):
        super().__init__(report["error"])
        self.report = report


class ColdQuery:
    """One uncached query per request; resident option vectors are read-only."""

    def __init__(self, encoder, options, query_prefix=""):
        self.encoder, self.options, self.query_prefix = encoder, options, query_prefix
        self.query_encodings = 0

    def __call__(self, texts):
        if not texts:
            raise ValueError("a shortlist embedding request must contain a query")
        # The first slot is a query even if its text is identical to a label.
        query = np.asarray(self.encoder([self.query_prefix + texts[0]]), dtype=np.float32)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("encoder did not return one query vector")
        self.query_encodings += 1
        return np.concatenate([query] + [self.options[text][None, :] for text in texts[1:]])


class CheckedDecision:
    """Validate each real round before a later round can hide its fallback."""

    def __init__(self, agent):
        self.agent, self.calls = agent, 0

    def predict(self, state, questions, **kwargs):
        choices = {qid: tuple(question["criteria"]) for qid, question in questions.items()}
        before = self._runtime(len(questions))
        self.calls += 1
        result = self.agent.predict(state, questions, **kwargs)
        if self._runtime(len(questions)) != before:
            raise ValueError("device, dtype or fallback changed within a decision round")
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ValueError("decision returned no answers mapping")
        for qid, candidates in choices.items():
            answer = result["answers"].get(qid)
            if (not isinstance(answer, dict) or answer.get("type") != "choice" or
                    answer.get("choice") not in candidates):
                raise ValueError("decision returned a missing or out-of-set choice for %r" % qid)
        return result

    def _runtime(self, rows):
        agent = self.agent
        return (str(agent.device), str(getattr(agent, "dtype", None)),
                getattr(agent, "amp_enabled", None), str(agent.dtype_for(rows)),
                getattr(agent, "cpu_fallback_count", 0), getattr(agent, "_amp_failures", 0))


def _decision_calls(arm, labels):
    if arm.strategy == "top1":
        return 0
    calls = 1
    if arm.strategy in ("combined", "tournament"):
        n = min(labels, arm.k) if arm.strategy == "combined" else labels
        while n > arm.group_size:
            n = -(-n // arm.group_size)
            calls += 1
    return calls


def _options(encoder, texts, prefix):
    unique = list(dict.fromkeys(texts))
    vectors = np.array(encoder([prefix + text for text in unique]), dtype=np.float32, copy=True)
    if vectors.ndim != 2 or len(vectors) != len(unique) or not np.isfinite(vectors).all():
        raise ValueError("invalid resident label embeddings")
    vectors.setflags(write=False)
    return MappingProxyType(dict(zip(unique, vectors)))


def _device(value):
    import torch
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def _placement(owner, requested):
    actual = _device(owner.device)
    if actual != requested:
        raise ValueError("requested device %s, actual device %s" % (requested, actual))
    tensors = list(owner.model.parameters()) + list(owner.model.buffers())
    devices = sorted({_device(tensor.device) for tensor in tensors})
    if not devices or devices != [requested]:
        raise ValueError("model tensors disagree with requested device: %r" % devices)
    return {"device": actual, "parameter_dtypes": sorted({str(p.dtype) for p in owner.model.parameters()}),
            "buffer_dtypes": sorted({str(p.dtype) for p in owner.model.buffers()})}


def _environment(agent, encoders, requested, arms, labels, requested_dtypes):
    import torch
    placement = _placement(agent, requested)
    rows = {1}
    for arm in arms:
        n = min(arm.k, labels) if arm.strategy == "combined" else labels
        if arm.strategy in ("combined", "tournament"):
            while n > arm.group_size:
                n = -(-n // arm.group_size)
                rows.add(n)
    placement.update(autocast_dtype=str(getattr(agent, "dtype", None)),
                     cpu_fallback_count=getattr(agent, "cpu_fallback_count", 0),
                     amp_failures=getattr(agent, "_amp_failures", 0),
                     effective_dtype_by_question_rows={str(n): str(agent.dtype_for(n)) for n in sorted(rows)})
    return {"requested_device": requested, "requested_dtypes": requested_dtypes,
            "laya": placement, "encoders": {key: _placement(enc, requested) for key, enc in encoders.items()},
            "torch_intra_op_threads": torch.get_num_threads(),
            "torch_inter_op_threads": torch.get_num_interop_threads()}


def _validate(cases, questions, arms, encoders, specs, warmup, passes):
    if not cases or not arms or isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValueError("cases/arms must be nonempty and passes must be positive")
    if set(questions) != {"q"} or questions["q"].get("type") != "choice":
        raise ValueError("latency-v2 requires one choice question named q")
    if len({arm.id for arm in arms}) != len(arms) or any(not arm.id for arm in arms):
        raise ValueError("arm ids must be nonempty and unique")
    n = len(questions["q"]["criteria"])
    for arm in arms:
        if arm.strategy not in STRATEGIES:
            raise ValueError("unknown strategy %r" % arm.strategy)
        if (isinstance(arm.k, bool) or not isinstance(arm.k, int) or arm.k < 1 or
                isinstance(arm.group_size, bool) or not isinstance(arm.group_size, int) or arm.group_size < 2):
            raise ValueError("invalid k or group size")
        if arm.strategy in DEDICATED:
            if arm.embedder not in encoders or arm.embedder not in specs:
                raise ValueError("missing encoder/spec for arm %s" % arm.id)
            encoder = encoders[arm.embedder]
            if hasattr(encoder, "spec"):
                actual = asdict(encoder.spec)
                declared = asdict(specs[arm.embedder])
                for key in ("query_prefix", "option_prefix"):
                    actual.pop(key)
                    declared.pop(key)
                if actual != declared:
                    raise ValueError("encoder weight/pooling contract disagrees with arm spec")
        elif arm.embedder is not None:
            raise ValueError("only dedicated arms take an embedder")
        if arm.strategy in DEDICATED + ("builtin",) and n <= (1 if arm.strategy == "top1" else arm.k):
            raise ValueError("retrieval timing requires k smaller than the label count")
    if len(warmup) != 3:
        raise ValueError("exactly three development warmups are required")
    for rows in (cases, warmup):
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate row ids")
        if len({fingerprint(row["state"]) for row in rows}) != len(rows):
            raise ValueError("latency rows must contain distinct requests")
        if any(row.get("state_sha256", fingerprint(row["state"])) != fingerprint(row["state"]) for row in rows):
            raise ValueError("state fingerprint mismatch")
    if ({row["id"] for row in cases} & {row["id"] for row in warmup} or
            {fingerprint(row["state"]) for row in cases} & {fingerprint(row["state"]) for row in warmup}):
        raise ValueError("development warmups must be disjoint from measured requests")


def run_interleaved(agent, cases, questions, arms, encoders, specs, warmup, requested_device="cpu",
                    passes=3, clock=time.perf_counter, progress=None, requested_dtypes=None):
    """Time an explicit list of Arm objects with independent cold-query adapters.

    ``encoders`` and ``specs`` map arm embedder IDs to TextEncoder-like objects and
    EmbeddingSpec objects. Different prefix IDs may share the same encoder weights.
    ``warmup`` contains exactly three distinct, disjoint development cases. Caller
    supplies checkpoint/dataset identities and persists returned or failed reports.
    The optional progress callback receives the cumulative report outside measured
    intervals, after each completed sample. Persist it synchronously; a callback
    retaining reports in memory must copy them because the report keeps growing.
    """
    _validate(cases, questions, arms, encoders, specs, warmup, passes)
    requested = _device(requested_device)
    used = {arm.embedder: encoders[arm.embedder] for arm in arms if arm.embedder is not None}
    requested_dtypes = requested_dtypes or {"laya": "checkpoint default", "dedicated": "torch.float32"}
    report = {"schema": "laya-dedicated-shortlist/latency-v2", "status": "running", "arms": {},
              "passes": passes, "rows_per_pass": len(cases), "independent_requests": len(cases),
              "case_ids": [row["id"] for row in cases], "warmup_case_ids": [row["id"] for row in warmup],
              "scope": "cold query encoding + ranking + all decision rounds; warm labels and models",
              "schedule": [], "samples": [], "label_setup": {}}
    try:
        before = _environment(agent, used, requested, arms, len(questions["q"]["criteria"]), requested_dtypes)
        report["environment_before"] = before
        texts = render_options({"t": "choice", "crit": questions["q"]["criteria"]})
        pools, adapters, runners, decisions = {}, {}, {}, {}
        builtin = embed_fn_from_agent(agent) if any(a.strategy == "builtin" for a in arms) else None
        for arm in arms:
            report["arms"][arm.id] = {"arm": asdict(arm), "status": "running",
                                       "passes_ms": [[] for _ in range(passes)]}
            # Check feasibility outside the timer, and retain the rejected arm.
            infeasible = next((value for row in cases + warmup
                               if (value := infeasible_budget(agent, row["state"], questions, arm.strategy))), None)
            if infeasible:
                report["arms"][arm.id].update(infeasible)
                continue
            decisions[arm.id] = CheckedDecision(agent)
            runners[arm.id] = Predictor(decisions[arm.id], arm.strategy, arm.group_size)
            if arm.strategy in DEDICATED or arm.strategy == "builtin":
                spec = specs[arm.embedder] if arm.strategy in DEDICATED else None
                encoder = used[arm.embedder] if spec else builtin
                key = (id(encoder), spec.option_prefix if spec else "")
                if key not in pools:
                    pools[key] = _options(encoder, texts, spec.option_prefix if spec else "")
                    report["label_setup"][arm.embedder or "builtin"] = {
                        "unique_texts": len(pools[key]), "encoder_calls": 1,
                        "option_prefix": spec.option_prefix if spec else ""}
                adapters[arm.id] = ColdQuery(encoder, pools[key], spec.query_prefix if spec else "")
                if spec:
                    report["arms"][arm.id]["embedding_spec"] = asdict(spec)

        def check_environment():
            current = _environment(agent, used, requested, arms, len(questions["q"]["criteria"]), requested_dtypes)
            if current != before:
                raise ValueError("device, dtype or thread placement changed during timing")
            return current

        for arm in arms:
            if arm.id not in runners:
                continue
            for row in warmup:
                check_environment()
                synchronize(agent.device)
                predict_strategy(runners[arm.id], row["state"], questions, adapters.get(arm.id), arm.k)
                synchronize(agent.device)
                check_environment()
        report["environment_after_setup"] = check_environment()
        for iteration in range(passes):
            for row_index, row in enumerate(cases):
                offset = (iteration + row_index) % len(arms)
                rotated = arms[offset:] + arms[:offset]
                report["schedule"].append({"pass": iteration, "id": row["id"], "arms": [a.id for a in rotated]})
                for arm in rotated:
                    if arm.id not in runners:
                        continue
                    check_environment()
                    adapter = adapters.get(arm.id)
                    count = adapter.query_encodings if adapter else 0
                    decision_count = decisions[arm.id].calls
                    synchronize(agent.device)
                    started = clock()
                    predict_strategy(runners[arm.id], row["state"], questions, adapter, arm.k)
                    synchronize(agent.device)
                    elapsed = (clock() - started) * 1000
                    check_environment()
                    encodings = adapter.query_encodings - count if adapter else 0
                    if encodings != (1 if adapter else 0):
                        raise ValueError("a timed request did not encode exactly one cold query")
                    decision_calls = decisions[arm.id].calls - decision_count
                    if decision_calls != _decision_calls(arm, len(questions["q"]["criteria"])):
                        raise ValueError("a timed request made an unexpected number of decision calls")
                    if not np.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("invalid timing sample")
                    sample = {"pass": iteration, "id": row["id"], "arm": arm.id,
                              "ms": elapsed, "query_encodings": encodings, "decision_calls": decision_calls}
                    report["samples"].append(sample)
                    report["arms"][arm.id]["passes_ms"][iteration].append(elapsed)
                    if progress is not None:
                        progress(report)
        for arm in arms:
            result = report["arms"][arm.id]
            if result["status"] == "infeasible":
                continue
            if any(len(values) != len(cases) for values in result["passes_ms"]):
                raise ValueError("missing samples for arm %s" % arm.id)
            result.update(status="complete", p50_ms=float(np.quantile(result["passes_ms"], .5)),
                          p95_ms=float(np.quantile(result["passes_ms"], .95)))
        report.update(status="complete", environment_after=check_environment(), memory=memory_snapshot(agent.device))
        return report
    except Exception as exc:
        report.update(status="failed", error="%s: %s" % (type(exc).__name__, exc))
        raise LatencyFailure(report) from exc
