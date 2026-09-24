# Production serving

`laya-serve` is a Jev-compatible HTTP server around `laya.Router`. It is built to
run behind a load balancer as a long-lived process, not just as a demo: requests
are gathered into small batches so one forward pass answers many callers, the
queue is bounded so load sheds instead of growing, `/ready` and `/metrics` speak
the language of orchestrators and Prometheus, and SIGTERM drains in-flight work
instead of dropping it.

## Quickstart

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve
```

The server binds `0.0.0.0:8000` and exposes:

| method | path | purpose |
|---|---|---|
| `POST` | `/v1/systemone` | the Jev-compatible decision endpoint |
| `GET` | `/health` | liveness: the process is up and the app is built |
| `GET` | `/ready` | readiness: checkpoints resident and the batcher accepting work; `503` while starting or draining |
| `POST` | `/drain` | start draining before SIGTERM: stop intake and fail readiness so an orchestrator stops routing; returns once set |
| `GET` | `/metrics` | Prometheus text; `404` when `LAYA_METRICS=0`, bearer-gated when `LAYA_API_KEY` is set |

With the default `LAYA_PRELOAD=1` the server binds and answers `/health`
immediately, then loads checkpoints in the background; `/ready` returns `503`
until they are resident. Point liveness probes at `/health` and readiness probes
(and `docker compose up --wait`) at `/ready`, and give the readiness probe a
generous `failureThreshold` (or use a `startupProbe`), because a cold load of all
three checkpoints can take minutes. If a preload fails, `/health` starts
returning `503` as well so the orchestrator restarts the pod instead of leaving a
process that never serves.

## Environment

| var | default | meaning |
|---|---|---|
| `LAYA_BATCH_WINDOW_MS` | `10` | how long a dispatch waits to collect other requests; `0` dispatches immediately |
| `LAYA_BATCH_MAX` | `16` | hard cap on requests per forward pass |
| `LAYA_QUEUE_MAX` | `64` | bounded queue; overflow is shed with `503` |
| `LAYA_REQUEST_TIMEOUT_S` | `120` | budget covering queue wait plus inference |
| `LAYA_SHUTDOWN_GRACE_S` | `30` | drain budget after SIGTERM before in-flight work is cancelled |
| `LAYA_INFER_WORKERS` | `1` | dispatch threads; see [Sizing](#sizing) |
| `LAYA_WARMUP` | `1` | run one warmup pass per resident checkpoint after preload, to pay first-call JIT cost at startup |
| `LAYA_CLIENT_MAX_INFLIGHT` | `0` | per-client in-flight cap; over-limit callers get `429` + `Retry-After`; `0` disables |
| `LAYA_BREAKER_THRESHOLD` | `0` | consecutive inference failures before a client's breaker opens; `0` disables |
| `LAYA_BREAKER_COOLDOWN_S` | `30` | how long a breaker stays open before half-open probes resume |
| `LAYA_BREAKER_PROBE_SUCCESSES` | `2` | successful half-open probes in a row before a breaker closes |
| `LAYA_PROXY_HEADERS` | `0` | trust `X-Forwarded-For`/`-Proto` from `LAYA_FORWARDED_ALLOW_IPS`; enable **only** behind a proxy you control |
| `LAYA_FORWARDED_ALLOW_IPS` | `127.0.0.1` | IPs/networks whose forwarded headers are trusted (`*` to trust all) |
| `LAYA_STATE_URL` | unset | shared admission/breaker backend, e.g. `redis://host:6379/0`; unset keeps state in-process |
| `LAYA_SHARED_BREAKER` | `1` | with a shared store, share the per-client breaker too; `0` keeps it per-replica |
| `LAYA_BREAKER_ENTRY_TTL_S` | auto | breaker entry TTL: `auto` = `max(cooldown, 60)`, a positive number, or `0` to disable |
| `LAYA_STATE_FAILURE_POLICY` | `local` | on a shared-store error: `local` degrades to per-replica enforcement, `open` stops enforcing until the store recovers |
| `LAYA_METRICS` | `1` | set `0` to disable the registry and `/metrics` |
| `LAYA_LOG_FORMAT` | `text` | set `json` for one structured JSON object per line |

`LAYA_HOST`, `LAYA_PORT`, `LAYA_DEVICE`, `LAYA_PRELOAD`, `LAYA_MODELS`,
`LAYA_THREADS`, `LAYA_AUTO_TASK`, `LAYA_API_KEY` and `LAYA_LOG_LEVEL` are
unchanged. Invalid values fail fast at startup with a clear message rather than
being silently defaulted.

Warmup runs through `Router.predict`, so Router-level `on_predict_*` hooks see the
warmup request too; set `LAYA_WARMUP=0` if your hooks must not. The circuit
breaker is global, not per-checkpoint: repeated failures on any request open it
for every caller, which is the point when the engine itself is unhealthy.

## Batching

When several requests are in flight, the server routes each one, groups the
routed requests by checkpoint and question schema, and calls
`Router.predict_batch` once per group. A caller never sees the grouping: each
request still gets its own routed result, and a client's `model`/`task`/`lang`
override is preserved. Router-level `on_predict_start`/`on_predict_end` hooks run
once per regrouped batch with `ctx.states` the batch and `ctx.results` the batch
results, so audit and metrics hooks installed on the `Router` see batched traffic
exactly as they see a single `predict`.

The batching path is also what the `Router.predict_batch` API exposes outside
the server, so a batch script gets the same hook contract.

Tune the window for your latency budget: `10` ms is invisible at human
interaction times and collects a burst from one client retrying, while `0`
removes even that wait for a strict p99. Raise `LAYA_BATCH_MAX` only as far as
your accelerator's memory allows; the server caps the batch, it does not grow it.

## Load shedding, timeouts and fallback

- **Backpressure.** The queue holds at most `LAYA_QUEUE_MAX` waiting requests.
  Further requests are rejected immediately with `503` and a `Retry-After`
  header rather than accepted and left to time out. This keeps latency bounded
  under overload.
- **Per-client admission.** With `LAYA_CLIENT_MAX_INFLIGHT` set, each caller
  (identified by the `x-laya-client` header, else the bearer credential, else the
  client IP) may hold at most that many requests in flight; beyond it they get
  `429` + `Retry-After` while other callers keep being served. Use it when
  several clients share one server so one burst cannot starve the rest. This is a
  fairness control, not a security boundary: an unauthenticated caller can rotate
  `x-laya-client`. Behind a proxy, set `LAYA_PROXY_HEADERS=1` (and scope
  `LAYA_FORWARDED_ALLOW_IPS`) or every caller keys on the proxy's IP; leave it off
  when the port is reachable directly, because a spoofed `X-Forwarded-For` then
  becomes a bypass.
- **Circuit breaker.** With `LAYA_BREAKER_THRESHOLD` set, that many consecutive
  inference failures open **that client's** breaker: their requests fail fast with
  `503` + `Retry-After` without touching the model until `LAYA_BREAKER_COOLDOWN_S`
  elapses. Then half-open admits exactly one probe at a time, and
  `LAYA_BREAKER_PROBE_SUCCESSES` successes in a row close the breaker, so a service
  that is still warming up cannot flap it. Keying per client is deliberate: a global
  breaker lets one client's poison payload fast-fail every other client, so this one
  does not. A genuinely broken engine still opens every client's breaker as each
  accumulates failures.
- **Timeout.** A request that waits longer than `LAYA_REQUEST_TIMEOUT_S` in the
  queue or in inference returns `504`. Some forward passes cannot be cancelled
  once started, so a late result is dropped; the queue slot is freed.
- **Per-request fallback.** If a batched forward pass raises, the server retries
  each request in that batch individually, so one bad input does not fail its
  innocent batch-mates. Only the request that actually failed returns an error.
- **Drain.** `POST /drain` starts draining: readiness turns `503` so the load
  balancer stops routing, new requests get `503`, and in-flight work still
  finishes. Wire it to a Kubernetes `preStop` hook so the pod is out of the
  rotation before SIGTERM arrives:

  ```yaml
  lifecycle:
    preStop:
      exec:
        command: ["/bin/sh", "-c", "curl -fsS -X POST http://localhost:8000/drain || true; sleep 5"]
  ```

  On SIGTERM the server stops accepting new work, finishes the in-flight batch
  within `LAYA_SHUTDOWN_GRACE_S`, and fails whatever is still queued with `503`. `LAYA_SHUTDOWN_GRACE_S` is also handed to uvicorn as its
  graceful-shutdown timeout, so it is the real shutdown budget. Set the
  orchestrator's termination grace period comfortably above it; work still in
  flight at the hard deadline is cut off by uvicorn (the client sees a `500` or
  a reset), which is why the grace should exceed your p99 batch latency.

### Sharing state across replicas

By default both the per-client in-flight cap and the circuit breaker live in the
process, so `N` replicas enforce `N x` the cap and open their breakers independently.
Set `LAYA_STATE_URL` (for example `redis://laya-redis:6379/0`, from
`pip install "laya[serve-redis]"`) to put both in Redis:

- admission becomes one cluster-wide budget: `INCR`/`DECR` with a safety expiry, so
  a crashed replica cannot leak slots;
- the breaker spans replicas (default), or stays per-replica with
  `LAYA_SHARED_BREAKER=0` while admission is still shared;
- breaker entries expire after `LAYA_BREAKER_ENTRY_TTL_S` (`auto` = `max(cooldown,
  60)`, or `0` to disable; Redis keeps a 24h safety expiry regardless);
- `laya_circuits_open`, `laya_circuits_tracked` and `laya_inflight_total` are
  **cluster-wide**: a counter for in-flight plus expiry-sorted sets for the circuits,
  so a scrape is O(log n) rather than a keyspace `SCAN`;
- on a store error the `LAYA_STATE_FAILURE_POLICY` decides what happens. `local`
  (default) **degrades to in-process enforcement**: the replica keeps capping its own
  callers and opening its own breaker, so overload protection does not vanish, only
  the cross-replica coordination does. `open` stops enforcing entirely until the
  store recovers. Either way the error is counted in
  `laya_shared_state_errors_total{op}` and `laya_state_degraded` is set to `1` until
  a call succeeds again.

`laya_clients_active` (distinct clients) remains a local view, since counting distinct
clients cluster-wide would need a `SCAN`; `laya_inflight_total` is the cluster-wide
number that matters for saturation.

## Metrics

With `prometheus-client` installed (`pip install "laya[serve]"`), `/metrics`
renders a per-app registry:

| metric | type | labels |
|---|---|---|
| `laya_requests_total` | counter | `model`, `status` (`ok`, `error`, `timeout`, `rejected`, `rate_limited`) |
| `laya_rejections_total` | counter | `reason` (`queue_full`, `draining`, `client_limit`, `circuit_open`) |
| `laya_request_latency_seconds` | histogram | end-to-end latency |
| `laya_queue_wait_seconds` | histogram | arrival to dispatch |
| `laya_queue_oldest_seconds` | gauge | age of the oldest request still waiting (scrape-time) |
| `laya_inference_seconds` | histogram | one batch forward pass |
| `laya_batch_size` | histogram | requests per forward pass |
| `laya_batches_total` | counter | `result` (`ok`, `fallback`, `error`) |
| `laya_queue_depth` | gauge | requests waiting |
| `laya_active_requests` | gauge | requests currently dispatching |
| `laya_circuits_open` | gauge | clients whose breaker is open (cluster-wide on a shared store) |
| `laya_circuits_tracked` | gauge | clients with tracked breaker state (cluster-wide on a shared store) |
| `laya_inflight_total` | gauge | in-flight requests (cluster-wide on a shared store) |
| `laya_clients_active` | gauge | clients with in-flight requests (local view) |
| `laya_shared_state_errors_total` | counter | `op`; shared-state backend errors |
| `laya_state_degraded` | gauge | `1` when the last shared-state call fell back |
| `laya_state_backend` | gauge | `mode` (`local` or `redis`), value `1` |
| `laya_models_loaded` | gauge | resident checkpoints |

Alert on `laya_rejections_total` by reason (queue full vs draining vs client limit
vs breaker) and on `laya_queue_oldest_seconds`, which shows a request aging out
even when every completed request still looks fast.

A scrape example:

```yaml
scrape_configs:
  - job_name: laya
    metrics_path: /metrics
    static_configs:
      - targets: ["laya:8000"]
```

When `LAYA_API_KEY` is set, `/metrics` requires the same
`Authorization: Bearer <key>` header as `/v1/systemone`, so a scrape config needs
`authorization: { credentials: <key> }`.

## Structured logs

`LAYA_LOG_FORMAT=json` emits one JSON object per line with `ts`, `level`,
`logger`, `message`, and, for HTTP requests, `request_id`, `path`, `status` and
`total_ms`. The `x-request-id` response header echoes the caller's `x-request-id`
when supplied and is generated otherwise, so a request can be traced from the
client through the server logs. Inference failures are logged with a traceback
server-side; the client only ever sees a generic `500` detail.

## Sizing

`LAYA_INFER_WORKERS` defaults to `1` on purpose. Batching is the concurrency: one
worker saturates a GPU because it feeds it a full batch, and a second worker only
contends for the same device and splits batches. Raise it only when you have
several independent accelerators, or when inference is CPU-bound and you have
spare cores. Keep `LAYA_THREADS` at or below the physical core count.

A reasonable starting point for a single GPU is the defaults. Watch
`laya_batch_size` (raise `LAYA_BATCH_WINDOW_MS` if it stays at 1 under load),
`laya_queue_depth` (raise `LAYA_QUEUE_MAX` or add replicas if it sits at the
cap), and `laya_batches_total{result="fallback"}` (a non-zero rate means inputs
that fail a shared pass).

## Benchmark

`scripts/bench_serve.py` measures throughput and latency at concurrency 1, 8 and
32, with batching on and off, and prints the batch-size distribution. Run it
against a server you are already running:

```bash
python scripts/bench_serve.py --url http://127.0.0.1:8000 --state "hello"
```
