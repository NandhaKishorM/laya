# Checkpoint integrity

Laya downloads model weights from the Hugging Face Hub at load time. By default it takes whatever
the repository's default revision points at, which is convenient and is what an offline cache
already holds. If you would rather pin a reviewed commit, or refuse to load a checkpoint whose
bytes have changed, both are available and both are opt-in.

Nothing here changes what Laya loads until you ask for it, so adding these options to an existing
deployment is safe. Both are library-level: digests reach the HTTP server through an environment
variable, and revision pinning through `LAYA_REVISION` — see [Pinning a revision in the server](#pinning-a-revision-in-the-server).

Related: [Docker](docker.md) for deployment variables, [`laya.load` and `Agent`](reference/agent.md),
and [`Router`](reference/router.md).

## Pin a revision

Pass `revision` to any loader. It accepts a commit SHA, a branch, or a tag, and is forwarded to the
Hub.

```python
import laya

agent = laya.load("convaiinnovations/laya", revision="<commit sha, branch, or tag>")
print(agent.revision)   # what the download resolved to
```

Prefer the reviewed SHAs that ship with Laya over a literal of your own: they are updated with the
checkpoints, so this form cannot go stale.

```python
from laya import PINNED_REVISIONS

agent = laya.load(
    "convaiinnovations/laya",
    revision=PINNED_REVISIONS["convaiinnovations/laya"],
)
```

`Router` takes the same `revision`, and `revisions` to pin each checkpoint separately. The keys of
`PINNED_REVISIONS` are the three *standalone* repositories, so pin a `Router` with
`standalone_repos=True`:

```python
router = laya.Router(standalone_repos=True, revisions={
    "english": PINNED_REVISIONS["convaiinnovations/laya"],
    "multilingual": PINNED_REVISIONS["convaiinnovations/laya-multilingual"],
    "typed-decisions": PINNED_REVISIONS["convaiinnovations/laya-typed-decisions"],
})
```

This matters because a default `Router` loads all three checkpoints from the one bundle repository
(`convaiinnovations/laya`, with `multilingual/` and `typed-decisions/` as subfolders), and a commit
SHA from `laya-multilingual` does not exist in the bundle repository. Without `standalone_repos`
the pin is not merely ignored — the load fails. If you would rather stay on the bundle repository,
pin it with one `revision=` for all three instead of per-model `revisions=`.

Pin all three even if you only serve two. A `Router` offers every checkpoint it knows about
regardless of what you preload, so an unpinned entry is one routing decision away from loading
unpinned.

### Why pinning is not the default

Pinning by default would break loading from an older cached snapshot, which matters for on-device
and air-gapped deployments: `HF_HUB_OFFLINE=1` with a cache that predates the pin would stop
working. Laya therefore keeps the Hub default unless you pass a revision, and makes the reviewed
SHAs available for when you want them.

## Verify artifact digests

A pinned revision says *which* commit to fetch. A digest says *which bytes* you expect. Every file
you list is hashed before any of them is parsed and before weights reach the runtime. The map is
`{path relative to the checkpoint: sha256 hex}` — generate it first, then pass it.

### Do you need both?

A pinned revision already fixes the content: the Hub is git, so the commit determines the tree, and
large files are addressed by their own SHA-256. If you pin and the download succeeds, you have the
bytes that commit names. So the digest is not there to repeat that check — it differs in **what it
trusts**.

A revision asks the Hub for a commit and believes the answer. A digest is a record *you* made and
*you* keep, compared on every load. That buys three things pinning does not:

- **Cover for the common case, which is unpinned.** Pinning is opt-in and off by default, so most
  deployments follow a moving branch. A digest is then the only thing that notices a change.
- **A check on your own disk.** After download, the checkpoint is ordinary files in a cache that
  anything on the box can edit. Nothing re-verifies them at load time — except a digest.
- **Independence from the source.** If a mirror, proxy or the Hub itself served different bytes,
  the digest is the only control not asking the thing under test to vouch for itself.

That independence is also why generating the map is a manual step: a fingerprint stops being an
independent record the moment the thing it checks produces it for you.

### Generate the map

Generate it from a checkpoint you have reviewed rather than copying digests from anywhere,
including this page. There is deliberately no command that produces this for you: a map computed
from the copy Laya just downloaded would hash those bytes and then verify them against themselves.
The check is worth something only because a person decided the bytes were the ones they wanted, so
generating the map is the step where that decision gets recorded. **A map belongs to exactly one checkpoint**: the bundle repository holds a
different `rl_agent_config.json` in its root (the english checkpoint) than in `multilingual/`, so a
map generated from one will fail against the other.

```python
import hashlib, json, os

CHECKPOINT = "/path/to/checkpoint"   # the directory a load actually reads
FILES = [
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "encoder/config.json",
    "model.safetensors",
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

digests = {rel: sha256(os.path.join(CHECKPOINT, rel)) for rel in FILES}
with open("digests.json", "w") as f:
    json.dump(digests, f, indent=2)
```

A torch `Agent` load parses five files, and those are four of them. (`ONNXAgent` reads a different
set and additionally accepts the keys `onnx` and `onnx_path` to digest the graph itself.) The fifth,
`tokenizer/tokenizer_config.json`, is deliberately left out: Laya may normalise it *after*
verification and write it back, in which case pinning it makes the **next** load fail. That rewrite
is conditional — it fires only when the file declares no `tokenizer_class`, or declares
`TokenizersBackend`, or carries `extra_special_tokens` as a list — so on some checkpoints it never
happens and pinning the file would appear to work. Leaving it out is the portable choice, and it
means one parsed file is unverified. See
[What this does and does not protect](#what-this-does-and-does-not-protect).

### Use the map

```python
import json

import laya

with open("digests.json") as f:
    agent = laya.load("convaiinnovations/laya", expected_sha256=json.load(f))
```

Keys are paths relative to the checkpoint directory. A mismatch raises `ValueError`, and a listed
file that is absent raises `FileNotFoundError`. Files you do not list are not checked at all, so
the map is also the definition of what you are protecting. This works on a local directory as well
as a Hub download.

### Without touching the code

`LAYA_SHA256_DIGESTS` holds the same map as JSON and applies whenever a loader is called without an
explicit `expected_sha256`:

```bash
export LAYA_SHA256_DIGESTS="$(cat digests.json)"
laya-serve
```

Nothing generates this for you: the value is your own map, from a checkpoint you reviewed. Under
Docker it has to be in the environment *before* compose starts, either exported as above or in the
`.env` file compose reads — the service passes `${LAYA_SHA256_DIGESTS:-}` through, so an unset
variable silently means no verification:

```bash
echo "LAYA_SHA256_DIGESTS=$(cat digests.json)" >> .env
docker compose -f compose.yaml -f compose.http.yaml up laya-serve
```

There is no `_FILE` variant of this variable: that indirection exists for secrets, and a digest map
is not one.

An unset or empty variable means no verification, so this is safe to leave out of environments that
do not need it. Malformed JSON raises rather than silently skipping the check.

**Name each checkpoint when a process loads more than one.** The variable takes two shapes, and the
value types say which:

```bash
# one set of files, checked on every checkpoint the process loads
LAYA_SHA256_DIGESTS='{"rl_agent_config.json": "<sha256>"}'

# a map per checkpoint, which is what a router serving several needs
LAYA_SHA256_DIGESTS='{"english": {"rl_agent_config.json": "<sha256>"},
                      "multilingual": {"rl_agent_config.json": "<sha256>"}}'
```

The flat form is `verify_digests`'s own reading and applies the same paths to everything, so on a
router it can only ever match one checkpoint and refuses the rest. The bundled repository ships a
separate `model.safetensors` and `rl_agent_config.json` per checkpoint, so name them:

```
flat map generated from the english checkpoint
  load english        ok
  load multilingual   ValueError: laya: SHA-256 mismatch for rl_agent_config.json
```

A checkpoint the nested map does not name is deliberately unpinned rather than an error, and a model
name the router does not know raises instead of leaving that checkpoint unverified. `en` resolves to
`english`, the same normalisation `Router(sha256_digests=...)` applies.

The two routes differ on that last point, which is easy to trip over if you use both. A checkpoint
the *environment's* nested map omits is pinned to an empty map, so a flat map cannot leak into it. A
checkpoint omitted from `Router(sha256_digests=...)` in code has no entry at all, so it still falls
back to whatever the environment says. Name every checkpoint you mean to pin in whichever one you
use.

An unset or empty variable means no verification, so this is safe to leave out of environments that
do not need it. Malformed JSON raises rather than silently skipping the check, and mixing the two
shapes in one object is refused by name.

### What a mismatch looks like in the server

How it surfaces depends on preloading. Bare `laya-serve` preloads by default (`LAYA_PRELOAD=1`), so a
mismatch fails at **startup** — loud and deterministic. The containers in this repository set
`LAYA_PRELOAD=0` (`compose.http.yaml`, and [Docker](docker.md) documents the override), so there the
first load happens on a request and nothing is verified until then. A mismatch is then a **422** on
whichever ticket routes to that checkpoint: `laya/serve.py` maps the `ValueError` to
`HTTPException(422)` and returns the digest text to the caller. A *listed but absent* file raises
`FileNotFoundError` instead, which falls through to a generic **500 "inference failed"** with the
reason only in the container log.

Plan for the 422. It sorts as a client error in logs, dashboards and alert rules, so the default
place an operator looks for a broken deployment is the one place this will not appear.

### Pinning a revision in the server

`LAYA_REVISION` holds a commit, branch or tag applied to every checkpoint download, or the word
`reviewed`, which looks each repository up in `PINNED_REVISIONS` and uses its own SHA:

```bash
LAYA_REVISION=reviewed laya-serve
```

`reviewed` for a repository the table has no entry for **raises** rather than loading it unpinned —
a pin that quietly resolves to nothing is the failure this control exists to prevent. An explicit
`revision=` argument still wins over the variable, and unset or blank means "not asked for", so an
`HF_HUB_OFFLINE=1` cache keeps loading exactly as before.

In code, `Router` takes both per model:

```python
router = laya.Router(
    revisions={"english": PINNED_REVISIONS["convaiinnovations/laya"]},
    sha256_digests={"english": {"rl_agent_config.json": "<sha256>"}},
)
```

Digests are always per model — there is no router-wide equivalent of `revision`, because a commit SHA
can be shared across checkpoints and a digest cannot. See [Docker](docker.md) for the deployment
variables and [Router](reference/router.md) for the full constructor.

## When a checkpoint is updated

The two controls behave differently, and only one of them needs anything from you.

**A pinned revision keeps you where you are.** A new checkpoint does not reach a pinned
deployment until you change the pin, which is the point of pinning. `PINNED_REVISIONS` moves with
the library, so taking a newer reviewed commit means upgrading Laya, not editing a SHA.

**Digests stop the load, on purpose.** Your map was generated from bytes you reviewed. Different
bytes raise `ValueError` before anything is parsed:

```
ValueError: laya: SHA-256 mismatch for rl_agent_config.json: expected ae287b56…, got 25061739…
```

That is the feature working, not a bug to route around. The order matters:

1. Find out why the bytes changed — an intended release, or something you did not expect.
2. Review the new checkpoint.
3. Regenerate the map from the reviewed copy.
4. Deploy the new map.

**Do not skip to step 3.** Re-running the generator against whatever just arrived makes the check
pass and verifies nothing — it records the new bytes as trusted because they are present, which is
precisely the state the digest existed to detect.

Two details. A new map delivered through `LAYA_SHA256_DIGESTS` needs the process restarted, because
a running server keeps the environment it started with. And this sequence only applies to a
deployment that is *not* revision-pinned: with both controls on, the new bytes never arrive until
you move the pin.

## Confirm what actually loaded

Every agent records the commit it came from, `None` for a local directory:

```python
agent.revision                 # Agent and ONNXAgent
router.loaded_revisions        # {"english": "55cf4c4e…", …} for each resident agent
```

`agent.revision` reports the snapshot the download resolved to, falling back to whatever you passed,
so pinning by branch or tag echoes that name rather than a SHA — pin by SHA if you want this field
to be one. Loading from a local directory reports `None`, and `revision` is ignored there, because
there is no Hub snapshot to resolve.

The server reports the same thing, which is the quickest way to confirm a deployment is running
the checkpoint you think it is:

```bash
curl -s localhost:8000/health
# {"status":"ok","loaded":["english"],"revisions":{"english":"55cf4c4e…"},"device":"auto"}
```

## laya-ts

The TypeScript package mirrors the pinning and digest parts — `revision`, `expectedSha256`, and
reading the revision back. It has no `LAYA_SHA256_DIGESTS` equivalent and no server, so the two
sections above do not apply to it:

```ts
import { loadNodeBundle, PINNED_REVISIONS } from "laya-ts";

const bundle = await loadNodeBundle("convaiinnovations/laya", {
  revision: PINNED_REVISIONS["convaiinnovations/laya"],
  expectedSha256: { "rl_agent_config.json": "<sha256 of that file>" },
});
```

An explicit revision joins the on-disk cache path under `~/.cache/laya-ts/`, so differently-pinned
artifacts never collide. In the browser the revision travels in the request URL instead, which
keys `CacheStorage` the same way. `createNodeProvider` accepts `expectedSha256` for the ONNX graphs
it loads.

## What this does and does not protect

It detects a checkpoint whose contents changed from what you reviewed — an upstream repository
edit, a compromised mirror, a corrupted download, or a modified local copy.

It does not make an unreviewed checkpoint safe. A digest only says the bytes match what you
recorded; deciding that those bytes are trustworthy is still yours.

Three limits worth knowing before you rely on it:

- **Only listed files are checked.** There is no "verify everything" mode and no way to reject a
  file you did not list, so an artifact missing from your map is loaded unverified. The map is the
  boundary of the guarantee.
- **One parsed file is therefore outside it.** `tokenizer/tokenizer_config.json` is parsed, but Laya
  may normalise it and write it back immediately after the digest check, so pinning it can succeed
  on the first load and fail on the next. The recommended map leaves it out for that reason, which
  means its bytes are not verified. The rewrite is conditional on what the file declares, so
  whether it happens depends on the checkpoint.
- **Verification happens at load time only.** Nothing re-checks a file afterwards, whether it is
  replaced by an attacker or by the process itself.
