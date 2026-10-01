"""Every autocast-dtype override the runtime reads has to be reachable from the shipped Compose
deployment.

`laya/agent.py` picks the dtype a forward runs in from the environment, and the deployment that
selects the device drops those names:

    $ LAYA_CUDA_AMP=fp16 docker compose -f compose.yaml -f compose.cuda.yaml config | grep LAYA_
          LAYA_DEVICE: cuda
          LAYA_DEVICE: cuda

The operator exported the variable, Compose saw it, and neither service's environment carries it,
so the image serves the checkpoint's own `amp_dtype` no matter what was set. That is not a
rounding difference: README's threshold section measures, on the `parity_fast.py` set (60 states,
288 questions per checkpoint, RTX 2000 Ada), bf16 flipping 3 of 864 argmaxes against the fp32
forward where fp16 flips none, at the same latency. The dtype is a policy dial, and the published
deployment could not turn it.

Why the names are derived rather than transcribed: the same three-way drift (package, markdown,
Compose) is what hid them, and a literal list in this file goes stale the moment the core reads
another one. So the set is read out of `laya/agent.py` -- the names it consumes *near a dtype
decision* -- and the rest of the file checks the deployment and the page against it, in both
directions. `LAYA_MPS_AMP_MIN_ROWS` is read by the same module and is deliberately not required
here, because no image in this repo can select MPS; check 5 keeps that reason falsifiable instead
of freezing it in a comment.

Weight-free, torch-free, no third-party YAML: the same textual style as the Compose section of
`tests/test_packaging.py`. The one docker-dependent check degrades to a note when docker is absent
and never to a pass.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List

PASS: List[str] = []
FAIL: List[str] = []
NOTES: List[str] = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE_FILES = ("compose.yaml", "compose.cuda.yaml", "compose.spark.yaml", "compose.http.yaml",
                 "compose.modelscope.yaml")
NAME = re.compile(r"\bLAYA_[A-Z0-9_]+\b")
# The dtype decision itself: a torch dtype object or the checkpoint's `amp_dtype` field. Prose
# that merely names a precision ("MPS fp16 autocast") is not a decision site.
DTYPE_SITE = re.compile(r"torch\.(?:float16|bfloat16|float32)\b|\bamp_dtype\b")
ENVIRON_GET = re.compile(r'os\.environ\.get\("(LAYA_[A-Z0-9_]+)"')


def check(what: str, got: object, want: object) -> None:
    if got == want:
        PASS.append(what)
    else:
        FAIL.append("%s: got %r, want %r" % (what, got, want))


def check_true(what: str, cond: bool, detail: object = "") -> None:
    if cond:
        PASS.append(what)
    else:
        FAIL.append("%s: %s" % (what, detail))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def dtype_names() -> List[str]:
    """Names `laya/agent.py` reads from the environment within a few lines of a dtype decision."""
    lines = read(os.path.join("laya", "agent.py")).splitlines()
    out = []
    for i, line in enumerate(lines):
        match = ENVIRON_GET.search(line)
        if not match:
            continue
        window = "\n".join(lines[max(0, i - 4): i + 5])
        if DTYPE_SITE.search(window):
            out.append(match.group(1))
    return sorted(set(out))


# The three idioms that make an empty environment value mean "not asked for". A Compose
# passthrough with an empty default -- `"${NAME:-}"` -- hands the container `""`, not an absent
# variable, so a reader that supplies a real default would take a decision away from an operator
# who set nothing. These are the only shapes that keep the empty string meaning "as before", and
# they are all in `laya/`: `os.environ.get("NAME", "")`, a falsy test on the same line, and the
# `raw = os.environ.get("NAME")` / `if not raw: return DEFAULT` resolver in the server module.
EMPTY_GET = 'os.environ.get("{name}", "")'
EMPTY_OR = 'os.environ.get("{name}") or '
# A resolver that returns its own default from the empty string: the read, then a falsy test on
# the next line, then the value it hands back. Only the first two lines are looked at, so a
# reader that ignores an empty value and keeps going is still reported.
EMPTY_RESOLVER = (r'os\.environ\.get\("{name}"(,\s*""\s*)?\)\s*\n'
                  r'\s*if (not \w+|\w+ is None or not \w+)\b')


def empty_unsafe_reads(name: str) -> List[str]:
    """Every read of ``name`` in the package an empty value would not survive."""
    out: List[str] = []
    for _dirpath, _dirs, _files in os.walk(os.path.join(ROOT, "laya")):
        for _filename in _files:
            if not _filename.endswith(".py"):
                continue
            _rel = os.path.relpath(os.path.join(_dirpath, _filename), ROOT)
            _text = read(_rel)
            for _match in ENVIRON_GET.finditer(_text):
                if _match.group(1) != name:
                    continue
                _line_start = _text.rfind("\n", 0, _match.start()) + 1
                _site = _text[_line_start:_match.end() + 60]
                _first = _site.split("\n")[0]
                if (EMPTY_GET.format(name=name) in _first
                        or EMPTY_OR.format(name=name) in _first):
                    continue
                if re.search(EMPTY_RESOLVER.format(name=name), _site):
                    continue
                out.append("%s:%d: %s" % (_rel, _text[:_match.start()].count("\n") + 1,
                                          _first.strip()))
    return out


def env_blocks(text: str) -> Dict[str, Dict[str, str]]:
    """service name -> its `environment` mapping, as (key, raw value) pairs.

    Enough of YAML to read a flat `KEY: "value"` block; these files use no anchors, no
    flow mappings and one service per top-level key, which is what `test_packaging.py` already
    relies on for the same reason (no third-party parser in this suite).
    """
    services: Dict[str, Dict[str, str]] = {}
    service = None
    in_env = False
    in_services = False
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if indent == 0:
            # `services:` opens the block; any other top-level key (`volumes:`, `networks:`)
            # closes it, so a volume name is never mistaken for a service.
            in_services = stripped == "services:"
            service = None
            in_env = False
            continue
        if not in_services:
            continue
        if indent == 2 and stripped.endswith(":"):
            service = stripped[:-1].strip()
            services[service] = {}
            in_env = False
            continue
        if service is None:
            continue
        if indent == 4 and stripped == "environment:":
            in_env = True
            continue
        if indent == 4:
            in_env = False
            continue
        if in_env and indent == 6 and ":" in stripped and not stripped.startswith("- "):
            key, _, value = stripped.partition(":")
            services[service][key.strip()] = value.strip()
    return services


names = dtype_names()

# 1. the derived set is what this PR is about, so it must not be silently empty or renamed away.
check_true("core/derives dtype names from laya/agent.py", len(names) >= 2, names)
check("core/derives the CUDA and CPU dtype overrides", names, ["LAYA_CPU_AMP", "LAYA_CUDA_AMP"])
for name in names:
    check_true("core/%s is read with os.environ.get" % name,
               'os.environ.get("%s"' % name in read(os.path.join("laya", "agent.py")),
               "the name this file is enforcing is no longer how the core reads it")

# 2. every Compose file that selects a device forwards every derived name, in every service.
for rel in COMPOSE_FILES:
    text = read(rel)
    blocks = env_blocks(text)
    check_true("compose/%s parses into services" % rel, bool(blocks), sorted(blocks))
    if "LAYA_DEVICE" not in text:
        NOTES.append("%s: no LAYA_DEVICE, so nothing to forward" % rel)
        continue
    for service, env in sorted(blocks.items()):
        for name in names:
            value = env.get(name)
            check_true("compose/%s/%s forwards %s" % (rel, service, name), value is not None,
                       "%s is read by the runtime but the rendered container would not get it"
                       % name)
            if value is not None:
                # A passthrough *with a default*, not a bare `${NAME}`: without the `:-` a
                # deployment that never mentions the variable is treated as asking for it, and
                # the default here has a meaning the core implements (empty = the checkpoint's
                # own amp_dtype).
                check_true("compose/%s/%s/%s is a host passthrough" % (rel, service, name),
                           re.match(r'^"\$\{%s:-[^"}]*\}"$' % name, value) is not None,
                           "%s: %r -- hardcoding or requiring a dtype here would pin it for "
                           "every user" % (name, value))

# 3. an empty host default has to mean "not asked for", which is what these images shipped before
#    any of this. Widened from the two dtype names to every name any Compose file forwards in the
#    `"${NAME:-}"` form, because the form is the deployment's promise and the promise is only kept
#    by the reader: an unset `LAYA_REVISION` has to leave the download on the Hub's default branch
#    the way `laya/revisions.py` says it does, and an unset `LAYA_MAX_CONCURRENT` has to leave the
#    server on 16. A reader that swaps in its own non-empty default when the container hands it
#    `""` silently outranks an operator who set nothing, so each read site is parsed, not trusted.
for rel in COMPOSE_FILES:
    for service, env in sorted(env_blocks(read(rel)).items()):
        for name, value in env.items():
            if NAME.fullmatch(name) and re.match(r'^"\$\{%s:-\}"$' % name, value):
                _unsafe = empty_unsafe_reads(name)
                check_true("compose/%s/%s/%s empty default is safe" % (rel, service, name),
                           not _unsafe, _unsafe)

# 4. the page that documents Compose variables carries a row per derived name.
docker_md = read(os.path.join("docs", "docker.md"))
for name in names:
    check_true("docs/docker.md has a %s row" % name,
               re.search(r"^\| `%s` \|" % name, docker_md, re.M) is not None,
               "a variable Compose forwards with no row in the table is one a reader cannot find")

# 4b. the other half of a passthrough. `"${NAME:-}"` promises the runtime's own default and says so
#     by being empty; `"${NAME:-x}"` promises `x`, which is a claim about what an operator gets for
#     free, and the table above prints exactly one such claim per name. Check 3 holds the empty
#     half to the code; this holds the non-empty half to the page.
#
#     The scope is every Compose file the tree has, not the four `COMPOSE_FILES` lists. That tuple
#     is this file's dtype sweep, and its subject is a device selection -- `compose.example.yml`
#     changes no device and forwards no dtype, so leaving it out of *that* loop is right. It was
#     also leaving it out of every loop, which is how a fifth file could ship a default the page
#     contradicts: its own header says it is for "request, checkpoint and secret-file mounts", and
#     it pinned `LAYA_MODEL` to `english` where the row prints `auto`. `examples/docker/quickstart.py`
#     reads that name, treats anything but `auto` as an explicit alias, and refuses it the moment
#     the operator enables the local-checkpoint mount the same file shows -- so the one line nobody
#     annotated both silences language routing and breaks the mount the file exists to demonstrate.
#
#     A file may still deviate, and the test for that is *where the deviation is said*: the entry
#     has to carry its own comment, on the line or in the block directly above it. The operator
#     reads the file at that line, so that is where a deliberate choice has to appear; a word that
#     happens to sit in some other comment in the same file is not a choice about this value. That
#     locality is what makes `english` the finding rather than `cuda` in `compose.cuda.yaml`, whose
#     row prints `cpu` / `cuda` both, and it is what lets an override file that bakes checkpoints
#     from another mirror pin `HF_HUB_OFFLINE=1` against a row that prints `0`, because that pin
#     sits under its own explanation. The stated choice covers the same variable's second copy in
#     the same file -- an override that forwards it to both services repeats one decision, not two.
#     The cost is honest: any comment on the line clears this, so the gate enforces that a deviation
#     is annotated where it is set, not that the annotation is true. Whether it is true is a
#     reviewer's call, and the line is now in front of one.
ALL_COMPOSE = tuple(sorted(f for f in os.listdir(ROOT)
                           if f.startswith("compose") and f.endswith((".yaml", ".yml"))))

PINNED = re.compile(r'^([A-Z][A-Z0-9_]+):\s*"\$\{([A-Z][A-Z0-9_]+):-([^"}]+)\}"'
                    r'(?:\s+(#.*))?$')


def pins(text: str) -> List[tuple]:
    """`(service, name, pinned default, annotation)` for each `"${NAME:-value}"` env entry.

    Same indent walk as `env_blocks`, and the same reason for it: a key at this indent under
    `environment:` is what the container gets, and nothing else is.

    The annotation is everything the entry itself carries: a trailing comment on its line plus the
    contiguous comment block directly above it. Group 4 has to be part of the match rather than a
    second pass, because the alternative is a shape that matches nothing -- an entry with a trailing
    comment would drop out of the sweep entirely instead of arriving annotated.
    """
    out: List[tuple] = []
    lines = text.splitlines()
    in_services = service = None
    in_env = False
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if indent == 0:
            in_services = stripped == "services:"
            service = in_env = None
            continue
        if not in_services:
            continue
        if indent == 2 and stripped.endswith(":"):
            service, in_env = stripped[:-1].strip(), False
            continue
        if service is None:
            continue
        if indent == 4 and stripped == "environment:":
            in_env = True
            continue
        if indent == 4:
            in_env = False
            continue
        if in_env and indent == 6 and not stripped.startswith("#") and ":" in stripped:
            match = PINNED.match(stripped)
            # The key and the interpolated name are the same variable here; a file that forwards
            # `FOO: "${BAR:-x}"` is a different drift, and the passthrough checks above own it.
            if not (match and match.group(1) == match.group(2)):
                continue
            above = []
            j = i - 1
            while j >= 0 and lines[j].lstrip().startswith("#"):
                above.insert(0, lines[j].strip())
                j -= 1
            out.append((service, match.group(1), match.group(3),
                        "\n".join(above + [(match.group(4) or "").strip()])))
    return out


# The two scopes have to stay two scopes, and the boundary has to be a reason rather than a list:
# what puts a file in the dtype sweep is that it can select a device the base file does not, which
# is the same fact that keeps the MPS row out of it. Derived, so a file another author adds is not a
# failure on its own -- an override that bakes weights and names no device joins neither sweep --
# while a hypothetical `compose.tpu.yaml` that pinned a device without joining the sweep would drop
# every dtype passthrough in it, which is the bug this whole file was written because of.
_devices = {v for _s, n, v, _a in pins(read("compose.yaml")) if n == "LAYA_DEVICE"}
check_true("compose/base file pins a device to compare against", _devices != set(),
           "compose.yaml has no LAYA_DEVICE default; retarget this if that changes")
check("compose/every file that selects a device the base file does not is in the dtype sweep",
      [rel for rel in ALL_COMPOSE
       if rel != "compose.yaml"
       and any(n == "LAYA_DEVICE" and v not in _devices for _s, n, v, _a in pins(read(rel)))
       and rel not in COMPOSE_FILES], [])
NOTES.append("compose/files outside the dtype sweep: %s"
             % sorted(set(ALL_COMPOSE) - set(COMPOSE_FILES)))


# name -> every default cell the page prints for it, across all of docker.md's tables. A row whose
# variable column names two spellings (`HF_TOKEN` / `HF_TOKEN_FILE`) documents both.
doc_rows: Dict[str, set] = {}
for _line in docker_md.splitlines():
    if not _line.startswith("|"):
        continue
    _cells = [c.strip() for c in _line.strip().strip("|").split("|")]
    if len(_cells) < 2:
        continue
    for _n in re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", _cells[0]):
        doc_rows.setdefault(_n, set()).add(_cells[1])

pinned_compared = 0
pinned_exempt = 0
for rel in ALL_COMPOSE:
    _found = pins(read(rel))
    # A deviation stated once covers every entry that repeats it in the same file: an override for
    # two services forwards the same variable twice, and the second copy carries no new decision.
    # Unioned per name, not per file -- the comment blocks inside `compose.yaml`'s environment
    # describe a *different* variable each, and treating them as cover for this one is how an
    # unexamined default passes a gate written to catch it.
    _said = {}
    for _s, n, _v, a in _found:
        _said[n] = (_said.get(n, "") + a).strip()
    for service, name, value, _annotated in _found:
        if name not in doc_rows:
            continue
        literals = {t for cell in doc_rows[name] for t in re.findall(r"`([^`]+)`", cell)}
        if not literals:
            # A row that describes its default in prose ("bundled request") states no value to
            # compare against. Counted below so the skip cannot quietly become the whole sweep.
            continue
        pinned_compared += 1
        if value not in literals and _said[name]:
            pinned_exempt += 1
        check_true("compose/%s/%s/%s default is what docs/docker.md prints" % (rel, service, name),
                   value in literals or _said[name] != "",
                   "the page's `%s` row prints %s, this file ships %r with no comment on its line "
                   "to say the choice is deliberate, and an operator following the page gets it "
                   "anyway"
                   % (name, sorted(literals), value))
# Non-vacuity: the comparison has to have run. Twenty-one of these hold on the current tree, all
# of them on-row; the floor is low enough to survive a table edit and high enough that a broken row
# parser or an empty `ALL_COMPOSE` reports itself instead of passing.
check_true("core/the default comparison compared the pinned entries", pinned_compared >= 15,
           "compared %d of %d Compose files; doc rows: %d" % (pinned_compared, len(ALL_COMPOSE),
                                                              len(doc_rows)))
# Recorded rather than asserted: today the tree clears zero of these on a comment, and the one file
# that will need the exemption is an override that bakes checkpoints from another mirror. Pinning
# the number to zero would fail that author's merge; leaving it unprinted would hide the day the
# exemption starts doing the work.
NOTES.append("compose/%d of %d pinned defaults are off-row and cleared by a comment on their line"
             % (pinned_exempt, pinned_compared))

# 5. the exclusion this file relies on stays falsifiable: MPS's row gate is not forwarded because
#    no Compose file here can select MPS. The day one does, this check fails and the decision has
#    to be made again rather than quietly missing a name.
agent_src = read(os.path.join("laya", "agent.py"))
mps_names = sorted(set(NAME.findall(agent_src)) - set(names))
check_true("core/MPS names are the only runtime names excluded", mps_names == ["LAYA_MPS_AMP_MIN_ROWS"],
           mps_names)
devices = set()
for rel in COMPOSE_FILES:
    devices.update(re.findall(r'LAYA_DEVICE: "\$\{LAYA_DEVICE:-([a-z0-9]+)\}"', read(rel)))
check("compose/devices any file defaults to", sorted(devices), ["cpu", "cuda"])
# The literal is deliberate, and matches how the rest of this repo gates prose: the reason is the
# thing that goes stale, so it has to be pinned, not just the variable's name.
mps_reason = re.search(r"`LAYA_MPS_AMP_MIN_ROWS`,[^.]*\.", docker_md, re.S)
check_true("docs/docker.md says why MPS is not forwarded",
           mps_reason is not None and "no image here can reach" in mps_reason.group(0),
           "the omission has to be explained where a reader meets it")

# 6. The same drift one level up. `dtype_names()` above is deliberately narrow -- a dtype
#    decision and the name it reads -- because that is the contract it was written for. The cost
#    is that a name read by any *other* module is invisible to it: `LAYA_REVISION` is read by
#    `laya/revisions.py`, it is documented, and it was forwarded by no Compose file at all. So an
#    operator's pin was interpolated, discarded without a warning, and the checkpoints loaded at
#    whatever the mutable default branch held.
#
#    This widens the derivation to every `LAYA_*` the package reads and checks the three-way
#    agreement -- package, docs, deployment -- in both directions, at "forwarded by at least one
#    file" granularity. Per-service granularity would be wrong the other way: `LAYA_PORT` belongs
#    to `laya-serve` and must not be demanded of the SDK service.
package_read: set = set()
for _dirpath, _dirs, _files in os.walk(os.path.join(ROOT, "laya")):
    for _filename in _files:
        if not _filename.endswith(".py"):
            continue
        _rel = os.path.relpath(os.path.join(_dirpath, _filename), ROOT)
        package_read |= set(ENVIRON_GET.findall(read(_rel)))

# A row may name two variables in one cell (``LAYA_API_KEY` / `LAYA_API_KEY_FILE`), so collect
# every name appearing anywhere in a table row as well. Matching only a lone backticked name
# would report a correctly documented variable as undocumented.
documented_rows = set(re.findall(r"^\| `(LAYA_[A-Z0-9_]+)` \|", docker_md, re.M))
for _row in re.findall(r"^\|.*$", docker_md, re.M):
    documented_rows |= set(NAME.findall(_row))

_mps_doc = re.search(r"`LAYA_MPS_AMP_MIN_ROWS`[^.]*\.", docker_md, re.S)
excluded = set(NAME.findall(_mps_doc.group(0))) if _mps_doc else set()

# `LAYA_MAX_CONCURRENT` and `LAYA_MAX_TOKEN_BUDGET` sat in this exclusion for one reason, and the
# reason was dated: `compose.http.yaml` is the only file that can forward them, and three PRs were
# inserting into that service's `environment:` block at the same time, so the names were left for
# whoever landed the cluster. The cluster landed -- #528 and #656 both merged 2026-09-29, and
# #528's `LAYA_SHA256_DIGESTS` is in the block this file now demands -- so the exclusion is empty
# and the two knobs are forwarded by `compose.http.yaml` with the rest of them.
#
# The tuple stays rather than disappearing, because an empty exclusion is the shape most likely to
# be re-filled by the next person who hits a conflict on that block. Re-deferring a name now costs
# a named check below, not a quiet pass.
DEFERRED = ()
required = sorted((package_read & documented_rows) - excluded - set(DEFERRED))

check_true("core/the widened set is not empty", len(required) >= 10, required)
check_true("core/the widened set includes the name this change forwards",
           "LAYA_REVISION" in required, sorted(required))
check_true("core/nothing is deferred out of the widened set", DEFERRED == (),
           "a deferred name is a documented knob that no shipped deployment can set")

forwarded_anywhere: set = set()
for _rel in COMPOSE_FILES:
    for _service, _env in env_blocks(read(_rel)).items():
        forwarded_anywhere |= {k for k in _env if NAME.fullmatch(k)}

for _name in required:
    check_true("compose/at least one file forwards %s" % _name, _name in forwarded_anywhere,
               "read by the runtime and documented, but no Compose file names it, so an "
               "operator's value is interpolated and then silently dropped")
    _values = [e[_name] for _r in COMPOSE_FILES
               for e in env_blocks(read(_r)).values() if _name in e]
    if _values:
        # Starts with `"${NAME:-` and ends with `}"`. The tail is matched loosely on purpose:
        # LAYA_THREADS nests a second default, `"${LAYA_THREADS:-${OMP_NUM_THREADS:-4}}"`,
        # which is still a host passthrough and is the idiom this file already used.
        check_true("compose/%s is a host passthrough" % _name,
                   all(v.startswith('"${%s:-' % _name) and v.endswith('}"') for v in _values),
                   _values)

# the other direction: a name Compose forwards with no row in the table is one a reader cannot
# find, which is the same drift seen from the docs side.
for _name in sorted(forwarded_anywhere - set(required)):
    check_true("docs/docker.md documents the forwarded %s" % _name,
               _name in documented_rows,
               "the deployment sets it and the page never mentions it")

# 6b. "Forwarded by at least one file" is the right ceiling for a name two services share, and
#     the wrong floor for one they do not. `LAYA_MAX_CONCURRENT` is read by `laya/serve.py` during
#     request admission, so a file that never starts a server forwarding it satisfies check 6
#     while the server that needs it still cannot be configured -- which is exactly how these two
#     knobs survived a merged gate. `compose.yaml`'s `laya` service runs a one-shot quickstart, and
#     `compose.http.yaml`'s own header says why that is not the same container: "`laya-serve` is a
#     separate service and overrides for `laya` never reach it".
#
#     The service a variable belongs to is taken from the page rather than from a literal here,
#     because the page is the artifact that says which name belongs where: the table under
#     "Server configuration" opens with "These apply to the `laya-serve` service only."
server_doc = re.search(r"^### Server configuration\n(.*?)(?=^#{2,3} )", docker_md, re.M | re.S)
check_true("docs/docker.md has a Server configuration section", server_doc is not None,
           "the derivation below needs the page to own the server variables")
if server_doc:
    _body = server_doc.group(1)
    _owner = re.search(r"These apply to the `([a-z0-9-]+)` service only\.", _body)
    check_true("docs/docker.md names the service the server table applies to", _owner is not None,
               _body.splitlines()[:3])
    server_names = sorted(set(re.findall(r"^\| `(LAYA_[A-Z0-9_]+)` \|", _body, re.M)) & package_read)
    check_true("core/the server table is not empty", len(server_names) >= 8, server_names)
    if _owner:
        _service = _owner.group(1)
        # Compose merges the `environment` of every file named on one command line, so the unit
        # that has to be complete is the pairing, and `compose.cuda.yaml` / `compose.spark.yaml`
        # legitimately name the service with three device keys and nothing else. The pairings are
        # the commands this page tells an operator to run, taken from the lines that name the
        # service -- so the demand is on what a reader boots, not on a file list invented here.
        _pairings = sorted({tuple(re.findall(r"-f ([a-z0-9._-]+\.ya?ml)", _line))
                            for _line in docker_md.splitlines()
                            if _service in _line and "-f " in _line})
        check_true("docs/docker.md shows a Compose command for the %s service" % _service,
                   bool(_pairings), _pairings)
        for _pair in _pairings:
            # A command on the page that names a file the tree does not have is the same class of
            # drift as a variable the page names and the deployment drops, so it is checked here
            # rather than crashing the suite on a FileNotFoundError.
            _missing = [rel for rel in _pair if not os.path.exists(os.path.join(ROOT, rel))]
            check_true("compose/%s names files that exist" % "+".join(_pair), not _missing, _missing)
            _merged: Dict[str, str] = {}
            for _rel in _pair:
                if _rel not in _missing:
                    _merged.update(env_blocks(read(_rel)).get(_service, {}))
            for _name in server_names:
                check_true("compose/%s forwards %s to %s" % ("+".join(_pair), _name, _service),
                           _name in _merged,
                           "the table applies to `%s` only and this command boots it, so an "
                           "operator's value is interpolated and dropped" % _service)
                if _name in _merged:
                    # The same loose tail check 6 uses, so `LAYA_THREADS`'s nested
                    # `"${LAYA_THREADS:-${OMP_NUM_THREADS:-4}}"` stays a host passthrough.
                    check_true("compose/%s/%s is a host passthrough" % ("+".join(_pair), _name),
                               _merged[_name].startswith('"${%s:-' % _name)
                               and _merged[_name].endswith('}"'),
                               "%s: %r" % (_name, _merged[_name]))
        # The names this change forwards, pinned as literals the way check 5 pins the MPS
        # exclusion: the pairing above would pass if another file started carrying them, and
        # `server_names` would pass if the page stopped documenting them. `compose.http.yaml` is
        # the file that owns this service's environment block, so it is the file that has to lose
        # one for this check to go red.
        _env = env_blocks(read("compose.http.yaml")).get(_service, {})
        check_true("compose/compose.http.yaml defines the %s service" % _service,
                   _service in env_blocks(read("compose.http.yaml")), sorted(_env))
        for _name in ("LAYA_MAX_CONCURRENT", "LAYA_MAX_TOKEN_BUDGET", "LAYA_REVISION"):
            check_true("compose/compose.http.yaml/%s forwards the %s this change adds"
                       % (_service, _name), _name in _env,
                       "it is documented and read by the runtime, and the service that needs it "
                       "is the one that never named it")

        # 6c. The default half of a passthrough is the part that can lie. `"${NAME:-}"` hands the
        #     container the empty string, `"${NAME:-8}"` hands it 8, and a plain `"8"` hands it a
        #     value nobody can move -- three different promises, and the page prints one of them.
        #     So each forwarded name is checked against its own row, and an empty default has to
        #     be a value the reader treats as "not asked for": the property check 3 pins for the
        #     dtype names, widened to the server table, where the empty form is the norm.
        _serve_src = read(os.path.join("laya", "serve.py"))
        _defaults = dict(re.findall(r"^(DEFAULT_[A-Z0-9_]+) = (\S+)$", _serve_src, re.M))
        _const_pinned = 0
        for _name, _cell in re.findall(r"^\| `(LAYA_[A-Z0-9_]+)` \| ([^|]+) \|", _body, re.M):
            _value = _env.get(_name)
            if not _value or not _value.startswith('"${%s:-' % _name) or not _value.endswith('}"'):
                continue
            _tail = _value[len('"${%s:-' % _name):-2]
            _cell = _cell.strip()
            if _tail.startswith("${"):
                _inner = re.match(r"^\$\{([A-Z0-9_]+)", _tail)
                check_true("compose/compose.http.yaml/%s nests a default its row names" % _name,
                           _inner is not None and _inner.group(1) in _cell,
                           "%s: row prints %r" % (_name, _cell))
            elif _tail:
                check_true("compose/compose.http.yaml/%s default is the documented %s"
                           % (_name, _cell),
                           _cell.startswith("`") and _cell.strip("`") == _tail,
                           "an unset %s gets a value the page does not print, so the row and the "
                           "deployment disagree about the container an operator gets for free"
                           % _name)
            else:
                _unsafe = empty_unsafe_reads(_name)
                check_true("compose/compose.http.yaml/%s empty default is safe" % _name,
                           not _unsafe, _unsafe)
                _const = "DEFAULT_%s" % _name[len("LAYA_"):]
                if _cell.startswith("`") and _const in _defaults:
                    _const_pinned += 1
                    check("docs/docker.md prints the same %s default as laya/serve.py" % _name,
                          _cell.strip("`"), _defaults[_const])
        check_true("core/both knobs this change forwards have a module default to pin",
                   _const_pinned == 2, _const_pinned)
        for _name in ("LAYA_MAX_CONCURRENT", "LAYA_MAX_TOKEN_BUDGET"):
            check_true("docs/docker.md keeps the %s row in the server table" % _name,
                       re.search(r"^\| `%s` \|" % _name, _body, re.M) is not None,
                       "the two sections above derive their demand from this table, so a row "
                       "deleted by accident retires the gate instead of reopening the question")

# 7. the real renderer, when this machine has one: the rendered environment must carry the value.
if shutil.which("docker"):
    env = dict(os.environ, LAYA_CUDA_AMP="fp16")
    proc = subprocess.run(["docker", "compose", "-f", "compose.yaml", "-f", "compose.cuda.yaml",
                           "config"], cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        rendered = proc.stdout
        blocks = env_blocks(rendered)
        got = sorted([str(b.get("LAYA_CUDA_AMP", "<absent>")) for b in blocks.values()])
        check_true("render/both GPU services get LAYA_CUDA_AMP=fp16",
                   got == ["fp16", "fp16"],
                   {s: b.get("LAYA_CUDA_AMP") for s, b in blocks.items()})
        check_true("render/LAYA_MPS_AMP_MIN_ROWS stays out of the rendered environment",
                   all("LAYA_MPS_AMP_MIN_ROWS" not in b for b in blocks.values()),
                   sorted(k for b in blocks.values() for k in b if "MPS" in k))
    else:
        NOTES.append("docker compose config exited %d: %s" % (proc.returncode,
                                                              proc.stderr.strip()[:200]))

    # The HTTP file is what this change edits, and an interpolated name that never reaches the
    # container is the failure the textual checks above are reasoning about. Both arms: set, and
    # left alone. The unset arm is the one an operator actually boots, and it must show the empty
    # default arriving as empty rather than as the literal `${LAYA_MAX_CONCURRENT:-}`.
    _KN = ("LAYA_MAX_CONCURRENT", "LAYA_MAX_TOKEN_BUDGET", "LAYA_REVISION")
    _arms = [(dict(zip(_KN, ("4", "16384", "reviewed"))), dict(zip(_KN, ("4", "16384", "reviewed")))),
             ({}, dict.fromkeys(_KN, ""))]
    for _set, _expect in _arms:
        proc = subprocess.run(["docker", "compose", "-f", "compose.yaml", "-f", "compose.http.yaml",
                               "config"], cwd=ROOT, env=dict(os.environ, **_set),
                               capture_output=True, text=True)
        if proc.returncode != 0:
            NOTES.append("docker compose -f compose.http.yaml config exited %d: %s"
                         % (proc.returncode, proc.stderr.strip()[:200]))
            continue
        _rendered = env_blocks(proc.stdout).get("laya-serve", {})
        for _name, _want in sorted(_expect.items()):
            # The renderer re-quotes: `4` comes back as `'4'` and an empty default as `""`.
            # Stripping one layer of quotes is what the container actually receives as a string.
            check("render/%s reaches the server as %r" % (_name, _want),
                  str(_rendered.get(_name, "<absent>")).strip("'\""), _want)
else:
    NOTES.append("docker not installed: checks 1-6 still hold; the render check did not run")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for note in NOTES:
    print("  NOTE " + note)
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
