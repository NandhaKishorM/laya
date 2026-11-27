"""Freeze development-only k choices, or print the selected follow-up commands.

This reader needs only the standard library and never loads a model. It checks
report consistency; it cannot establish whether someone inspected results
outside the report directory. A new selection records its real creation time,
not the historical timestamp of an earlier experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

VARIANTS = ("minilm", "nomic-qd", "nomic-qq")
KS = (5, 8, 12, 20, 32, 48)
METHODS = ("shortlist", "combined")
SOURCES = ("amended_comparison.py", "interleaved.py", "compare_embeddings.py",
           "embedders.py", "diagnostics.py", "benchmark.py")
SCHEMA = "laya-dedicated-shortlist/selection-v1"
CHECKPOINT = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
DATASET_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"
SPECS = {
    "minilm": {"repo": "sentence-transformers/all-MiniLM-L6-v2",
               "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
               "pooling": "mean", "dimensions": 384, "max_length": 256,
               "query_prefix": "", "option_prefix": "", "dtype": "float32", "device": "cpu"},
    "nomic-qd": {"repo": "nomic-ai/modernbert-embed-base",
                 "revision": "d556a88e332558790b210f7bdbe87da2fa94a8d8",
                 "pooling": "mean", "dimensions": 768, "max_length": 512,
                 "query_prefix": "search_query: ", "option_prefix": "search_document: ",
                 "dtype": "float32", "device": "cpu"},
}
SPECS["nomic-qq"] = dict(SPECS["nomic-qd"], option_prefix="search_query: ")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("non-finite JSON value: " + value)

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def read(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value, size=64):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{%d}" % size, value) is not None


def parameters(**changes):
    result = dict(mode="quality", dataset="banking77", split="dev", limit=200, offset=0,
                  seed=0, variant="minilm", strategies="shortlist,combined", k=20,
                  group_size=16, permutation_seed=None, diagnose=True, device="cpu",
                  threads=4, passes=3, arms_file=None, local_files_only=True)
    result.update(changes)
    return result


def stage_summary(cases):
    counts = {name: sum(c["outcome"] == name for c in cases)
              for name in ("correct", "retrieval_miss", "elimination_miss", "final_miss")}
    recalled = len(cases) - counts["retrieval_miss"]
    survived = recalled - counts["elimination_miss"]
    return dict(counts=counts, rows=len(cases), accuracy=counts["correct"] / len(cases),
                recall_at_k=recalled / len(cases),
                survival_on_recalled=survived / recalled if recalled else None,
                accuracy_on_finalists=counts["correct"] / survived if survived else None)


def validate_case(row, method, k):
    kept, final = row["shortlist"], row["finalists"]
    require(isinstance(kept, list) and len(kept) == k and
            all(isinstance(x, str) and x for x in kept) and len(set(kept)) == k, "invalid shortlist")
    require(row["decision_order"] == kept, "natural decision order differs from retrieval")
    require(isinstance(final, list) and final and len(final) == len(set(final)) and
            set(final) <= set(kept) and row["choice"] in final, "invalid finalists or choice")
    require(isinstance(row["expected"], str) and row["expected"], "missing gold label")
    require(type(row["correct"]) is bool and row["correct"] == (row["choice"] == row["expected"]),
            "correct must be a boolean matching choice and gold")
    expected = ("retrieval_miss" if row["expected"] not in kept else
                "elimination_miss" if row["expected"] not in final else
                "correct" if row["correct"] else "final_miss")
    require(row["outcome"] == expected, "outcome differs from actual candidate sets")
    rank = kept.index(row["expected"]) + 1 if row["expected"] in kept else None
    require(row["gold_rank"] == rank and type(row["gold_rank"]) is type(rank), "gold rank differs")
    calls = 2 if method == "combined" and k > 16 else 1
    require(type(row["decision_calls"]) is int and row["decision_calls"] == calls,
            "wrong number of decision calls")
    require(len(row["groups"]) == len(row["calls"]) == calls, "missing round evidence")
    if calls == 1:
        require(final == kept and row["groups"] == [[kept]], "single-pass candidates differ")
    else:
        groups = row["groups"][0]
        n = (k + 15) // 16
        require(groups == [kept[i * k // n:(i + 1) * k // n] for i in range(n)],
                "elimination groups differ from balanced core grouping")
        require(len(final) == n and all(label in group for label, group in zip(final, groups))
                and row["groups"][1] == [final], "finalists do not preserve one winner per group")
    for call, groups in zip(row["calls"], row["groups"]):
        require([q["labels"] for q in call["questions"].values()] == groups,
                "token diagnostics and actual groups differ")


def validate_report(report, variant, k):
    require(report["schema"] == "laya-dedicated-shortlist/amendment-v2" and
            report["status"] == "complete", "grid report incomplete or wrong schema")
    params = parameters(variant=variant, k=k)
    require(report["parameters"] == params, "grid parameters differ")
    for key in ("limit", "offset", "seed", "k", "group_size", "threads", "passes"):
        require(type(report["parameters"][key]) is int, "boolean or noninteger parameter: " + key)
    require(report["parameters"]["diagnose"] is True and report["parameters"]["local_files_only"] is True,
            "missing diagnostics or offline policy")
    require(report["embedder"] == SPECS[variant], "model revision/prefix/pooling differs")
    require(set(report["source_sha256"]) == set(SOURCES) and
            all(digest(x) for x in report["source_sha256"].values()), "missing source identity")
    env = report["environment"]
    require(digest(env["code_commit"], 40) and env["python"] and env["platform"] and
            set(env["versions"]) == {"torch", "transformers", "numpy"}, "missing execution identity")
    require(env["runner_sha256"] == {k: report["source_sha256"][k] for k in SOURCES[2:]},
            "runner identities disagree")
    checkpoint = report["checkpoint"]
    require(checkpoint["revision"] == CHECKPOINT and
            set(checkpoint["sha256"]) == {"model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json"}
            and all(digest(x) for x in checkpoint["sha256"].values()), "invalid checkpoint identity")
    rows, identity = report["rows"], report["identity"]
    require(len(rows) == 200 and len({r["id"] for r in rows}) == 200 and
            all(set(r) == {"id", "state_sha256"} and digest(r["state_sha256"]) and
                re.fullmatch(r"banking77:train:\d+", r["id"]) for r in rows), "invalid dev200 row identities")
    require(identity["dataset"] == "banking77" and identity["split"] == "train" and
            identity["dataset_revision"] == DATASET_REVISION and identity["seed"] == 0 and
            type(identity["seed"]) is int and identity["offset"] == 0 and type(identity["offset"]) is int and
            identity["sampling"] == "seeded permutation prefix" and
            identity["checkpoint_revision"] == CHECKPOINT and digest(identity["questions_sha256"]) and
            identity["case_ids_sha256"] == fingerprint([r["id"] for r in rows]), "invalid dataset/question identity")
    require(report["ordering"] == {"seed": None, "global_labels": None, "policy": "natural"},
            "grid order must stay natural")
    runtime = report["runtime"]
    placement = runtime["placement_before"]
    require(runtime["requested_device"] == runtime["actual_laya_device"] ==
            runtime["actual_embedder_device"] == "cpu" and type(runtime["torch_threads"]) is int and
            runtime["torch_threads"] == 4 and runtime["placement_after"] == placement,
            "runtime changed or differs from CPU/four-thread protocol")
    require(placement["requested_device"] == "cpu" and placement["cpu_fallback_count"] == 0 and
            placement["amp_failures"] == 0 and placement["amp_enabled"] is False and
            placement["autocast_dtype"] == "torch.float32", "runtime fallback or precision differs")
    for model in ("laya", "encoder"):
        require(placement[model]["device"] == "cpu" and
                placement[model]["parameter_dtypes"] == ["torch.float32"], "model placement differs")
    require(set(report["strategies"]) == set(METHODS), "missing or unexpected grid strategy")
    gold = None
    for method in METHODS:
        arm = report["strategies"][method]
        require(arm["status"] == "complete" and arm["strategy"] == method and type(arm["k"]) is int and
                arm["k"] == k and type(arm["group_size"]) is int and arm["group_size"] == 16 and
                arm["identity"] == dict(identity, embedder_id=fingerprint(report["embedder"])) and
                arm["runtime"] == placement, "arm identity, runtime or configuration differs")
        cases = arm["cases"]
        require([{key: row[key] for key in ("id", "state_sha256")} for row in cases] == rows,
                "arm samples differ from report rows")
        for row in cases:
            validate_case(row, method, k)
        current_gold = [row["expected"] for row in cases]
        require(gold is None or current_gold == gold, "gold differs between strategies")
        gold = current_gold
        require(arm["stages"] == stage_summary(cases), "stage summary differs from cases")
    left, right = (report["strategies"][name] for name in METHODS)
    require([c["shortlist"] for c in left["cases"]] == [c["shortlist"] for c in right["cases"]],
            "same embedder retrieved different candidates")
    if k <= 16:
        require(right.get("equivalent_to") == "shortlist" and right.get("degenerate_single_pass") is True and
                left["cases"] == right["cases"], "missing exact combined/single-pass alias")
    else:
        require("equivalent_to" not in right and "degenerate_single_pass" not in right, "invalid alias above group size")
    return gold


def load_grid(directory):
    directory = Path(directory)
    expected = {f"grid-dev200-{variant}-k{k}.json" for variant in VARIANTS for k in KS}
    require({p.name for p in directory.glob("grid-dev200-*.json")} == expected, "incomplete or unexpected development grid")
    grid, hashes, anchor, runtimes = {}, {}, None, {}
    for variant in VARIANTS:
        for k in KS:
            path = directory / f"grid-dev200-{variant}-k{k}.json"
            raw = path.read_bytes()
            report = loads(raw.decode("utf-8"))
            try:
                gold = validate_report(report, variant, k)
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
            runtime = report["runtime"]
            require(runtimes.setdefault(variant, runtime) == runtime, "runtime differs within model variant")
            common = {key: report[key] for key in ("rows", "identity", "checkpoint", "source_sha256", "environment")}
            # Encoder buffer dtypes legitimately differ between MiniLM and Nomic.
            common["runtime"] = {key: value for key, value in runtime.items() if not key.startswith("placement_")}
            common["laya_placement"] = {key: value for key, value in runtime["placement_before"].items()
                                       if key != "encoder"}
            common["gold"] = gold
            require(anchor is None or common == anchor, "grid cells do not share samples, labels or execution identity")
            anchor = common
            grid[variant, k] = report
            hashes[f"{variant}/k{k}"] = hashlib.sha256(raw).hexdigest()
    return grid, hashes, anchor


def choose(grid):
    require(set(grid) == {(v, k) for v in VARIANTS for k in KS}, "incomplete development grid")
    selected = {}
    for family, variants in (("minilm", ("minilm",)), ("nomic", ("nomic-qd", "nomic-qq"))):
        selected[family] = {}
        for method in METHODS:
            cells = [dict(variant=v, k=k, correct=sum(row["correct"] for row in grid[v, k]["strategies"][method]["cases"]))
                     for v in variants for k in KS]
            selected[family][method] = min(cells, key=lambda c: (-c["correct"], c["k"], c["variant"] == "nomic-qq"))
    return selected


def selected_jobs(selected, split, limit, offset=0):
    configs = {}
    for family in selected.values():
        for method, cell in family.items():
            configs.setdefault((cell["variant"], cell["k"]), []).append(method)
    seen = set()
    for (variant, k), methods in sorted(configs.items()):
        methods = list(methods)
        if "combined" in methods and k <= 16 and "shortlist" not in methods:
            methods.insert(0, "shortlist")
        if split == "test" and variant not in seen:
            methods.insert(0, "top1")
            seen.add(variant)
        yield f"selected-{split}{limit}-{variant}-k{k}.json", parameters(
            variant=variant, split=split, limit=limit, offset=offset, k=k, strategies=",".join(methods))


def test_jobs(selected):
    jobs = list(selected_jobs(selected, "test", 500))
    sources = {p["variant"]: name for name, p in jobs if "top1" in p["strategies"].split(",")}
    return [dict(report=name, parameters=p, matched_top1_report=sources[p["variant"]]) for name, p in jobs]


def reject_existing_test(directory):
    evidence = [p for pattern in ("selected-test*", "latency-selected*")
                for p in Path(directory).glob(pattern)]
    require(not evidence, "selected test/timing evidence already exists; preserve the original selection timestamp")


def freeze(directory, destination):
    directory, destination = Path(directory), Path(destination)
    require(not destination.exists(), "refusing to overwrite a selection")
    for path in {directory.resolve(), destination.parent.resolve()}:
        reject_existing_test(path)
    grid, hashes, common = load_grid(directory)
    selected = choose(grid)
    artifact = dict(schema=SCHEMA, selected=selected, based_only_on="BANKING77 first200 frozen dev rows",
                    selection_rule="max correct; ties smaller k, then nomic-qd",
                    grid_sha256=hashes, selected_test_jobs=test_jobs(selected), identity=common["identity"],
                    sample_rows_sha256=fingerprint(common["rows"]), gold_sha256=fingerprint(common["gold"]),
                    source_sha256=common["source_sha256"], environment=common["environment"],
                    checkpoint=common["checkpoint"],
                    scope="Exploratory supplement after visible fixed-k test results; not a new test-blind study",
                    saved_before_new_test_utc=datetime.now(timezone.utc).isoformat())
    for path in {directory.resolve(), destination.parent.resolve()}:
        reject_existing_test(path)
    require(all(sha256(directory / f"grid-dev200-{variant}-k{k}.json") == hashes[f"{variant}/k{k}"]
                for variant in VARIANTS for k in KS), "grid changed while selecting; rerun before creating a selection")
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return artifact


def timing_arms(chosen):
    arms = [dict(id=method, strategy=method, k=chosen[method]["k"], group_size=16,
                 embedder=chosen[method]["variant"]) for method in METHODS]
    arms.insert(1, dict(id="tournament", strategy="tournament", k=20, group_size=16, embedder=None))
    arms.extend(dict(id="top1-" + variant, strategy="top1", k=1, group_size=16, embedder=variant)
                for variant in sorted({cell["variant"] for cell in chosen.values()}))
    return arms


def command_plan(directory, selection):
    directory = Path(directory)
    grid, hashes, _common = load_grid(directory)
    require(selection["grid_sha256"] == hashes and selection["selected"] == choose(grid),
            "selection differs from complete development evidence")
    require(selection["selected_test_jobs"] == test_jobs(selection["selected"]), "matched test jobs differ")
    timestamp = selection["saved_before_new_test_utc"]
    require(isinstance(timestamp, str) and datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None,
            "selection needs its original timezone-aware timestamp")
    jobs, arms_files = [], {}
    for split, limit, offset in (("dev", 400, 200), ("test", 500, 0)):
        jobs.extend(selected_jobs(selection["selected"], split, limit, offset))
    for family, chosen in selection["selected"].items():
        filename = f"latency-selected-banking77-{family}-arms.json"
        arms_files[filename] = timing_arms(chosen)
        jobs.append((f"latency-selected-banking77-{family}.json", parameters(mode="latency", split="test",
                    limit=50, seed=1, variant="minilm" if family == "minilm" else "nomic-qd",
                    strategies="shortlist,tournament,combined,top1", arms_file=str(directory / filename))))
    commands = []
    for filename, params in jobs:
        argv = ["python", "research/benchmarks/tournament/amended_comparison.py"]
        for key, value in params.items():
            if value is not None and value is not False:
                argv.append("--" + key.replace("_", "-"))
                if value is not True:
                    argv.append(str(value))
        argv.extend(("--out", str(directory / filename)))
        commands.append(dict(report=filename, argv=argv))
    return dict(selection_time=timestamp, instructions="Write arms_files under --reports unchanged; run argv from repository root, serially.",
                selected=selection["selected"], commands=commands, arms_files=arms_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new selection JSON; never overwritten")
    parser.add_argument("--selection", type=Path, help="existing, frozen selection for command generation")
    parser.add_argument("--commands", action="store_true", help="print command and arms-file definitions as JSON; do not run them")
    args = parser.parse_args()
    try:
        if args.commands:
            require(args.selection is not None and args.out is None, "--commands requires --selection and forbids --out")
            print(json.dumps(command_plan(args.reports, read(args.selection)), indent=2, ensure_ascii=False, allow_nan=False))
        else:
            require(args.out is not None and args.selection is None, "selection requires --out; use --commands to read an existing selection")
            freeze(args.reports, args.out)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
