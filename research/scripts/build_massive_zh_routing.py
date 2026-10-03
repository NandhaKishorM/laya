"""Build a traceable choice dataset for the existing Chinese voice-routing eval.

The source is Amazon MASSIVE 1.1. Hard labels remain hard labels in the
evaluation files. The optional training export represents them as one-hot
targets and is explicitly marked as such; it is not a teacher distribution.
"""

import argparse
import hashlib
import json
import tarfile
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
SOURCE_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
SOURCE_REVISION = "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740"
SOURCE_MEMBER = "1.1/data/zh-CN.jsonl"
BENCHMARK = ROOT / "research/evals/zh_decision_bench.jsonl"
SPLITS = ("train", "dev", "test")
SCENARIO_LABELS = {
    "calendar": "日历安排",
    "alarm": "闹钟计时",
    "audio": "音量控制",
    "music": "音乐点播",
    "weather": "天气查询",
    "transport": "交通出行",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain_archive(path, *, download):
    path = Path(path)
    if not path.exists():
        if not download:
            raise ValueError("MASSIVE archive is missing; pass --download or --archive PATH")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        try:
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, open(temporary, "wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            if file_sha256(temporary) != SOURCE_SHA256:
                raise ValueError("downloaded MASSIVE archive does not match the pinned SHA256")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    actual = file_sha256(path)
    if actual != SOURCE_SHA256:
        raise ValueError("MASSIVE archive SHA256 mismatch: %s" % actual)
    return path


def read_source(archive):
    with tarfile.open(archive, "r:gz") as bundle:
        handle = bundle.extractfile(SOURCE_MEMBER)
        if handle is None:
            raise ValueError("MASSIVE archive has no %s" % SOURCE_MEMBER)
        for line_number, raw in enumerate(handle, 1):
            if raw.strip():
                try:
                    yield json.loads(raw)
                except ValueError as exc:
                    raise ValueError("%s:%d: invalid JSON" % (SOURCE_MEMBER, line_number)) from exc


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def benchmark_contract(path):
    question = None
    held_out = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            if "voice_assistant_routing" not in row.get("tags", []):
                continue
            current = row["questions"]["route"]
            if current.get("type") != "choice" or set(current["criteria"]) != set(SCENARIO_LABELS.values()):
                raise ValueError("benchmark voice-routing question is incompatible at line %d" % line_number)
            if question is not None and current != question:
                raise ValueError("benchmark voice-routing questions differ at line %d" % line_number)
            question = current
            held_out.add(normalized_text(row["state"]))
    if question is None:
        raise ValueError("benchmark has no voice-routing cases")
    return question, held_out


def source_id_key(row):
    value = row["source_id"]
    return (0, int(value)) if value.isdecimal() else (1, value)


def build_rows(source, question, held_out):
    """Keep official partitions while removing exact normalized text leakage."""
    candidates = {split: [] for split in SPLITS}
    counts = Counter()
    ids = set()
    labels_by_text = {}
    for row in source:
        if row.get("locale") != "zh-CN" or row.get("scenario") not in SCENARIO_LABELS:
            continue
        counts["relevant_source"] += 1
        split, source_id, text = row.get("partition"), str(row.get("id")), row.get("utt")
        if split not in SPLITS or source_id == "None" or not isinstance(text, str):
            raise ValueError("MASSIVE row has invalid partition, id or utterance")
        if source_id in ids:
            raise ValueError("duplicate MASSIVE source id: %s" % source_id)
        ids.add(source_id)
        text = text.strip()
        if not 4 <= len(text) <= 60 or not row.get("judgments") or not all(
            judgment.get("intent_score") == 1 for judgment in row["judgments"]
        ):
            counts["quality_excluded"] += 1
            continue
        key = normalized_text(text)
        label = SCENARIO_LABELS[row["scenario"]]
        previous = labels_by_text.setdefault(key, label)
        if previous != label:
            raise ValueError("conflicting labels for normalized text: source id %s" % source_id)
        candidates[split].append({"source_id": source_id, "scenario": row["scenario"], "state": text,
                                  "label": label, "key": key})
        counts["quality_selected"] += 1

    if any(not candidates[split] for split in SPLITS):
        raise ValueError("each official split must contain selected cases")

    # Protect the final test set first, then dev, then train. A duplicate keeps
    # its original partition or is excluded; nothing is reassigned to a split.
    seen = set()
    output = {split: [] for split in SPLITS}
    for split in reversed(SPLITS):
        for row in sorted(candidates[split], key=source_id_key):
            if row["key"] in held_out:
                counts["benchmark_overlap_excluded_" + split] += 1
                continue
            if row["key"] in seen:
                counts["duplicate_text_excluded_" + split] += 1
                continue
            seen.add(row["key"])
            output[split].append({
                "state": row["state"], "questions": {"route": question},
                "expected": {"route": row["label"]}, "language": "zh-CN",
                "source": "AmazonScience/massive:1.1", "source_id": row["source_id"],
                "license": "CC BY 4.0",
                "tags": ["massive", "voice_assistant_routing", "scenario:" + row["scenario"],
                         "source_id:" + row["source_id"]],
            })
    if any(not output[split] for split in SPLITS):
        raise ValueError("deduplication left an official split empty")
    return output, dict(sorted(counts.items()))


def hard_target_rows(rows):
    labels = tuple(rows[0]["questions"]["route"]["criteria"])
    for row in rows:
        label = row["expected"]["route"]
        if label not in labels:
            raise ValueError("hard target label is not in the choice criteria: %s" % label)
        yield {"state": row["state"], "questions": row["questions"],
               "gold": {"route": {"label": label,
                                  "probabilities": {candidate: float(candidate == label) for candidate in labels}}},
               "target_source": "hard_label_one_hot", "tags": row["tags"],
               "source": row["source"], "source_id": row["source_id"], "license": row["license"]}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build(archive, output_dir, *, benchmark=BENCHMARK, export_hard_targets=False):
    question, held_out = benchmark_contract(benchmark)
    rows, audit = build_rows(read_source(archive), question, held_out)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty to avoid mixing results from different builds")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        path = output_dir / (split + "_eval.jsonl")
        write_jsonl(path, rows[split])
        files[path.name] = {"rows": len(rows[split]), "sha256": file_sha256(path),
                            "labels": dict(sorted(Counter(r["expected"]["route"] for r in rows[split]).items()))}
    if export_hard_targets:
        path = output_dir / "train_hard_targets.jsonl"
        write_jsonl(path, hard_target_rows(rows["train"]))
        files[path.name] = {"rows": len(rows["train"]), "sha256": file_sha256(path),
                            "target_source": "hard_label_one_hot; not teacher probabilities"}
    archive_hash = file_sha256(archive)
    manifest = {
        "source": {"name": "Amazon MASSIVE 1.1", "url": SOURCE_URL, "archive_sha256": archive_hash,
                   "pinned_archive_sha256": SOURCE_SHA256, "matches_pinned_source": archive_hash == SOURCE_SHA256,
                   "huggingface_revision": SOURCE_REVISION, "locale": "zh-CN", "license": "CC BY 4.0"},
        "benchmark_sha256": file_sha256(benchmark),
        "task": "six-way Chinese voice routing, choice only",
        "filter": "unanimous intent_score=1; utterance length 4..60; normalized exact-text deduplication",
        "split_priority_for_duplicate_text": ["test", "dev", "train"],
        "audit": audit, "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, default=ROOT / ".cache/amazon-massive-dataset-1.1.tar.gz")
    parser.add_argument("--download", action="store_true", help="download the pinned source archive if absent")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/massive-zh-routing")
    parser.add_argument("--export-hard-targets", action="store_true",
                        help="also export explicitly marked one-hot hard-label training targets")
    args = parser.parse_args()
    archive = obtain_archive(args.archive, download=args.download)
    manifest = build(archive, args.output_dir, export_hard_targets=args.export_hard_targets)
    print(json.dumps({"output_dir": str(args.output_dir), "audit": manifest["audit"],
                      "files": manifest["files"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
