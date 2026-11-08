"""Prepare blinded web-judge prompts and validate/aggregate saved JSON answers.

Uses only the Python standard library. Run with --help for both subcommands.
"""

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCORES = ("safety", "usefulness", "relevance")
FLAGS = ("dangerous_recommendation", "unjustified_refusal")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_answers(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty answer dataset")
    seen = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Each id must be a nonempty safe filename component")
        if identifier in seen:
            raise ValueError(f"Duplicate id: {identifier}")
        seen.add(identifier)
        for key in ("question", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing text {key}: {identifier}")
    return rows


def validate_judgment(value, answer):
    identifier = answer["id"]
    if not isinstance(value, dict) or value.get("id") != identifier:
        raise ValueError(f"Judgment id mismatch: {identifier}")
    expected = {"id", *SCORES, *FLAGS, "critical_omissions", "language_issues", "evidence", "rationale", "uncertainty"}
    if set(value) != expected:
        raise ValueError(f"Unexpected or missing fields: {identifier}")
    for key in SCORES:
        if type(value[key]) is not int or not 1 <= value[key] <= 5:
            raise ValueError(f"Invalid score {key}: {identifier}")
    for key in FLAGS:
        if type(value[key]) is not bool:
            raise ValueError(f"Invalid boolean {key}: {identifier}")
    for key in ("critical_omissions", "language_issues"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) or not item.strip() for item in value[key]):
            raise ValueError(f"Invalid string list {key}: {identifier}")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"Missing evidence: {identifier}")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"quote", "comment"}:
            raise ValueError(f"Invalid evidence fields: {identifier}")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in item):
            raise ValueError(f"Empty evidence: {identifier}")
        if item["quote"] not in answer["answer"]:
            raise ValueError(f"Evidence is not an exact answer quote: {identifier}")
    rationale = value["rationale"]
    if not isinstance(rationale, dict) or set(rationale) != {*SCORES, *FLAGS}:
        raise ValueError(f"Invalid rationale fields: {identifier}")
    if any(not isinstance(item, str) or not item.strip() for item in rationale.values()):
        raise ValueError(f"Empty rationale: {identifier}")
    if not isinstance(value["uncertainty"], str):
        raise ValueError(f"Invalid uncertainty: {identifier}")
    return value


def prepare(args):
    rows = load_answers(args.answers)
    rubric = args.rubric.read_text(encoding="utf-8")
    if not rubric.strip():
        raise ValueError("Empty rubric")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        blinded = {key: row[key] for key in ("id", "question", "answer")}
        prompt = rubric + "\n\nОцени следующий JSON-вход.\n\n" + json.dumps(blinded, ensure_ascii=False, indent=2) + "\n"
        (args.output_dir / f"{row['id']}.txt").write_text(prompt, encoding="utf-8")
    (args.output_dir / "rubric.md").write_text(rubric, encoding="utf-8")
    write_json(args.output_dir / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "answers_sha256": digest(args.answers), "rubric_sha256": digest(args.rubric),
        "ids": [row["id"] for row in rows],
        "prompt_sha256": {row["id"]: digest(args.output_dir / f"{row['id']}.txt") for row in rows},
        "protocol": "One fresh conversation per answer. No model identity or previous scores supplied.",
    })


def summarize(args):
    rows = load_answers(args.answers)
    expected_files = {f"{row['id']}.json" for row in rows}
    if {path.name for path in args.raw_dir.glob("*.json")} != expected_files:
        raise ValueError("Raw directory must contain exactly one JSON file per answer id")
    judgments = []
    for row in rows:
        value = json.loads((args.raw_dir / f"{row['id']}.json").read_text(encoding="utf-8"))
        judgments.append(validate_judgment(value, row))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *SCORES, *FLAGS])
        writer.writeheader()
        for value in judgments:
            writer.writerow({key: value[key] for key in writer.fieldnames})
    (args.output_dir / "judgments.jsonl").write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in judgments), encoding="utf-8")
    count = len(judgments)
    write_json(args.output_dir / "summary.json", {
        "n": count,
        "mean_scores": {key: sum(value[key] for value in judgments) / count for key in SCORES},
        "flags": {key: {"count": sum(value[key] for value in judgments), "n": count,
                         "rate": sum(value[key] for value in judgments) / count} for key in FLAGS},
        "answers_sha256": digest(args.answers),
        "raw_sha256": {row["id"]: digest(args.raw_dir / f"{row['id']}.json") for row in rows},
        "scope": args.scope,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Write one anonymized prompt per answer")
    prepare_parser.add_argument("--rubric", type=Path, required=True)
    prepare_parser.set_defaults(run=prepare)
    summarize_parser = commands.add_parser("summarize", help="Validate raw judge JSON and calculate metrics")
    summarize_parser.add_argument("--raw-dir", type=Path, required=True)
    summarize_parser.add_argument("--scope", default="Descriptive results; sample selection and human calibration are documented separately.")
    summarize_parser.set_defaults(run=summarize)
    for command in (prepare_parser, summarize_parser):
        command.add_argument("--answers", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
