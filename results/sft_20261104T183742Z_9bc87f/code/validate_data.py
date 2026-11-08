"""Validate the corpus and optionally load its JSONL with datasets."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import tempfile

SYSTEM_PROMPT = "Ты помощник по вопросам дома и квартиры. Отвечай на русском языке, понятно и по существу."
SPLIT_FILES = {"train": "train.jsonl", "validation": "validation_questions.jsonl", "test": "test_questions.jsonl"}


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected nonempty JSON objects: {path}")
    return rows


def normalize(text):
    return " ".join(re.findall(r"[a-zа-яё0-9]+", text.lower().replace("ё", "е")))


def check_data(data_dir, smoke_path):
    errors = []
    warnings = []
    scenarios = read_rows(data_dir / "scenarios.jsonl")
    sources = read_rows(data_dir / "sources.jsonl")
    families = {row["family_id"]: row for row in scenarios}
    source_ids = {row["id"] for row in sources}
    if len(families) != len(scenarios):
        errors.append("Duplicate family_id in scenario registry")
    if len(source_ids) != len(sources):
        errors.append("Duplicate source id")
    for source in sources:
        if not source.get("url", "").startswith("https://") or not source.get("rights"):
            errors.append(f"Missing source URL or rights note: {source.get('id')}")
    ids = set()
    question_index = {}
    family_splits = {}
    questions = []
    answers = []
    statistics = {}
    for split, filename in SPLIT_FILES.items():
        rows = read_rows(data_dir / filename)
        for row in rows:
            identifier = row.get("id")
            family = row.get("family_id")
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                errors.append(f"Missing/duplicate example id: {identifier}")
            ids.add(identifier)
            if row.get("split") != split:
                errors.append(f"Wrong split field: {identifier}")
            spec = families.get(family)
            if not spec or spec.get("split") != split:
                errors.append(f"Unregistered, excluded or wrong-split family: {identifier}")
            elif (row.get("topic"), row.get("risk")) != (spec.get("topic"), spec.get("risk")):
                errors.append(f"Registry metadata mismatch: {identifier}")
            if family in family_splits and family_splits[family] != split:
                errors.append(f"Family crosses splits: {family}")
            family_splits[family] = split
            if row.get("risk") not in {"low", "medium", "high"}:
                errors.append(f"Invalid risk: {identifier}")
            if split == "train":
                messages = row.get("messages", [])
                if not isinstance(messages, list) or len(messages) != 3 or any(not isinstance(m, dict) for m in messages):
                    errors.append(f"Expected three messages: {identifier}")
                    continue
                if [m.get("role") for m in messages] != ["system", "user", "assistant"]:
                    errors.append(f"Wrong message roles: {identifier}")
                if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
                    errors.append(f"Missing message content: {identifier}")
                    continue
                if messages[0]["content"] != SYSTEM_PROMPT:
                    errors.append(f"System prompt mismatch: {identifier}")
                question, answer = messages[1]["content"], messages[2]["content"]
                answers.append((identifier, answer))
                references = row.get("source_ids")
                if not isinstance(references, list) or any(ref not in source_ids for ref in references):
                    errors.append(f"Invalid source references: {identifier}")
                if row.get("risk") == "high" and not references:
                    errors.append(f"High-risk answer has no source: {identifier}")
                if row.get("safety_profile") not in {None, "gas_emergency", "electrical_fault"}:
                    errors.append(f"Invalid safety profile: {identifier}")
                if not row.get("origin") or not row.get("review_status"):
                    errors.append(f"Missing provenance: {identifier}")
                if re.search(r"[\u3400-\u9fff]", answer):
                    errors.append(f"Unexpected CJK character: {identifier}")
                letters = re.findall(r"[^\W\d_]", answer, re.UNICODE)
                russian = re.findall(r"[А-Яа-яЁё]", answer)
                if not letters or len(russian) / len(letters) < 0.8:
                    errors.append(f"Russian-letter fraction below 0.8: {identifier}")
                sentences = [normalize(s) for s in re.split(r"[.!?]+", answer) if len(normalize(s)) > 25]
                if len(sentences) != len(set(sentences)):
                    errors.append(f"Repeated answer sentence: {identifier}")
            else:
                references = row.get("source_ids", [])
                if not isinstance(references, list) or any(ref not in source_ids for ref in references):
                    errors.append(f"Invalid evaluation source references: {identifier}")
                if row.get("risk") == "high" and not references:
                    warnings.append(f"High-risk evaluation criteria lack a source: {identifier}")
                if row.get("safety_profile") not in {None, "gas_emergency", "electrical_fault"}:
                    errors.append(f"Invalid evaluation safety profile: {identifier}")
                question = row.get("question")
                if "messages" in row or "answer" in row:
                    errors.append(f"Evaluation file contains a training target: {identifier}")
                for field in ("expected_elements", "forbidden_actions"):
                    values = row.get(field)
                    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                        errors.append(f"Invalid evaluation checklist: {identifier}/{field}")
                if not row.get("expected_elements"):
                    errors.append(f"Missing expected elements: {identifier}")
            if not isinstance(question, str) or not normalize(question):
                errors.append(f"Empty question: {identifier}")
                continue
            key = normalize(question)
            if key in question_index:
                errors.append(f"Duplicate normalized question: {question_index[key]}, {identifier}")
            question_index[key] = identifier
            questions.append({"id": identifier, "split": split, "text": question})
        statistics[split] = {
            "n": len(rows), "topics": dict(Counter(row.get("topic") for row in rows)),
            "risk": dict(Counter(row.get("risk") for row in rows)),
        }
    normalized_answers = [normalize(answer) for _, answer in answers]
    if len(set(normalized_answers)) != len(normalized_answers):
        errors.append("Duplicate normalized training answer")
    for row in read_rows(smoke_path):
        if normalize(row["question"]) in question_index:
            errors.append(f"Smoke question reused: {row['id']}")
        questions.append({"id": row["id"], "split": "smoke", "text": row["question"]})
    near_pairs = []
    for left, right in itertools.combinations(questions, 2):
        if left["split"] == right["split"]:
            continue
        texts = [normalize(row["text"]) for row in (left, right)]
        grams = [{text[i:i+3] for i in range(len(text)-2)} for text in texts]
        union = grams[0] | grams[1]
        score = len(grams[0] & grams[1]) / len(union) if union else 0.0
        if score >= 0.65:
            near_pairs.append({"left": left["id"], "right": right["id"], "character_trigram_jaccard": round(score, 4)})
    lengths = [len(answer) for _, answer in answers]
    return {
        "schema_version": "data-validation-v2", "errors": errors, "warnings": warnings, "statistics": statistics,
        "source_count": len(sources), "family_count": len(families),
        "answer_characters": {"min": min(lengths, default=0), "max": max(lengths, default=0)},
        "cross_split_near_duplicate_candidates": near_pairs,
        "near_duplicate_threshold": 0.65,
        "limitations": ["Lexical similarity cannot establish semantic independence.", "Safety content and evaluation checklists require human review.", "Character lengths are not tokenizer lengths."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pilot"))
    parser.add_argument("--smoke", type=Path, default=Path("results/baseline_20261104T112446Z_7e87c62c/questions.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-with-datasets", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new")
    result = check_data(args.data_dir, args.smoke)
    result["python"] = platform.python_version()
    result["datasets_load"] = {"performed": False}
    if args.load_with_datasets and not result["errors"]:
        import datasets
        import pyarrow
        loaded_rows = {}
        with tempfile.TemporaryDirectory(prefix="home-data-cache-") as cache:
            for split, filename in SPLIT_FILES.items():
                loaded = datasets.load_dataset("json", data_files=str(args.data_dir / filename), split="train", cache_dir=cache)
                originals = read_rows(args.data_dir / filename)
                if list(loaded) != originals:
                    raise ValueError(f"datasets changed record contents: {split}")
                loaded_rows[split] = len(loaded)
        result["datasets_load"] = {"performed": True, "datasets_version": datasets.__version__, "pyarrow_version": pyarrow.__version__, "rows": loaded_rows, "all_records_equal": True}
    files = [args.data_dir / name for name in [*SPLIT_FILES.values(), "sources.jsonl", "scenarios.jsonl"]] + [args.smoke, Path(__file__)]
    result["sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("errors", "warnings", "statistics", "datasets_load", "cross_split_near_duplicate_candidates")}, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
