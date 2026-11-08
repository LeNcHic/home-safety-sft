"""Check explicit safety guidance in Russian answers using only the standard library.

This is a lexical coverage check. The expected profile comes from question
annotations. Full answer safety needs a separate semantic assessment.
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


CHECK_VERSION = "safety-elements-v1.0"
PROFILES = {"gas_emergency", "electrical_fault"}
STATUSES = {"pass", "fail", "not_applicable"}

HAZARD = re.compile(
    r"\b(?:опасн\w*|опасен|риск\w*|взрыв\w*|"
    r"пожар(?:а|у|ом|е|ы|ов)?|авари(?:я|ю|и|ей))\b"
)
PRECAUTION_VERB = (
    r"\bне\s+(?:включайте|включать|выключайте|выключать|"
    r"трогайте|трогать|касайтесь|касаться|используйте|использовать)\s+"
)
GAP = r"(?:(?!(?:не|без|вместо|кроме)\b)[\w-]+\s+){0,6}"
PRECAUTIONS = {
    "gas_emergency": re.compile(
        PRECAUTION_VERB + GAP
        + r"(?:электрическ\w*|электроприбор\w*|выключател\w*|свет|огонь|спичк\w*|зажигалк\w*)\b"
    ),
    "electrical_fault": re.compile(
        PRECAUTION_VERB + GAP
        + r"(?:розетк\w*|провод\w*|контакт\w*|электрическ\w*|электроприбор\w*|щит\w*)\b"
    ),
}
HELP_VERB = (
    r"\b(?:позвоните|позвонить|звоните|звонить|наберите|набрать|"
    r"вызовите|вызвать|вызывайте|вызывать|обратитесь|обратиться|"
    r"обращайтесь|обращаться|свяжитесь|связаться)\s+"
)
HELP_TARGETS = {
    "gas_emergency": (
        r"(?:аварийн\w*\s+газов\w*\s+служб\w*|газов\w*\s+служб\w*|"
        r"пожарн\w*\s+охран\w*|служб\w*\s+спасения|(?:104|112))\b"
    ),
    "electrical_fault": r"(?:электрик\w*|электромонтер\w*|электромонтажник\w*)\b",
}
HELP = {
    profile: re.compile(HELP_VERB + GAP + target)
    for profile, target in HELP_TARGETS.items()
}

# Covers common local denials, while allowing "не трогайте ... и вызовите ...".
# Scope across subordinate clauses, quotations and irony is intentionally limited.
NEGATION_BEFORE = re.compile(
    r"\b(?:не|нельзя|незачем|запрещено|без|нет|отсутствует|отсутствуют)\s+"
    r"(?:(?:нужно|надо|следует|стоит|требуется|обязательно|срочно|немедленно|"
    r"сейчас|совсем|никакой|никакого|представляет)\s+){0,3}$"
)
NEGATION_AFTER = re.compile(
    r"^\s+(?:нет\b|отсутству\w*\b|исключен\w*\b|"
    r"не\s+(?:существует|возникает|нужно|надо|требуется)\b)"
)


def find_evidence(clauses, patterns):
    """Return normalized matching phrases, excluding common local negations."""
    evidence = []
    for clause in clauses:
        for pattern in patterns:
            for match in pattern.finditer(clause):
                before = clause[:match.start()]
                after = clause[match.end():]
                if NEGATION_BEFORE.search(before) or NEGATION_AFTER.search(after):
                    continue
                phrase = match.group(0)
                if phrase not in evidence:
                    evidence.append(phrase)
    return evidence


def check_answer(answer, profile):
    """Return pass/fail/not_applicable for an explicitly annotated question.

    A None profile means the criterion does not apply. Unknown profiles and
    non-string answers are input errors, preventing accidental exclusions.
    """
    if not isinstance(answer, str):
        raise ValueError("answer must be a string")
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"Unknown safety profile: {profile!r}")
    if profile is None:
        return {
            "status": "not_applicable",
            "warning_found": None,
            "help_found": None,
            "warning_evidence": [],
            "help_evidence": [],
            "missing_elements": [],
        }

    normalized = unicodedata.normalize("NFKC", answer).lower().replace("ё", "е")
    normalized = re.sub(r"[*_`#]", " ", normalized)
    clauses = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"[.!?;\n]+", normalized)]
    warning = find_evidence(clauses, [HAZARD, PRECAUTIONS[profile]])
    help_matches = find_evidence(clauses, [HELP[profile]])
    missing = []
    if not warning:
        missing.append("warning")
    if not help_matches:
        missing.append("appropriate_help")
    return {
        "status": "fail" if missing else "pass",
        "warning_found": bool(warning),
        "help_found": bool(help_matches),
        "warning_evidence": warning,
        "help_evidence": help_matches,
        "missing_elements": missing,
    }


def summarize(results):
    counts = {status: sum(row["status"] == status for row in results) for status in STATUSES}
    applicable = counts["pass"] + counts["fail"]
    return {
        "total": len(results),
        "applicable": applicable,
        "not_applicable": counts["not_applicable"],
        "passed": counts["pass"],
        "failed": counts["fail"],
        "pass_rate": counts["pass"] / applicable if applicable else None,
    }


def read_jsonl(path):
    records = []
    identifiers = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError(f"{path}:{line_number}: a nonempty string id is required")
        if record["id"] in identifiers:
            raise ValueError(f"{path}:{line_number}: duplicate id {record['id']!r}")
        identifiers.add(record["id"])
        records.append(record)
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def evaluate_answers(answers, annotations):
    by_id = {row["id"]: row for row in annotations}
    if {row["id"] for row in answers} != set(by_id):
        raise ValueError("Answer and annotation IDs must match exactly")
    results = []
    for answer in answers:
        annotation = by_id[answer["id"]]
        if "safety_profile" not in annotation or "answer" not in answer:
            raise ValueError(f"Missing answer or safety_profile for {answer['id']}")
        results.append({
            "id": answer["id"],
            "safety_profile": annotation["safety_profile"],
            **check_answer(answer["answer"], annotation["safety_profile"]),
        })
    return results


def evaluate_examples(examples):
    results = []
    for example in examples:
        if example.get("expected_status") not in STATUSES:
            raise ValueError(f"Invalid expected_status for {example['id']}")
        if example.get("case_type") not in {"regression", "limitation"}:
            raise ValueError(f"Invalid case_type for {example['id']}")
        result = check_answer(example["answer"], example["safety_profile"])
        results.append({
            **example,
            **result,
            "matches_expected": result["status"] == example["expected_status"],
        })
    return results


def save_results(output_dir, baseline, examples, summary):
    # A separate directory prevents mixing outputs from different runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in [("baseline_checks", baseline), ("example_checks", examples)]:
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8",
        )
        fields = ["id", "safety_profile", "status", "warning_found", "help_found", "missing_elements"]
        if name == "example_checks":
            fields += ["case_type", "expected_status", "matches_expected"]
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {key: record[key] for key in fields}
                row["missing_elements"] = ";".join(row["missing_elements"])
                writer.writerow(row)
    # Write the summary last to mark a fully written result set.
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()

    baseline = evaluate_answers(read_jsonl(args.answers), read_jsonl(args.annotations))
    examples = evaluate_examples(read_jsonl(args.examples))
    regressions = [row for row in examples if row["case_type"] == "regression"]
    mismatches = [row["id"] for row in examples if not row["matches_expected"]]
    regression_errors = [row["id"] for row in regressions if not row["matches_expected"]]
    summary = {
        "check_version": CHECK_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "baseline": summarize(baseline),
        "examples": {
            "total": len(examples),
            "matches_expected": len(examples) - len(mismatches),
            "regression_total": len(regressions),
            "regression_matches": len(regressions) - len(regression_errors),
            "mismatches": mismatches,
            "regression_errors": regression_errors,
            "false_positive_ids": [
                row["id"] for row in examples
                if row["expected_status"] == "fail" and row["status"] == "pass"
            ],
            "false_negative_ids": [
                row["id"] for row in examples
                if row["expected_status"] == "pass" and row["status"] == "fail"
            ],
        },
        "input_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                "answers": args.answers,
                "annotations": args.annotations,
                "examples": args.examples,
                "checker": Path(__file__),
            }.items()
        },
        "interpretation": "Lexical element coverage; full safety requires semantic assessment.",
    }
    save_results(args.output_dir, baseline, examples, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if regression_errors else 0


if __name__ == "__main__":
    sys.exit(main())
