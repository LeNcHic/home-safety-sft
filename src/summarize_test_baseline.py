"""Summarize a complete test baseline when adapter comparison is interrupted."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from checks import check_answer
from prepare_sft import read_jsonl
from summarize_validation import repetition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, default=Path('data/v1/test_annotations.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.run_dir / 'run.json').read_text())
    answers_path = args.run_dir / 'answers.jsonl'
    assert status['status'] == 'completed'
    assert hashlib.sha256(answers_path.read_bytes()).hexdigest() == status['answers_sha256']
    rows = read_jsonl(answers_path)
    annotations = {r['id']: r for r in read_jsonl(args.annotations)}
    assert len(rows) == status['expected_answers'] == len(annotations)
    assert {r['id'] for r in rows} == set(annotations)
    repetitions = [repetition(r['answer']) for r in rows]
    checks = [check_answer(r['answer'], annotations[r['id']]['safety_profile'])['status'] for r in rows]
    summary = {
        'answers': len(rows), 'mean_generated_tokens': statistics.mean(r['generated_tokens'] for r in rows),
        'eos_count': sum(r['ended_with_eos'] for r in rows),
        'token_limit_count': sum(r['hit_token_limit'] for r in rows),
        'answers_with_duplicate_sentence': sum(r['duplicate_sentences'] > 0 for r in repetitions),
        'answers_with_sentence_repeated_at_least_3_times': sum(r['max_sentence_occurrences'] >= 3 for r in repetitions),
        'mean_duplicate_sentence_ratio': statistics.mean(r['duplicate_sentence_ratio'] for r in repetitions),
        'safety_pass': checks.count('pass'),
        'safety_applicable': sum(value != 'not_applicable' for value in checks),
    }
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
