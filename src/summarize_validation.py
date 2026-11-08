"""Summarize paired generation diagnostics and lexical safety checks.

Requires both complete runs on identical prompts. Repetition flags are technical
symptoms and require qualitative interpretation.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

from prepare_sft import read_jsonl
from checks import check_answer


def repetition(answer):
    sentences = [re.sub(r'\s+', ' ', sentence).strip().casefold()
                 for sentence in re.split(r'[.!?]+(?:\s+|$)|\n+', answer)]
    sentences = [sentence for sentence in sentences if len(re.findall(r'\w+', sentence)) >= 4]
    counts = Counter(sentences)
    repeated = sum(count - 1 for count in counts.values())
    return {'sentence_count': len(sentences), 'duplicate_sentences': repeated,
            'duplicate_sentence_ratio': repeated / len(sentences) if sentences else 0,
            'max_sentence_occurrences': max(counts.values(), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--sft', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = {variant: read_jsonl(directory / 'answers.jsonl') for variant, directory in
            [('base', args.base), ('sft', args.sft)]}
    for directory in (args.base, args.sft):
        status = json.loads((directory / 'run.json').read_text())
        if status['status'] != 'completed' or status['completed_answers'] != status['expected_answers']:
            raise ValueError('Both generation runs must be complete')
    annotations = {row['id']: row for row in read_jsonl(args.annotations)}
    identifiers = [[row['id'] for row in rows] for rows in runs.values()]
    if identifiers[0] != identifiers[1] or len(set(identifiers[0])) != len(identifiers[0]):
        raise ValueError('Different or duplicate question IDs')
    for base, sft in zip(runs['base'], runs['sft']):
        if base['messages'] != sft['messages'] or base['rendered_prompt'] != sft['rendered_prompt']:
            raise ValueError(f"Different model prompts: {base['id']}")
    configurations = [json.loads((directory / 'config.json').read_text()) for directory in (args.base, args.sft)]
    for key in ('model_id', 'model_revision', 'seed', 'system_prompt', 'split', 'questions_sha256', 'quantization', 'dtype', 'attention_implementation', 'generation'):
        if configurations[0][key] != configurations[1][key]:
            raise ValueError(f'Different generation configuration: {key}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics = []
    summary = {}
    for variant, answers in runs.items():
        for row in answers:
            check = check_answer(row['answer'], annotations[row['id']]['safety_profile'])
            metrics.append({key: row[key] for key in ('id', 'topic', 'risk', 'generated_tokens', 'generation_seconds', 'ended_with_eos', 'hit_token_limit')}
                           | {'variant': variant, **repetition(row['answer']), 'safety_status': check['status'],
                              'empty_answer': not row['answer'].strip()})
        subset = [row for row in metrics if row['variant'] == variant]
        applicable = [row for row in subset if row['safety_status'] != 'not_applicable']
        summary[variant] = {
            'answers': len(subset), 'mean_generated_tokens': statistics.mean(row['generated_tokens'] for row in subset),
            'eos_count': sum(row['ended_with_eos'] for row in subset),
            'token_limit_count': sum(row['hit_token_limit'] for row in subset),
            'empty_answer_count': sum(row['empty_answer'] for row in subset),
            'answers_with_duplicate_sentence': sum(row['duplicate_sentences'] > 0 for row in subset),
            'answers_with_sentence_repeated_at_least_3_times': sum(row['max_sentence_occurrences'] >= 3 for row in subset),
            'mean_duplicate_sentence_ratio': statistics.mean(row['duplicate_sentence_ratio'] for row in subset),
            'safety_applicable': len(applicable),
            'safety_pass': sum(row['safety_status'] == 'pass' for row in applicable),
        }
    summary['method'] = {
        'version': 'generation-diagnostics-v1',
        'repetition': 'Exact case-insensitive sentences of at least four words; split at .!? followed by whitespace/end, or line breaks. Duplicate ratio = occurrences beyond first / sentence count.',
        'interpretation': 'Repetition and length diagnose output degeneration. Lexical safety coverage is a limited proxy; substantive safety and usefulness require content review.',
        'input_sha256': {variant: hashlib.sha256((directory / 'answers.jsonl').read_bytes()).hexdigest()
                         for variant, directory in [('base', args.base), ('sft', args.sft)]},
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    with (args.output_dir / 'per_answer.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
