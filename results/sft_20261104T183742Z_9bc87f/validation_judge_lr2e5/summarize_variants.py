"""Aggregate validated judgments using the separate blinded mapping."""
from collections import defaultdict
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
mapping = {row['id']: row for row in json.loads((root / 'mapping.json').read_text())}
judgments = [json.loads(line) for line in (root / 'metrics/judgments.jsonl').read_text().splitlines()]
groups = defaultdict(list)
paired = defaultdict(dict)
for row in judgments:
    meta = mapping[row['id']]
    groups[meta['variant']].append(row)
    paired[meta['question_id']][meta['variant']] = row
summary = {}
for variant, rows in sorted(groups.items()):
    summary[variant] = {
        'n': len(rows),
        **{key: sum(row[key] for row in rows) / len(rows) for key in ('safety', 'usefulness', 'relevance')},
        **{key: sum(row[key] for row in rows) for key in ('dangerous_recommendation', 'unjustified_refusal')},
    }
(root / 'summary_by_variant.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
with (root / 'paired_scores.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=['question_id', 'metric', 'base', 'sft', 'sft_minus_base'])
    writer.writeheader()
    for identifier, pair in sorted(paired.items()):
        for metric in ('safety', 'usefulness', 'relevance'):
            writer.writerow({'question_id': identifier, 'metric': metric, 'base': pair['base'][metric],
                             'sft': pair['sft'][metric], 'sft_minus_base': pair['sft'][metric]-pair['base'][metric]})
print(json.dumps(summary, indent=2))
