"""Aggregate completed validation and optional test comparisons without inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

RUN = 'results/sft_20261104T183742Z_9bc87f'
JUDGE_FIELDS = ('n', 'safety', 'usefulness', 'relevance', 'dangerous_recommendation', 'unjustified_refusal')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def collect_rows(root, run, folders, summaries, judgments, split):
    first, second = [read_json(run / path / 'summary.json') for path in summaries]
    assert first['base'] == second['base'], 'Baseline summaries differ'
    rows, inputs = [], {}
    for name, folder, automatic, judge, expected_hash in (
        ('base', folders[0], first['base'], judgments[0], first['method']['input_sha256']['base']),
        ('sft_lr1e4', folders[1], first['sft'], judgments[1], first['method']['input_sha256']['sft']),
        ('sft_lr2e5', folders[2], second['sft'], judgments[2], second['method']['input_sha256']['sft']),
    ):
        path = run / folder / 'answers.jsonl'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected_hash, f'Answers changed: {folder}'
        inputs[str(path.relative_to(root))] = digest
        row = {'split': split, 'variant': name, **automatic}
        row.update({'judge_' + key: judge.get(key) for key in JUDGE_FIELDS})
        rows.append(row)
    return rows, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--test-run', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run = root / RUN
    judge_first = read_json(run / 'validation_judge/summary_by_variant.json')
    judge_second = read_json(run / 'validation_judge_lr2e5/summary_by_variant.json')
    assert judge_first['base'] == judge_second['base'], 'Baseline judgments differ'
    rows, inputs = collect_rows(root, run,
        ['validation_base', 'validation_sft', 'validation_sft_lr2e5'],
        ['validation_summary', 'validation_summary_lr2e5'],
        [judge_first['base'], judge_first['sft'], judge_second['sft']], 'validation')
    if args.test_run:
        test_run = (root / args.test_run).resolve()
        assert read_json(test_run / 'status.json')['status'] == 'completed'
        extra, hashes = collect_rows(root, test_run, ['base', 'sft_lr1e4', 'sft_lr2e5'],
            ['summary_sft_lr1e4', 'summary_sft_lr2e5'], [{}, {}, {}], 'test')
        rows.extend(extra)
        inputs.update(hashes)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'date': '2026-11-05', 'scope': ['validation', 'test'] if args.test_run else ['validation'],
        'validation_decision': 'Both evaluated SFT configurations degraded validation performance.',
        'new_training_or_inference': False, 'test_evaluated': bool(args.test_run),
        'human_judge_calibration': False,
        'judge_sample': 'Nine purposively selected validation questions per variant; 27 unique judgments.',
        'limits': ['Validation informed the second learning rate.', 'One seed per training configuration.',
                   'Synthetic training data with partial user review.', 'Judge exact model is unknown.',
                   'Additional assistant review is not independent.', 'Test_056 and test_062 share a scenario family.'],
        'input_sha256': inputs, 'metrics': rows,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    splits = summary['scope']
    fig, axes = plt.subplots(1, len(splits), figsize=(8, 2.7), layout='constrained', squeeze=False)
    for ax, split in zip(axes[0], splits):
        subset = [row for row in rows if row['split'] == split]
        for offset, key, label, color in (
            (-0.18, 'answers_with_duplicate_sentence', 'Дословный повтор', '#315b8a'),
            (0.18, 'token_limit_count', 'Лимит 512 токенов', '#b76d35'),
        ):
            values = [row[key] / row['answers'] * 100 for row in subset]
            bars = ax.bar([x + offset for x in range(3)], values, width=0.34, label=label, color=color)
            ax.bar_label(bars, labels=[f'{v:g}%' for v in values], padding=3, fontsize=8)
        ax.set_xticks(range(3), ['База', 'SFT 1e-4', 'SFT 2e-5'], fontsize=8)
        ax.set_ylabel('Ответы, %')
        ax.set_title(f'{split}, n={subset[0]["answers"]}')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=7, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.15)
    fig.savefig(args.output_dir / 'generation_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'rows': len(rows), 'answers': sum(r['answers'] for r in rows), 'scope': splits}))


if __name__ == '__main__':
    main()
