"""Plot paired validation generation diagnostics from summarize_validation.py."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--label', default='QLoRA')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(args.summary.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='constrained')
    for ax, key, title in zip(axes,
                             ['answers_with_sentence_repeated_at_least_3_times', 'token_limit_count'],
                             ['Same sentence at least 3 times', 'Reached 512-token limit']):
        values = [data[variant][key] for variant in ('base', 'sft')]
        bars = ax.bar(['Base', args.label], values, color=['#377eb8', '#e69f00'])
        ax.bar_label(bars, labels=[f"{value}/{data[variant]['answers']}" for value, variant in zip(values, ('base', 'sft'))])
        ax.set(title=title, ylabel='Number of answers', ylim=(0, max(data['base']['answers'], data['sft']['answers'])))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(args.summary.parent / 'generation_comparison.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
