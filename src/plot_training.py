"""Render the recorded training loss, without rerunning training."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in (args.run_dir / 'training-log.jsonl').read_text().splitlines()]
    rows = [row for row in rows if 'loss' in row]
    with (args.run_dir / 'loss.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['step', 'epoch', 'loss', 'learning_rate'])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
    ax.plot([row['step'] for row in rows], [row['loss'] for row in rows], marker='o', linewidth=1.5)
    ax.set(xlabel='Optimizer step', ylabel='Training loss', title='QLoRA on 80 dialogues, 2 epochs')
    ax.grid(alpha=.25)
    fig.savefig(args.run_dir / 'loss.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
