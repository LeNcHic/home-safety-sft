"""Create a deterministic ZIP of code and frozen data for Colab upload."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('artifacts/home-sft-colab.zip'))
    args = parser.parse_args()
    filenames = [
        'src/validate_data.py', 'src/prepare_sft.py', 'src/train_qlora.py',
        'src/evaluate_model.py', 'src/summarize_validation.py', 'src/plot_training.py', 'src/plot_validation.py',
        'src/checks.py', 'src/summarize_test_baseline.py', 'src/finalize_evaluation.py',
        'data/safety_check_examples.jsonl',
        'requirements-sft.txt',
        'results/baseline_20261104T112446Z_7e87c62c/questions.jsonl',
    ]
    filenames.extend(str(p.relative_to(args.root)) for p in sorted((args.root/'data/v1').glob('*.json*')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle, ZipFile(handle, 'w', compression=ZIP_DEFLATED) as archive:
        for filename in sorted(filenames):
            entry = ZipInfo(filename, date_time=(2026,11,4,0,0,0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, (args.root/filename).read_bytes())
    print(hashlib.sha256(args.output.read_bytes()).hexdigest(), args.output)


if __name__ == '__main__':
    main()
