"""Package the two saved adapters and frozen data for manual publication."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

RUN = 'results/sft_20261104T183742Z_9bc87f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    packages = []
    for suffix, directory in [('lr1e4', 'full'), ('lr2e5', 'full_lr2e5')]:
        run = args.root / RUN / directory
        manifest = json.loads((run / 'adapter_manifest.json').read_text())
        for name, expected in manifest.items():
            if digest(run / 'adapter' / name) != expected:
                raise ValueError(f'Adapter hash mismatch: {directory}/{name}')
        members = [(run / 'adapter' / name, 'adapter/' + name) for name in sorted(manifest)]
        members.extend((run / name, name) for name in ['adapter_manifest.json', 'config.json', 'environment.json'])
        packages.append((f'home-safety-lora-{suffix}.zip', members, 'negative_experiment_adapter'))
    data_dir = args.root / 'data/v1'
    data_manifest = json.loads((data_dir / 'manifest.json').read_text())
    for name, expected in data_manifest['sha256'].items():
        if digest(data_dir / name) != expected:
            raise ValueError(f'Data hash mismatch: {name}')
    members = [(path, 'data/v1/' + path.name) for path in sorted(data_dir.glob('*.json*'))]
    packages.append(('home-safety-data-v1.zip', members, 'frozen_data'))
    records = []
    for filename, members, kind in packages:
        destination = args.output_dir / filename
        with ZipFile(destination, 'x', compression=ZIP_DEFLATED) as archive:
            for source, name in members:
                entry = ZipInfo(name, date_time=(2026, 11, 5, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                archive.writestr(entry, source.read_bytes())
        with ZipFile(destination) as archive:
            assert archive.testzip() is None
            for source, name in members:
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest(source)
        records.append({'filename': filename, 'local_path': 'artifacts/release/' + filename,
                        'kind': kind, 'bytes': destination.stat().st_size, 'sha256': digest(destination),
                        'members': len(members), 'download_url': None, 'verified_access': False,
                        'access_verified_on': None})
    manifest = {'created_on': '2026-11-05', 'publication_status': 'prepared_locally',
                'repository': 'https://github.com/LeNcHic/home-safety-sft',
                'base_model': 'Qwen/Qwen2.5-3B-Instruct',
                'base_revision': 'aa8e72537993ba99e69dfaafa59ed015b17504d1', 'artifacts': records}
    (args.output_dir / 'release_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
