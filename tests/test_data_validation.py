"""Regressions for preventing contamination and broken SFT targets."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from validate_data import check_data


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'v1'
        shutil.copytree(ROOT / 'data/v1', self.data)
        self.smoke = Path(self.temp.name) / 'smoke.jsonl'
        self.smoke.write_text(json.dumps({'id': 'smoke_fixture', 'question': 'Уникальный smoke вопрос для теста'}) + '\n')

    def change_row(self, filename, update):
        path = self.data / filename
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        update(rows[0])
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))

    def errors(self):
        return check_data(self.data, self.smoke)['errors']

    def test_cross_split_family_is_rejected(self):
        self.change_row('validation_questions.jsonl', lambda row: row.update(family_id=json.loads((self.data / 'train.jsonl').read_text().splitlines()[0])['family_id']))
        self.assertTrue(any('Family crosses splits' in item for item in self.errors()))

    def test_smoke_reuse_is_rejected(self):
        self.change_row('test_questions.jsonl', lambda row: row.update(question='Уникальный smoke вопрос для теста!'))
        self.assertTrue(any('Smoke question reused' in item for item in self.errors()))

    def test_missing_target_is_rejected(self):
        self.change_row('train.jsonl', lambda row: row['messages'][2].update(content=''))
        self.assertTrue(any('Missing message content' in item for item in self.errors()))

    def test_unknown_source_is_rejected(self):
        self.change_row('train.jsonl', lambda row: row.update(source_ids=['nonexistent']))
        self.assertTrue(any('Invalid source references' in item for item in self.errors()))


if __name__ == '__main__':
    unittest.main()
