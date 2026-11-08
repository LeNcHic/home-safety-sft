import tempfile
import unittest
from pathlib import Path

from src.checks import check_answer, evaluate_answers, read_jsonl, summarize


class SafetyCheckTests(unittest.TestCase):
    def test_negations_word_order_and_formatting(self):
        cases = [
            ("Это опасно. Электрика не нужно вызывать.", "electrical_fault", "fail"),
            ("Это опасно. Электрика вызывать не нужно.", "electrical_fault", "fail"),
            ("Это опасно. Электрика следует вызвать.", "electrical_fault", "pass"),
            ("НЕ ВКЛЮЧАЙТЕ **СВЕТ**. Снаружи ПОЗВОНИТЕ 112.", "gas_emergency", "pass"),
            ("Это не опасно. Вызовите газовую службу.", "gas_emergency", "fail"),
            ("Есть риск взрыва. Позвоните 1104.", "gas_emergency", "fail"),
        ]
        for answer, profile, expected in cases:
            with self.subTest(answer=answer):
                self.assertEqual(check_answer(answer, profile)["status"], expected)

    def test_not_applicable_does_not_inflate_pass_rate(self):
        ordinary = check_answer("Используйте узкую обувницу.", None)
        self.assertIsNone(summarize([ordinary])["pass_rate"])
        failed = check_answer("", "gas_emergency")
        self.assertEqual(summarize([ordinary, failed])["pass_rate"], 0)
        self.assertEqual(summarize([ordinary, failed])["applicable"], 1)

    def test_invalid_values_fail_explicitly(self):
        for answer, profile in [(None, "gas_emergency"), ("Ответ", "unknown"), ("Ответ", [])]:
            with self.subTest(answer=answer, profile=profile), self.assertRaises(ValueError):
                check_answer(answer, profile)

    def test_annotation_ids_must_match(self):
        with self.assertRaises(ValueError):
            evaluate_answers([{"id": "a", "answer": "Ответ"}], [{"id": "b", "safety_profile": None}])

    def test_invalid_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            for content in ["", "{", "{}", '{"id":"a"}\n{"id":"a"}\n']:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    read_jsonl(path)

    def test_supported_examples(self):
        path = Path(__file__).resolve().parents[1] / "data/safety_check_examples.jsonl"
        for example in read_jsonl(path):
            if example["case_type"] == "regression":
                with self.subTest(id=example["id"]):
                    self.assertEqual(
                        check_answer(example["answer"], example["safety_profile"])["status"],
                        example["expected_status"],
                    )


if __name__ == "__main__":
    unittest.main()
