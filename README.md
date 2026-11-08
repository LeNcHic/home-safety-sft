# home-safety-sft

Дообучение Qwen2.5-3B-Instruct для темы «Дом и квартира». Два эксперимента QLoRA
выполнены в Google Colab на Tesla T4. Оба ухудшили ответы относительно базовой модели.

| Показатель на 40 вопросах validation | База | SFT 1e-4 | SFT 2e-5 |
|---|---:|---:|---:|
| Ответы с дословными повторами | 3/40 | 23/40 | 14/40 |
| Достижение лимита 512 токенов | 1/40 | 16/40 | 7/40 |
| Безопасность ChatGPT, 9 вопросов | 3,89/5 | 3,67/5 | 3,11/5 |

Сравнение включает 120 ответов и 27 уникальных оценок судьи. Validation
использовалась для выбора второго learning rate. На test дополнительно изучены
80 ответов базы; показатели адаптеров на test в выводах не используются.

- [Отчёт Markdown](report.md) и [PDF, 7 страниц](report.pdf).
- [Итоговые метрики и график](results/final_evaluation/).
- [Обучение, ответы и оценки судьи](results/sft_20261104T183742Z_9bc87f/).
- [Данные и их описание](docs/dataset_v1.md).
- [Архивы весов и порядок публикации](docs/submission.md).

## Запуск в Google Colab на T4

1. Загрузите [01_baseline.ipynb](notebooks/01_baseline.ipynb) в Colab. Выберите
   T4 через меню «Среда выполнения → Сменить среду выполнения» и выполните ячейки.
2. Для обучения откройте [02_qlora_training.ipynb](notebooks/02_qlora_training.ipynb)
   в новой сессии с T4. В первой ячейке загрузите
   [home-sft-colab.zip](artifacts/home-sft-colab.zip).
3. Установите зависимости и выполните проверку данных, токенизации и короткое
   обучение на восьми примерах. При запросе Colab перезапустите среду и продолжите.
4. Установите `RUN_FULL_TRAINING = True` для первого обучения и
   `RUN_VALIDATION = True` для сравнения базы и адаптера на 40 вопросах.
5. Установите `RUN_LOWER_LR_EXPERIMENT = True` для второго обучения с 2e-5
   и его оценки на тех же вопросах. Скачайте результаты последней ячейкой.

Нужны интернет и T4. Модель скачивается с Hugging Face без API-ключа.
Использована ревизия `aa8e72537993ba99e69dfaafa59ed015b17504d1`, NF4,
double quantization, FP16, жадная генерация, seed 42 и максимум 512 новых токенов.

Окружение исходных запусков: Python 3.13.16, torch 2.11.0+cu130, CUDA 13.0,
transformers 4.57.1, PEFT 0.17.1. PyTorch предоставляется Colab, остальные пакеты
зафиксированы в [requirements-sft.txt](requirements-sft.txt). Фактические версии
сохранены в `environment.json` и `pip-freeze.txt` каждого запуска. Оба обучения
и три validation-прогона выполнены на T4; полный повтор окончательного комплекта
с нуля в новой сессии отдельно не проводился.

## Команды воспроизведения

Выполняйте команды из корня проекта. Для GPU-команд используйте терминал Colab
или ячейку с префиксом `!`. Выходные каталоги должны быть новыми.

```bash
python -m pip install -r requirements-sft.txt
python src/validate_data.py --data-dir data/v1 --output-dir results/repro_data --load-with-datasets
python src/prepare_sft.py --data-dir data/v1 --output-dir results/repro_tokenization
python src/train_qlora.py --mode sanity --output-dir results/repro_sanity
python src/train_qlora.py --mode train --learning-rate 1e-4 --output-dir results/repro_full
python src/train_qlora.py --mode train --learning-rate 2e-5 --output-dir results/repro_lr2e5
python src/evaluate_model.py --split validation --output-dir results/repro_base
python src/evaluate_model.py --split validation --adapter results/repro_full/adapter --output-dir results/repro_sft
python src/evaluate_model.py --split validation --adapter results/repro_lr2e5/adapter --output-dir results/repro_sft_lr2e5
python src/summarize_validation.py --base results/repro_base --sft results/repro_sft --annotations data/v1/validation_annotations.jsonl --output-dir results/repro_summary
python src/summarize_validation.py --base results/repro_base --sft results/repro_sft_lr2e5 --annotations data/v1/validation_annotations.jsonl --output-dir results/repro_summary_lr2e5
python src/plot_training.py --run-dir results/repro_full
python src/plot_validation.py --summary results/repro_summary/summary.json
```

Для пересчёта сохранённых результатов GPU не нужен. Unit-тесты и автоматическая
проверка используют стандартную библиотеку Python; итоговый график требует matplotlib.

```bash
python3 -m unittest discover -s tests -v
python3 src/checks.py --answers results/baseline_20261104T112446Z_7e87c62c/answers.jsonl --annotations data/safety_annotations.jsonl --examples data/safety_check_examples.jsonl --output-dir results/repro_checks
python3 src/finalize_evaluation.py --output-dir results/repro_final_evaluation
python3 src/summarize_test_baseline.py --run-dir results/test_20261105_final/base --output results/repro_test_baseline.json
```

`finalize_evaluation.py` сводит два сохранённых validation-сравнения и проверяет
контрольные суммы ответов. Для независимого пересчёта из текстов используйте
`summarize_validation.py` с папками `validation_base`, `validation_sft` и
`validation_sft_lr2e5` основного запуска. [Описание автоматической проверки](docs/safety_check.md).

## Оценка моделью-судьёй

[Рубрика home-judge-v1](docs/judge_rubric.md) задаёт оценки безопасности, полезности,
соответствия, опасных рекомендаций и необоснованных отказов. Использован ChatGPT
через веб-интерфейс. Точная версия неизвестна. Для каждого ответа открывался новый
чат; название модели скрыто. Подборка из девяти вопросов выбрана целевым образом.

В `validation_judge` и `validation_judge_lr2e5` сохранены протокол, обезличенные
ответы, соответствие ID, запросы, исходные оценки и таблицы. Базовые оценки во втором
сравнении использованы повторно. Команда для проверки первой подборки и её цитат:

```bash
python3 src/judge.py summarize --answers results/sft_20261104T183742Z_9bc87f/validation_judge/answers_blinded.jsonl --raw-dir results/sft_20261104T183742Z_9bc87f/validation_judge/raw --output-dir results/repro_judge --scope '9 вопросов validation, база и SFT 1e-4'
```

Для новых ответов используйте `judge.py prepare --answers <файл> --rubric
docs/judge_rubric.md --output-dir <новая папка>`, затем отправьте каждый запрос
в отдельный чат судьи и сохраните первый JSON-ответ. Входы и расчёт метрик
воспроизводимы; повторная веб-оценка может дать другие баллы. Независимой
человеческой калибровки нет.

## Данные и веса

[Набор v1](data/v1/) содержит 80 train, 40 validation и 80 test. JSONL совместим
с `datasets.load_dataset('json', data_files=...)`. Содержимое зафиксировано по SHA256
в manifest.json. Данные синтетические; происхождение и статус проверки сохранены.

Веса и копии токенизатора исключены из Git. Два архива адаптеров и архив данных
подготовлены для загрузки в GitHub Release. Размеры, SHA256 и поля фактических
ссылок находятся в [реестре артефактов](artifacts/release_manifest.json).
Ссылки пока не опубликованы. [Порядок загрузки и проверки доступа](docs/submission.md).

## Сборка архивов и отчёта

Если архив Colab нужно собрать заново, укажите новый выходной файл:

```bash
python3 src/build_colab_bundle.py --output /tmp/home-sft-colab.zip
```

Markdown служит исходником PDF. Для сборки нужны reportlab, DejaVu Sans и CJK-шрифт
в формате TTF для точной цитаты с двумя китайскими символами:

```bash
python -m pip install reportlab==4.4.9
python src/render_report.py --font-dir /path/to/dejavu --cjk-font /path/to/cjk.ttf
```

В Git включены код, финальные данные, результаты и отчёт. Черновики, неполные
прогоны, скриншоты интерфейса и служебные файлы остаются локально согласно `.gitignore`.
