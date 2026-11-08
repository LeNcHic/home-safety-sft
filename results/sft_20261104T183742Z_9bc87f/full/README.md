# QLoRA с learning rate 1e-4

Обучение выполнено на Tesla T4: 80 диалогов, 2 эпохи и 20 шагов оптимизатора.

- config.json - настройки обучения и хеши входных данных.
- environment.json и pip-freeze.txt - окружение запуска.
- training-log.jsonl и trainer_state.json - шаги и loss.
- training_metrics.json - время обучения и память GPU.
- loss.csv и loss.png - кривая обучения.
- adapter_manifest.json - контрольные суммы файлов адаптера.
- weight_check.json и local_verification.json - результаты проверки сохранённых весов.
- reload_probe.json - диагностический ответ после загрузки адаптера.

Качество оценивается по отдельным ответам validation. Веса и токенизатор
исключены из Git и упакованы в отдельный ZIP для Release. [Реестр архивов](../../../artifacts/release_manifest.json).
