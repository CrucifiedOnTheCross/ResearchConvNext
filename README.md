# HAM10000 ConvNeXt-Base research pipeline

Воспроизводимый GPU-пайплайн для сравнения supervised contrastive и metric-learning методов на HAM10000. Он сохраняет точную конфигурацию, системную информацию, логи, TensorBoard, checkpoint, историю, logits/probabilities/embeddings, полные метрики и графики.

## Быстрый запуск в Docker

Требуются Docker Desktop/Engine, NVIDIA Container Toolkit и доступная из контейнера NVIDIA GPU.

```powershell
docker compose run --rm trainer scripts/download_data.py --root data/ham10000
docker compose run --rm trainer scripts/prepare_data.py --root data/ham10000 --size 224 --cache
docker compose run --rm trainer scripts/train.py --config configs/default.yaml
```

Или одной командой:

```powershell
.\scripts\run_all.ps1 -Method supcon -Epochs 50
```

## Deploy на сервер одной командой

Локально нужны только Windows OpenSSH Client и Git. На сервере нужны Docker Desktop с WSL2 backend и NVIDIA-драйвер; скрипт сам установит portable MinGit без прав администратора, выполнит pull/build и GPU smoke-test:

```powershell
.\scripts\deploy_server.ps1 -Mode smoke
```

Режимы: `smoke` — доставка + GPU-тест; `services` — JupyterLab и TensorBoard; `prepare` — дополнительно скачать/разбить/кэшировать HAM10000; `train-smoke` — весь pipeline на 1 эпоху; `train` — полное обучение. Любая ошибка возвращает ненулевой exit code, успешный конец печатает зелёный `SUCCESS`. Следующие deploy используют `git pull` и Docker cache, поэтому обычно занимают секунды.

### Веб-интерфейсы

```powershell
.\scripts\deploy_server.ps1 -Mode services
```

JupyterLab доступен без дополнительной авторизации внутри VPN. В нём видны все файлы проекта, `data/`, `runs/`, встроенный терминал и RTX 5080. Обучение можно запускать в терминале: `python scripts/train.py --config configs/default.yaml`. TensorBoard автоматически перечитывает `runs/*/tensorboard` каждые 5 секунд на `http://10.200.1.180:6006`.

Скачивание использует публичный Kaggle dataset `kmader/skin-cancer-mnist-ham10000`; положите API-токен в `%USERPROFILE%\.kaggle\kaggle.json`. Альтернатива без Kaggle API:

```powershell
docker compose run --rm trainer scripts/download_data.py --root data/ham10000 --archive data/HAM10000.zip
```

## Разбиение без утечки

`prepare_data.py` выполняет стратифицированное групповое разбиение. Единица группировки — `lesion_id`, поэтому снимки одного поражения никогда не попадают в разные train/validation/test. Скрипт аварийно завершается при любой утечке и пишет `splits.csv` и `split_summary.json`. Изображения один раз декодируются и кэшируются в JPEG 224×224.

## Методы

Classification baselines: `ce`, `weighted_ce`, `focal`, `logit_adjustment`, `balanced_softmax`. Representation/metric методы: `supcon`, `triplet`, `n_pairs`, `multi_similarity`, `circle`, `proxy_anchor`, `arcface`, `cosface`, `center`, `prototype`, `meta_prototype`, `paco_lite`, `bcl_lite`, `sbcl_lite`.

```powershell
docker compose run --rm trainer scripts/train.py --config configs/arcface.yaml
```

Linear classifier обучается от backbone features, а metric objectives — в отдельном projected embedding-пространстве. ArcFace/CosFace используют отдельную angular-margin classification head без параллельной обычной CE-head. `paco_lite`, `bcl_lite`, `sbcl_lite` и `meta_prototype` — явно обозначенные компактные адаптации, не точные реплики официальных реализаций статей. BCL-lite использует глобальные train class counts. Method-конфиги наследуют `default.yaml` через `_base_`, поэтому общие параметры задаются в одном месте.

## Производительность

- BF16 AMP по умолчанию (`torch.amp.autocast("cuda")`), `GradScaler` только для FP16.
- TF32 / `torch.set_float32_matmul_precision("high")`, cuDNN benchmark, channels-last.
- Автопроба максимального batch size с очисткой после OOM; gradient accumulation до effective batch.
- Pinned memory, persistent workers, prefetch и non-blocking H2D.
- `torch.compile`/Inductor с безопасным fallback, gradient checkpointing опционально.
- Docker: `ipc: host`, 16 GB shared memory и `expandable_segments` allocator.

Для стабильного сравнения сначала запустите с `model.compile=false`, затем включите compile и убедитесь, что validation-метрики совпадают в пределах ожидаемой стохастической вариации.

## Метрики и артефакты

Каждый каталог `runs/<timestamp>_<method>/` содержит:

- Accuracy, Balanced Accuracy, Macro-F1, MCC;
- precision/recall/F1/support и ROC-AUC/PR-AUC по каждому классу;
- binary malignant (`mel + bcc + akiec`) sensitivity, specificity, F1, MCC, ROC-AUC, PR-AUC;
- ECE, multiclass Brier и NLL до/после temperature scaling; температура подбирается только на validation и применяется к test;
- Silhouette, intra/inter-class distance и ratio;
- normalized confusion matrix, UMAP и t-SNE;
- `best.pt`, `config.yaml`, `system.json`, `train.log`, `history.json`, TensorBoard и `.npz` с embeddings/logits.

Дополнительно сохраняются checkpoints по Macro-F1, Balanced Accuracy, multiclass MCC и malignant MCC, `split_summary.json`, validation-пороги для malignant/melanoma и таблицы агрегации.

## Серии экспериментов и агрегация

```bash
python scripts/run_experiments.py --methods ce,focal,balanced_softmax,supcon,arcface --seeds 42,52,62
python scripts/collect_results.py --root runs
```

Runner автоматически выбирает `configs/<method>.yaml`. Результаты сохраняются в `runs/summary_per_run.csv` и `runs/summary_by_method.csv` с mean/std/count по seed.

Повторная оценка checkpoint без обучения:

```bash
python scripts/train.py --eval-only --checkpoint runs/<run>/best.pt
```

Temperature scaling и пороги Youden/MCC/F1/Balanced Accuracy подбираются только на validation и без переоценки применяются к test.

## Рекомендуемый порядок экспериментов

Сделайте smoke-test на 1 эпоху, затем отдельные seed для каждого метода. Не подбирайте гиперпараметры по test. Значения margin/temperature/metric weight подбирайте только по validation Macro-F1/Balanced Accuracy. Для профилирования используйте TensorBoard Profiler, `nvidia-smi dmon` или `nvtop`; высокое потребление VRAM само по себе не означает высокую загрузку GPU.
