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

Команда напечатает URL с уникальным токеном для JupyterLab. В Jupyter доступны все файлы проекта, `data/`, `runs/`, встроенный терминал и RTX 5080. Обучение можно запускать в терминале: `python scripts/train.py --config configs/default.yaml`. TensorBoard автоматически перечитывает `runs/*/tensorboard` каждые 5 секунд на `http://10.200.1.180:6006`.

Скачивание использует публичный Kaggle dataset `kmader/skin-cancer-mnist-ham10000`; положите API-токен в `%USERPROFILE%\.kaggle\kaggle.json`. Альтернатива без Kaggle API:

```powershell
docker compose run --rm trainer scripts/download_data.py --root data/ham10000 --archive data/HAM10000.zip
```

## Разбиение без утечки

`prepare_data.py` выполняет стратифицированное групповое разбиение. Единица группировки — `lesion_id`, поэтому снимки одного поражения никогда не попадают в разные train/validation/test. Скрипт аварийно завершается при любой утечке и пишет `splits.csv` и `split_summary.json`. Изображения один раз декодируются и кэшируются в JPEG 224×224.

## Методы

Выбор: `supcon`, `triplet` (batch-hard mining), `n_pairs`, `multi_similarity`, `circle`, `proxy_anchor`, `arcface`, `cosface`, `center`, `paco`, `bcl`, `sbcl`, `prototype`, `meta_prototype`.

```powershell
docker compose run --rm trainer scripts/train.py --config configs/default.yaml --set training.method=arcface
```

SupCon, Triplet, Multi-Similarity, Circle, Proxy-Anchor, ArcFace/CosFace и Center реализованы как самостоятельные objectives. Реализации PaCo/BCL/SBCL и prototype-вариантов здесь являются компактными экспериментальными адаптациями общей схемы `CE + metric objective`; перед заявлением чисел в статье их следует валидировать против официальных репозиториев конкретных публикаций и провести ablation. Это намеренно отмечено, чтобы не выдавать удобный baseline за точную репликацию статьи.

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

## Рекомендуемый порядок экспериментов

Сделайте smoke-test на 1 эпоху, затем отдельные seed для каждого метода. Не подбирайте гиперпараметры по test. Значения margin/temperature/metric weight подбирайте только по validation Macro-F1/Balanced Accuracy. Для профилирования используйте TensorBoard Profiler, `nvidia-smi dmon` или `nvtop`; высокое потребление VRAM само по себе не означает высокую загрузку GPU.
