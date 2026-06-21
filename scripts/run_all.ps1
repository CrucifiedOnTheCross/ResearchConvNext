param(
  [string]$Method = "supcon",
  [int]$Epochs = 50
)
$ErrorActionPreference = "Stop"
docker compose build trainer
if (-not (Test-Path "data/ham10000/HAM10000_metadata.csv")) {
  docker compose run --rm trainer scripts/download_data.py --root data/ham10000
}
docker compose run --rm trainer scripts/prepare_data.py --root data/ham10000 --size 224 --cache
$Config = "configs/$Method.yaml"
if (-not (Test-Path $Config)) { $Config = "configs/default.yaml" }
docker compose run --rm trainer scripts/train.py --config $Config --set "training.method=$Method" --set "training.epochs=$Epochs"
