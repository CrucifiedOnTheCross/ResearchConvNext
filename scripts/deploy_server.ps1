param(
  [string]$Server = "lab-bio@10.200.1.180",
  [string]$Repo = "https://github.com/CrucifiedOnTheCross/ResearchConvNext.git",
  [string]$RemoteDir = "ResearchConvNext",
  [ValidateSet("smoke","services","prepare","train-smoke","train")][string]$Mode = "smoke"
)
$ErrorActionPreference = "Stop"

function Invoke-RemotePs([string]$Code) {
  $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Code))
  & ssh -o BatchMode=yes $Server powershell -NoProfile -EncodedCommand $encoded
  if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}

$bootstrap = @"
`$ErrorActionPreference='Stop'
`$ProgressPreference='SilentlyContinue'
`$git="`$HOME\MinGit\cmd\git.exe"
if (-not (Test-Path `$git)) {
  `$zip="`$HOME\MinGit.zip"
  curl.exe -L --fail --output `$zip "https://github.com/git-for-windows/git/releases/download/v2.54.0.windows.1/MinGit-2.54.0-64-bit.zip"
  Expand-Archive -Path `$zip -DestinationPath "`$HOME\MinGit" -Force
}
if (-not (Test-Path "$RemoteDir\.git")) {
  New-Item -ItemType Directory -Force "$RemoteDir" | Out-Null
  & `$git -C "$RemoteDir" init
}
`$ErrorActionPreference='SilentlyContinue'; & `$git -C "$RemoteDir" remote get-url origin *>`$null; `$hasRemote=(`$LASTEXITCODE -eq 0); `$ErrorActionPreference='Stop'
if (-not `$hasRemote) {
  & `$git -C "$RemoteDir" remote add origin "$Repo"
} else {
  & `$git -C "$RemoteDir" remote set-url origin "$Repo"
}
& `$git -C "$RemoteDir" fetch origin main
`$ErrorActionPreference='SilentlyContinue'; & `$git -C "$RemoteDir" rev-parse --verify HEAD *>`$null; `$hasHead=(`$LASTEXITCODE -eq 0); `$ErrorActionPreference='Stop'
if (-not `$hasHead) {
  & `$git -C "$RemoteDir" checkout -B main origin/main --force
} else {
  & `$git -C "$RemoteDir" checkout main
  & `$git -C "$RemoteDir" pull --ff-only origin main
}
& `$git -C "$RemoteDir" checkout-index -a -f
docker compose -f "$RemoteDir\compose.yaml" build trainer
docker compose -f "$RemoteDir\compose.yaml" run --rm trainer scripts/smoke_gpu.py
"@
Invoke-RemotePs $bootstrap

if ($Mode -eq "services") {
  $services = @"
`$ErrorActionPreference='Stop'
docker compose -f "$RemoteDir\compose.yaml" up -d jupyter tensorboard
try {
  if (-not (Get-NetFirewallRule -DisplayName 'ResearchConvNext Web' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName 'ResearchConvNext Web' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8888,6006 | Out-Null
  }
} catch { Write-Warning 'Firewall ports were not opened (run once from elevated PowerShell if URLs are unreachable).' }
Write-Output "JUPYTER_URL=http://10.200.1.180:8888/lab"
Write-Output "TENSORBOARD_URL=http://10.200.1.180:6006"
docker compose -f "$RemoteDir\compose.yaml" ps
"@
  Invoke-RemotePs $services
}

if ($Mode -in @("prepare","train-smoke","train")) {
  Invoke-RemotePs "docker compose -f '$RemoteDir\compose.yaml' run --rm trainer scripts/download_data.py --root data/ham10000; if (`$LASTEXITCODE) { exit `$LASTEXITCODE }; docker compose -f '$RemoteDir\compose.yaml' run --rm trainer scripts/prepare_data.py --root data/ham10000 --size 224 --cache"
}
if ($Mode -eq "train-smoke") {
  Invoke-RemotePs "docker compose -f '$RemoteDir\compose.yaml' run --rm trainer scripts/train.py --config configs/default.yaml --set training.epochs=1 --set model.compile=false --set output.visualizations=false"
}
if ($Mode -eq "train") {
  Invoke-RemotePs "docker compose -f '$RemoteDir\compose.yaml' run --rm trainer scripts/train.py --config configs/default.yaml"
}
Write-Host "SUCCESS: deploy mode '$Mode' completed on $Server" -ForegroundColor Green
