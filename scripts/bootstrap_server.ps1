param([string]$ProjectDir = "$HOME\ham10000-pipeline")
$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$HOME\.local\bin;$env:Path"
}
Set-Location $ProjectDir
uv python install 3.12
uv venv --python 3.12 .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory//2**20, 'MiB')"

