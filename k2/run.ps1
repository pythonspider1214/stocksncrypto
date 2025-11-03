# PowerShell runner: creates venv, installs deps, loads env, runs bot
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $HERE

$VENV = ".venv"
if (-not (Test-Path $VENV)) {
  Write-Host "[+] Creating venv"
  python -m venv $VENV
}

& "$VENV\Scripts\Activate.ps1"
python -m pip install -U pip | Out-Null
pip install -r requirements.txt | Out-Null

# Load env.ps1 if present, else rely on .env auto-load by python-dotenv
if (Test-Path "env.ps1") {
  . .\env.ps1
}

python .\live_rsi_bot.py @args
