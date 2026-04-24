[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$trainScript = Join-Path $repoRoot "scripts\\train_dqn.py"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing .venv. Run .\\scripts\\setup_dev.ps1 first."
}

& $venvPython $trainScript @TrainArgs
