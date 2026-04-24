[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PytestArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\\Scripts\\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing .venv. Run .\\scripts\\setup_dev.ps1 first."
}

if (-not $PytestArgs -or $PytestArgs.Count -eq 0) {
    $PytestArgs = @("tests")
}

& $venvPython -m pytest @PytestArgs
