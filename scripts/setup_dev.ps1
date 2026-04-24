[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [switch]$Cuda,
    [switch]$InstallRipgrep
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\\python.exe"
$venvPip = Join-Path $venvDir "Scripts\\pip.exe"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Command {
    param(
        [string]$CommandName,
        [string]$ErrorMessage
    )

    $command = Get-Command $CommandName -ErrorAction SilentlyContinue
    if (-not $command) {
        throw $ErrorMessage
    }
    return $command
}

Write-Step "Repository root"
Write-Host $repoRoot

Write-Step "Checking base Python"
Ensure-Command -CommandName $PythonCommand -ErrorMessage (
    "Could not find '$PythonCommand'. Install Python 3.12+ or pass -PythonCommand with a full path."
)
& $PythonCommand --version

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Step "Creating .venv"
    & $PythonCommand -m venv $venvDir
} else {
    Write-Step "Reusing existing .venv"
}

Write-Step "Upgrading pip"
& $venvPython -m pip install --upgrade pip

Write-Step "Installing runtime dependencies"
& $venvPip install -r (Join-Path $repoRoot "requirements.txt")

Write-Step "Installing development dependencies"
& $venvPip install -r (Join-Path $repoRoot "requirements-dev.txt")

if ($Cuda) {
    Write-Step "Installing CUDA-specific PyTorch wheel"
    & $venvPip install -r (Join-Path $repoRoot "requirements-cuda.txt")
}

Write-Step "Checking toolchain"
& $venvPython -c "import matplotlib, pytest, torch; print('matplotlib ok'); print('pytest ok'); print(f'torch {torch.__version__}'); print(f'cuda available: {torch.cuda.is_available()}')"

$rgCommand = Get-Command rg -ErrorAction SilentlyContinue
if ($rgCommand) {
    Write-Host "rg ok: $($rgCommand.Source)"
    & $rgCommand.Source --version
} elseif ($InstallRipgrep) {
    Write-Step "Installing ripgrep with winget"
    Ensure-Command -CommandName winget -ErrorMessage (
        "ripgrep is missing and winget is not available. Install ripgrep manually: winget install BurntSushi.ripgrep.MSVC"
    )
    & winget install --id BurntSushi.ripgrep.MSVC -e --accept-package-agreements --accept-source-agreements
    $rgCommand = Get-Command rg -ErrorAction SilentlyContinue
    if ($rgCommand) {
        Write-Host "rg ok: $($rgCommand.Source)"
        & $rgCommand.Source --version
    } else {
        Write-Warning "ripgrep installation completed, but rg is not visible in this session yet. Open a new terminal and run 'rg --version'."
    }
} else {
    Write-Warning "ripgrep (rg) is not installed or not on PATH. Recommended command: winget install BurntSushi.ripgrep.MSVC"
}

Write-Step "Done"
Write-Host "Use these commands next:"
Write-Host "  .\\.venv\\Scripts\\python.exe -m pytest"
Write-Host "  .\\scripts\\test.ps1"
Write-Host "  .\\scripts\\train_dqn.ps1 --map-name default --episodes 10"
