# Setup CPU-only environment for RAG_LLM_CVE
# This script creates venv-cpu and installs PyTorch CPU version and all dependencies
# Does NOT activate the environment - user must activate manually

$ErrorActionPreference = "Stop"

# Get project root directory (parent of scripts folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAG_LLM_CVE Environment Setup (CPU)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project root: $ProjectRoot" -ForegroundColor Yellow
Write-Host ""

# Check if virtual environment exists
$VenvPath = Join-Path $ProjectRoot "venv-cpu"

if (Test-Path $VenvPath) {
    Write-Host "Virtual environment already exists at: $VenvPath" -ForegroundColor Green
    Write-Host "Skipping venv creation. Will install packages in existing environment." -ForegroundColor Yellow
    Write-Host ""
} else {
    # Create virtual environment
    Write-Host "Creating virtual environment: venv-cpu" -ForegroundColor Green
    python -m venv $VenvPath
    Write-Host ""
}

# Use the venv's python directly (no activation needed)
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip

# Install PyTorch CPU version
Write-Host ""
Write-Host "Installing PyTorch (CPU-only)..." -ForegroundColor Green
& $VenvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
Write-Host ""
Write-Host "Installing project dependencies..." -ForegroundColor Green
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
& $VenvPip install -r $RequirementsPath

# Verify installation
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verifying PyTorch installation..." -ForegroundColor Yellow
& $VenvPython -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

Write-Host ""
Write-Host "To use this environment, activate it with:" -ForegroundColor Green
Write-Host "  .\venv-cpu\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "To deactivate later, run: deactivate" -ForegroundColor Yellow
