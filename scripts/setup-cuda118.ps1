# Setup CUDA 11.8 environment for RAG_LLM_CVE
# This script creates .venv-cuda118 and installs PyTorch with CUDA 11.8 support and all dependencies
# Does NOT activate the environment - user must activate manually
# Requires: NVIDIA GPU with CUDA 11.8 installed

$ErrorActionPreference = "Stop"

# Get project root directory (parent of scripts folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAG_LLM_CVE Environment Setup (CUDA 11.8)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Project root: $ProjectRoot" -ForegroundColor Yellow
Write-Host ""
Write-Host "Prerequisites:" -ForegroundColor Yellow
Write-Host "  - NVIDIA GPU" -ForegroundColor White
Write-Host "  - CUDA Toolkit 11.8 installed" -ForegroundColor White
Write-Host "  - cuDNN compatible with CUDA 11.8" -ForegroundColor White
Write-Host ""

# Check if virtual environment exists
$VenvPath = Join-Path $ProjectRoot ".venv-cuda118"

if (Test-Path $VenvPath) {
    Write-Host "Virtual environment already exists at: $VenvPath" -ForegroundColor Green
    Write-Host "Skipping venv creation. Will install packages in existing environment." -ForegroundColor Yellow
    Write-Host ""
} else {
    # Create virtual environment
    Write-Host "Creating virtual environment: .venv-cuda118" -ForegroundColor Green
    python -m venv $VenvPath
    Write-Host ""
}

# Use the venv's python directly (no activation needed)
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
Write-Host ""
Write-Host "Installing PyTorch (CUDA 11.8)..." -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow
& $VenvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

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
& $VenvPython -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None; print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None; print(f'GPU name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"

Write-Host ""
$cudaAvailable = & $VenvPython -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaAvailable -eq "True") {
    Write-Host "CUDA is properly configured!" -ForegroundColor Green
} else {
    Write-Host "WARNING: CUDA is not available. Check your CUDA installation." -ForegroundColor Red
}

Write-Host ""
Write-Host "To use this environment, activate it with:" -ForegroundColor Green
Write-Host "  .\.venv-cuda118\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "To deactivate later, run: deactivate" -ForegroundColor Yellow
