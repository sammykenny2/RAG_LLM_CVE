# Cleanup Llama Model Cache
# Remove all cached Llama 3.2 model artifacts from local Hugging Face cache
# Can be run from project root or scripts/windows directory

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Cleanup Model Cache" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Scanning Hugging Face cache directories..." -ForegroundColor Yellow
Write-Host ""

# Determine cache paths to check
$CachePaths = @()

# Check environment variables
if ($env:HF_HOME) {
    $CachePaths += Join-Path $env:HF_HOME "hub"
}
if ($env:HF_HUB_CACHE) {
    $CachePaths += $env:HF_HUB_CACHE
}

# Add default cache locations
$CachePaths += Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$CachePaths += Join-Path $env:LOCALAPPDATA "huggingface\hub"

# Remove duplicates
$CachePaths = $CachePaths | Select-Object -Unique

# Find and remove Llama 3.2 directories
$RemovedDirs = @()
$TotalSize = 0

foreach ($CachePath in $CachePaths) {
    if (-not (Test-Path $CachePath)) {
        continue
    }

    Write-Host "Checking: $CachePath" -ForegroundColor Gray

    # Find all Llama 3.2 Instruct model directories
    $LlamaDirs = Get-ChildItem -Path $CachePath -Directory -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -like "*Llama-3.2*" -and $_.Name -like "*Instruct*" }

    foreach ($Dir in $LlamaDirs) {
        try {
            # Calculate size before removal
            $Size = (Get-ChildItem -Path $Dir.FullName -Recurse -File -ErrorAction SilentlyContinue |
                     Measure-Object -Property Length -Sum).Sum
            $TotalSize += $Size

            # Remove directory
            Remove-Item -Path $Dir.FullName -Recurse -Force -ErrorAction Stop
            $RemovedDirs += [PSCustomObject]@{
                Path = $Dir.FullName
                Size = $Size
            }
            Write-Host "  ✓ Removed: $($Dir.Name)" -ForegroundColor Green
        }
        catch {
            Write-Host "  ✗ Failed to remove: $($Dir.Name) - $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# Display summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Cleanup Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($RemovedDirs.Count -gt 0) {
    Write-Host "Removed: $($RemovedDirs.Count) directory/directories" -ForegroundColor White
    $SizeMB = [math]::Round($TotalSize / 1MB, 2)
    $SizeGB = [math]::Round($TotalSize / 1GB, 2)

    if ($SizeGB -ge 1) {
        Write-Host "Space freed: $SizeGB GB" -ForegroundColor White
    } else {
        Write-Host "Space freed: $SizeMB MB" -ForegroundColor White
    }

    Write-Host ""
    Write-Host "Removed cached Llama 3.2 model directories:" -ForegroundColor Yellow
    foreach ($Item in $RemovedDirs) {
        $ItemSizeMB = [math]::Round($Item.Size / 1MB, 2)
        Write-Host "  - $($Item.Path) ($ItemSizeMB MB)" -ForegroundColor Gray
    }
} else {
    Write-Host "No cached Llama 3.2 model directories found." -ForegroundColor Yellow
}

Write-Host ""
