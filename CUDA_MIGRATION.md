# CUDA 版本遷移指南

> **✅ 已在 RTX 4060 上測試成功** (2025-01)

## CUDA 12.4 下載與安裝

### 下載 CUDA Toolkit 12.4
**官方下載頁面**：https://developer.nvidia.com/cuda-12-4-0-download-archive

**Windows 安裝建議**：
1. 選擇 **Windows** → **x86_64** → **10/11** → **exe (local)**
2. 下載大小：約 3.5 GB
3. 安裝後重啟電腦

### 系統需求
- Windows 10 或 11 (64-bit)
- RTX 4060 或更高等級 GPU
- 至少 6 GB 可用硬碟空間

## 移除 CUDA 12.1

### 方法 1：使用 Windows 控制台（推薦）

1. 開啟 **設定** → **應用程式** → **已安裝的應用程式**
2. 搜尋 "CUDA"，找到以下項目並依序卸載：
   - NVIDIA CUDA Visual Studio Integration 12.1
   - NVIDIA CUDA Runtime 12.1
   - NVIDIA CUDA Development 12.1
   - NVIDIA CUDA Documentation 12.1
   - NVIDIA CUDA Samples 12.1
   - NVIDIA nsight 系列工具 (12.1)

3. 重啟電腦

### 方法 2：使用 PowerShell 腳本

```powershell
# 列出所有 CUDA 12.1 相關程式
Get-WmiObject -Class Win32_Product | Where-Object {
    $_.Name -like "*CUDA*12.1*"
} | Select-Object Name, Version

# 卸載（需要 Admin 權限）
Get-WmiObject -Class Win32_Product | Where-Object {
    $_.Name -like "*CUDA*12.1*"
} | ForEach-Object {
    Write-Host "Uninstalling: $($_.Name)" -ForegroundColor Yellow
    $_.Uninstall()
}
```

**警告**：此方法較慢（每個程式約需 5-10 分鐘），建議使用方法 1。

### 方法 3：手動清理殘留檔案

卸載後，手動刪除以下目錄（如果存在）：

```powershell
# 檢查並刪除 CUDA 12.1 目錄
$cuda121Paths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "$env:ProgramData\NVIDIA Corporation\CUDA Samples\v12.1"
)

foreach ($path in $cuda121Paths) {
    if (Test-Path $path) {
        Write-Host "Found: $path" -ForegroundColor Yellow
        Write-Host "Delete manually if needed" -ForegroundColor Gray
    }
}
```

## 環境變數清理

### 檢查 CUDA 相關環境變數

```powershell
# 查看系統環境變數中的 CUDA 路徑
[System.Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
[System.Environment]::GetEnvironmentVariable('CUDA_PATH_V12_1', 'Machine')

# 查看 PATH 中的 CUDA 項目
$env:PATH -split ';' | Where-Object { $_ -like "*CUDA*" }
```

### 更新環境變數（安裝 CUDA 12.4 後自動完成）

CUDA 12.4 安裝程式會自動更新：
- `CUDA_PATH` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
- `CUDA_PATH_V12_4` → 新建
- `PATH` 變數會自動加入 CUDA 12.4 bin 和 libnvvp 路徑

## 驗證安裝

### 檢查 CUDA 版本

```powershell
# 查看已安裝的 CUDA 版本
nvcc --version

# 查看 GPU 驅動支援的最高 CUDA 版本
nvidia-smi
```

### 測試 PyTorch CUDA 支援

```powershell
# 啟動新環境
.\.venv-cuda124\Scripts\Activate.ps1

# 測試 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**預期輸出**：
```
PyTorch: 2.5.1+cu124
CUDA Available: True
CUDA Version: 12.4
```

## 常見問題

### Q: 可以同時安裝多個 CUDA 版本嗎？
A: 可以，但不建議。多版本共存可能導致環境變數衝突。

### Q: 移除 CUDA 12.1 會影響其他程式嗎？
A: 只影響依賴 CUDA 12.1 的程式。大部分程式使用驅動內建的 CUDA runtime，不受影響。

### Q: PyTorch cu124 相容 CUDA 12.1 嗎？
A: **不建議**。PyTorch cu124 需要 CUDA 12.4+ runtime libraries。

### Q: 如何確認清理完成？
A: 執行 `nvcc --version`，如果顯示 12.4 或找不到 12.1，表示清理成功。

## 遷移檢查清單

- [ ] 下載 CUDA Toolkit 12.4
- [ ] 卸載 CUDA 12.1 相關程式
- [ ] 安裝 CUDA Toolkit 12.4
- [ ] 重啟電腦
- [ ] 執行 `nvcc --version` 驗證
- [ ] 執行 `nvidia-smi` 確認驅動正常
- [ ] 刪除舊的虛擬環境 `.venv-cuda121`
- [ ] 執行 `.\scripts\setup-cuda124.ps1`
- [ ] 測試 PyTorch CUDA 支援

## 額外資源

- CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
- PyTorch Installation Guide: https://pytorch.org/get-started/locally/
- NVIDIA Driver Downloads: https://www.nvidia.com/Download/index.aspx
