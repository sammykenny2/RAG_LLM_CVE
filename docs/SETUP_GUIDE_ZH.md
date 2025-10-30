# RAG CVE 驗證系統 - 新手設置指南

本指南將帶領你從零開始，建立完整的開發環境並成功運行 Web 介面。

## 目錄
1. [前置準備](#前置準備)
2. [Clone 必要專案](#clone-必要專案)
3. [設置 Python 虛擬環境](#設置-python-虛擬環境)
4. [配置環境變數](#配置環境變數)
5. [建立 Embeddings 資料庫](#建立-embeddings-資料庫)
6. [啟動 Web 介面](#啟動-web-介面)
7. [常見問題](#常見問題)

---

## 前置準備

### 系統需求

**必要條件：**
- Python 3.10 或更高版本
- Git
- 至少 8GB RAM（CPU 模式）或 6GB VRAM（GPU 模式）
- 硬碟空間：至少 20GB（用於 CVE 資料和模型）

**硬體選擇：**
- **CPU 模式**：8-16GB RAM，處理速度較慢但可正常運作
- **GPU 模式（推薦）**：
  - GTX 1660 Ti (6GB VRAM)：速度提升 10-20 倍
  - RTX 4060+ (12GB VRAM)：速度提升 20-40 倍

### 安裝 Python

**Windows:**
1. 前往 [Python 官網](https://www.python.org/downloads/)
2. 下載 Python 3.10 或更高版本
3. 安裝時**勾選** "Add Python to PATH"

驗證安裝：
```powershell
python --version
# 應顯示：Python 3.10.x 或更高
```

### 安裝 Git

**Windows:**
1. 前往 [Git 官網](https://git-scm.com/download/win)
2. 下載並安裝 Git for Windows

驗證安裝：
```powershell
git --version
# 應顯示：git version x.x.x
```

---

## Clone 必要專案

### 1. 解壓縮主專案

你應該已經收到 `RAG_LLM_CVE-main.zip` 壓縮檔，將它解壓縮到合適的工作目錄。

**使用 PowerShell 解壓縮：**

```powershell
# 假設 ZIP 檔在 Downloads 資料夾
cd C:\Users\你的用戶名\Downloads

# 解壓縮到指定目錄
Expand-Archive -Path RAG_LLM_CVE-main.zip -DestinationPath C:\Users\你的用戶名\Source

# 進入專案目錄（注意解壓縮後的資料夾名稱會是 RAG_LLM_CVE-main）
cd C:\Users\你的用戶名\Source\RAG_LLM_CVE-main
```

**或手動解壓縮：**

1. 在檔案總管中找到 `RAG_LLM_CVE-main.zip`
2. 右鍵點擊 → **"解壓縮全部"**
3. 選擇解壓縮位置（例如 `C:\Users\你的用戶名\Source`）
4. 完成後，你會看到一個名為 `RAG_LLM_CVE-main` 的資料夾
5. 進入 `RAG_LLM_CVE-main` 資料夾

### 2. Clone CVE 資料專案

CVE 資料有兩個版本，建議都下載以獲得最完整的資料：

**CVE List V5（主要資料來源，最新格式）：**
```powershell
# 回到上層目錄
cd ..

# Clone CVE V5 資料
git clone https://github.com/CVEProject/cvelistV5.git

# 這會花費一些時間，因為資料庫很大（約 5-10 分鐘）
```

**CVE List V4（舊格式，作為備援）：**
```powershell
# Clone CVE V4 資料
git clone https://github.com/CVEProject/cvelist.git

# 同樣需要一些時間下載
```

完成後，你的目錄結構應該如下：
```
C:\Users\你的用戶名\Source\
├── RAG_LLM_CVE-main\    ← 主專案
├── cvelistV5\           ← CVE V5 資料
└── cvelist\             ← CVE V4 資料
```

### 3. 驗證 CVE 資料

確認資料已正確下載：

```powershell
# 檢查 V5 資料
ls ..\cvelistV5\cves\2025\

# 應該會看到多個目錄，如：0xxx, 1xxx, 2xxx, ...

# 檢查 V4 資料
ls ..\cvelist\2025\

# 同樣應該看到多個目錄
```

---

## 設置 Python 虛擬環境

### 使用自動化腳本（推薦）

專案提供了自動化設置腳本，會自動創建虛擬環境並安裝所有依賴。

**CPU 模式（無需 GPU）：**
```powershell
cd C:\Users\你的用戶名\Source\RAG_LLM_CVE-main
.\scripts\windows\Setup-CPU.ps1
```

這會：
1. 創建 `.venv-cpu` 虛擬環境
2. 安裝 PyTorch CPU 版本（約 200MB）
3. 安裝所有相依套件

**GPU 模式（如果你有 NVIDIA GPU）：**

首先確認你的 GPU 和 CUDA 版本：
```powershell
nvidia-smi
# 查看 "CUDA Version" 欄位
```

根據你的 CUDA 版本選擇：

**CUDA 11.8（適用於 GTX 1660 Ti 等）：**
```powershell
# 先安裝 CUDA Toolkit 11.8
# 下載：https://developer.nvidia.com/cuda-11-8-0-download-archive

# 然後執行設置腳本
.\scripts\windows\Setup-CUDA118.ps1
```

**CUDA 12.4（適用於 RTX 4060+ 等，推薦）：**
```powershell
# 先安裝 CUDA Toolkit 12.4
# 下載：https://developer.nvidia.com/cuda-12-4-0-download-archive

# 然後執行設置腳本
.\scripts\windows\Setup-CUDA124.ps1
```

### 啟動虛擬環境

**CPU 模式：**
```powershell
.\.venv-cpu\Scripts\Activate.ps1
```

**GPU 模式（CUDA 11.8）：**
```powershell
.\.venv-cuda118\Scripts\Activate.ps1
```

**GPU 模式（CUDA 12.4）：**
```powershell
.\.venv-cuda124\Scripts\Activate.ps1
```

成功啟動後，你會看到提示符前面出現環境名稱，例如：
```
(.venv-cuda118) PS C:\Users\你的用戶名\Source\RAG_LLM_CVE-main>
```

### 手動安裝（進階）

如果自動化腳本無法正常運作，可以手動安裝：

```powershell
# 1. 創建虛擬環境
python -m venv .venv

# 2. 啟動虛擬環境
.\.venv\Scripts\Activate.ps1

# 3. 安裝 PyTorch（選擇一個）
# CPU 版本：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或 CUDA 11.8 版本：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.4 版本：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安裝其他相依套件
pip install -r requirements.txt
```

---

## 配置環境變數

### 1. 登入 Hugging Face

首次使用需要登入 Hugging Face 以下載模型：

```powershell
# 確保虛擬環境已啟動
hf auth login
```

系統會提示你輸入 Access Token：
1. 前往 [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. 創建一個新的 Access Token（Read 權限即可）
3. 複製 Token 並貼上（注意：貼上時不會顯示任何字元）

### 2. 申請 Llama 模型訪問權限

使用的模型是 Meta 的 Llama 3.2-1B-Instruct，需要申請訪問權限：

1. 前往 [Llama 3.2-1B-Instruct 模型頁面](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
2. 點擊 "Request Access"
3. 填寫表單並提交（通常幾分鐘內會獲得批准）

### 3. 創建 .env 配置檔

複製範例配置檔並進行修改：

```powershell
# 複製範例檔案
copy .env.example .env

# 用文字編輯器開啟 .env
notepad .env
```

**必要修改：**

確認 CVE 資料路徑正確（根據你的實際路徑調整）：

```ini
# CVE 資料路徑
CVE_V5_PATH=../cvelistV5/cves
CVE_V4_PATH=../cvelist

# 如果你的 CVE 資料在其他位置，修改為絕對路徑，例如：
# CVE_V5_PATH=C:/Users/你的用戶名/Source/cvelistV5/cves
# CVE_V4_PATH=C:/Users/你的用戶名/Source/cvelist
```

**推薦設定：**

```ini
# 預設速度等級（fast 為推薦，在速度和準確度間取得平衡）
DEFAULT_SPEED=fast

# 預設處理模式（full 為完整功能）
DEFAULT_MODE=full

# 預設 CVE 資料來源（v5 為 CVE 5.0 格式，速度最快）
DEFAULT_SCHEMA=v5

# 預設 embedding 格式（chroma 為向量資料庫，效能最佳）
DEFAULT_EMBEDDING_FORMAT=chroma

# Embedding 精度（float16 可節省記憶體並加速）
EMBEDDING_PRECISION=float16

# Web UI 設定
GRADIO_SERVER_PORT=7860          # Pure Python 版本
GRADIO_SERVER_PORT_LANGCHAIN=7861  # LangChain 版本
GRADIO_SHARE=False               # 設為 True 可產生公開連結
```

儲存並關閉檔案。

---

## 建立 Embeddings 資料庫

Embeddings 是將文字轉換為向量的資料庫，讓系統能快速檢索相關的 CVE 資訊。

### 使用案例：建立 SNMP 相關的 CVE 資料庫

假設你想建立一個專注於 SNMP（簡單網路管理協定）漏洞的知識庫。

### 執行 build_embeddings.py

```powershell
# 確保虛擬環境已啟動
python .\cli\build_embeddings.py
```

### 互動式設定流程

**步驟 1：選擇資料來源**
```
============================================================
Build Embeddings - Create New Knowledge Base
============================================================

What would you like to build embeddings from?
1. PDF file
2. CVE data by year (from JSON feeds)
3. CVE text/JSONL file (from extract_cve.py)

Enter your choice (1-3): 2
```
**選擇 `2`**（從 CVE JSON 資料建立）

**步驟 2：選擇年份範圍**
```
Enter year(s) for CVE data (e.g., 2025, 2023-2025, or 'all'): all
```
**輸入 `all`**（處理所有年份的資料，約 1999-2025）

**步驟 3：選擇 CVE 資料格式**
```
Select CVE schema:
1. v5 only (fastest)
2. v4 only
3. v5 with v4 fallback (default)
Enter your choice (1-3, default=3): 1
```
**選擇 `1`**（使用 V5 格式，速度最快）

系統會顯示即將處理的年份：
```
Processing all available years: [1999, 2000, 2001, ..., 2024, 2025]
```

**步驟 4：設定關鍵字過濾（重要！）**
```
Filter CVEs by keyword (leave empty for all): SNMP
```
**輸入 `SNMP`**（只包含與 SNMP 相關的 CVE）

> **💡 提示：** 如果不輸入任何關鍵字，系統會處理**所有** CVE（約 20 萬筆以上），這會花費數小時並佔用大量硬碟空間。使用關鍵字過濾可以大幅縮短時間。

### 處理過程

系統會開始處理資料，你會看到：

```
============================================================
Configuration:
============================================================
Source:      cve_json
Years:       all (1999-2025)
Schema:      v5
Filter:      SNMP
Format:      chroma
Precision:   float16
============================================================

Scanning V5 directory: ..\cvelistV5\cves\1999
Processing v5 (1999): 100%|████████████| 8/8 [00:01<00:00]
Year 1999: Extracted 12 CVE descriptions

Scanning V5 directory: ..\cvelistV5\cves\2000
Processing v5 (2000): 100%|████████████| 43/43 [00:03<00:00]
Year 2000: Extracted 45 CVE descriptions

...（持續處理）...

============================================================
[OK] Total: Extracted 847 CVE descriptions (filtered by 'SNMP')
============================================================

Initializing embedding model on cuda...
Generating embeddings (batch_size=64)...
Batches: 100%|████████████| 14/14 [00:08<00:00,  1.65it/s]

Storing embeddings in Chroma database...
  └─ Stored 847 embeddings in Chroma database

[OK] Generated: .\embeddings\CVEEmbeddings.chroma
[INFO] To use this with validate_report.py, run:
   python validate_report.py --extension=chroma
```

**時間估計：**
- CPU 模式：約 10-15 分鐘（SNMP 過濾後約 800-1000 筆 CVE）
- GPU 模式：約 2-5 分鐘

### 驗證資料庫

確認 embeddings 已成功建立：

```powershell
ls .\embeddings\

# 應該會看到：
# CVEEmbeddings.chroma\  (目錄)
```

進一步檢查內容：
```powershell
ls .\embeddings\CVEEmbeddings.chroma\

# 應該會看到 Chroma 資料庫的檔案，如：
# chroma.sqlite3
# *.bin 等檔案
```

---

## 啟動 Web 介面

系統提供兩個 Web 介面版本，功能相同但底層實作不同：

### 版本 1：Pure Python（推薦新手）

```powershell
# 確保虛擬環境已啟動並在專案根目錄
python .\web\web_ui.py
```

成功啟動後，你會看到：

```
Loading Llama model: meta-llama/Llama-3.2-1B-Instruct
  └─ FP16 precision enabled
[OK] Model loaded on device: cuda:0

Loading embedding model: sentence-transformers/all-mpnet-base-v2
  └─ Device: cuda
[OK] Embedding model loaded on cuda

Connecting to Chroma database...
[OK] Loaded existing collection: cve_embeddings

[OK] SessionManager initialized (session_id=abc12345...)
[OK] RAG system ready (speed=fast)

Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

### 版本 2：LangChain（進階使用者）

```powershell
python .\web\web_ui_langchain.py
```

這個版本使用 LangChain 框架，功能與 Pure Python 版本相同，但可以更容易擴展到其他 LangChain 生態系統的工具。

啟動後的訊息類似，但使用 port 7861：
```
Running on local URL:  http://127.0.0.1:7861
```

### 使用 Web 介面

1. **開啟瀏覽器**，前往：
   - Pure Python 版本：http://127.0.0.1:7860
   - LangChain 版本：http://127.0.0.1:7861

2. **介面說明：**

   ```
   ┌──────────────────────────────────────────────────────────┐
   │  左欄：對話區                │  右欄：設定與知識庫         │
   │                             │                            │
   │  💬 Conversation            │  ⚙️ Analysis Settings      │
   │  [對話歷史顯示區]            │  Speed: [fast ▼]           │
   │                             │  Mode:  [full ▼]           │
   │  Your message:              │  Schema: [all ▼]           │
   │  [_______________] [Send]   │                            │
   │                             │  📚 Knowledge Base         │
   │  📎 Upload Report            │  [+ Add Files]             │
   │     for Validation          │                            │
   │                             │  Current Sources:          │
   │                             │  • CVE_1999-2025_v5        │
   │                             │    (847 chunks)            │
   │                             │  [🔄 Refresh]              │
   └──────────────────────────────────────────────────────────┘
   ```

3. **測試查詢：**

   在對話框中輸入測試問題：
   ```
   What are the most critical SNMP vulnerabilities?
   ```

   系統會：
   - 搜尋 SNMP 相關的 CVE
   - 使用 Llama 模型生成專業的回答
   - 列出相關的 CVE 編號

4. **上傳報告進行驗證：**

   - 點擊 "➕ Add File" 上傳一份威脅情報報告（PDF 格式）
   - 詢問：`summarize this report`（生成摘要）
   - 或詢問：`validate CVE usage`（驗證 CVE 使用是否正確）

### 停止 Web 介面

在終端機按 `Ctrl+C` 即可停止服務。

---

## 常見問題

### Q1: 執行 PowerShell 腳本時出現「無法載入，因為這個系統上已停用指令碼執行」

**問題：** Windows 預設禁止執行未簽署的腳本。

**完整錯誤訊息範例：**
```
.\scripts\windows\Setup-CPU.ps1: File C:\...\Setup-CPU.ps1 cannot be loaded.
The file is not digitally signed. You cannot run this script on the current system.
```

**解決方法（依推薦順序）：**

**方法 1：為當前 PowerShell 進程設置執行策略（推薦，已驗證）**

在**非管理員** PowerShell 中執行：
```powershell
# 設置當前進程的執行策略（僅影響當前視窗）
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 然後執行腳本
.\scripts\windows\Setup-CPU.ps1
```

這個方法：
- ✅ 不需要管理員權限
- ✅ 只影響當前 PowerShell 視窗
- ✅ 關閉視窗後設定自動失效，安全性高

**方法 2：解除檔案的「網際網路下載」封鎖標記**

```powershell
# 解除腳本的封鎖
Unblock-File -Path .\scripts\windows\Setup-CPU.ps1

# 然後正常執行
.\scripts\windows\Setup-CPU.ps1
```

這個方法：
- ✅ 只需要執行一次
- ✅ 腳本會被視為本地建立的檔案

**方法 3：使用 Bypass 參數直接執行**

```powershell
# 一次性使用 Bypass 策略執行
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Setup-CPU.ps1
```

這個方法：
- ✅ 不修改任何設定
- ✅ 每次執行都需要使用完整命令

**方法 4：永久設置用戶層級執行策略**

以**管理員身分**開啟 PowerShell：
```powershell
# 設置當前用戶的執行策略
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 選擇 "Y" 確認
```

然後在**非管理員** PowerShell 中執行腳本：
```powershell
.\scripts\windows\Setup-CPU.ps1
```

如果仍然失敗，可能需要再執行方法 2 解除檔案封鎖。

**為什麼會出現這個問題？**

當你從 ZIP 解壓縮或從網路下載檔案時，Windows 會標記這些檔案為「從網際網路下載」。PowerShell 的 `RemoteSigned` 策略會要求這些檔案必須有數位簽章才能執行，但我們的腳本沒有簽章。

**快速驗證：哪個方法適合我？**

```powershell
# 檢查當前執行策略設定
Get-ExecutionPolicy -List

# 如果所有 Scope 都是 Undefined 或 Restricted，建議使用方法 1 或方法 2
```

### Q2: Hugging Face 下載模型失敗或速度很慢

**問題：** 網路連線問題或 Hugging Face 服務不穩定。

**解決方法：**

1. **使用鏡像站（中國大陸用戶）：**
   ```powershell
   # 設定環境變數
   $env:HF_ENDPOINT = "https://hf-mirror.com"
   ```

2. **手動下載模型：**
   - 前往 [Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) 和 [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
   - 下載所有檔案到本地
   - 修改 `.env` 中的 `LLAMA_MODEL_NAME` 和 `EMBEDDING_MODEL_NAME` 為本地路徑

### Q3: GPU 記憶體不足（CUDA out of memory）

**問題：** GPU VRAM 不足以載入模型。

**解決方法：**

1. **降低 batch size：**
   編輯 `.env`：
   ```ini
   EMBEDDING_BATCH_SIZE=32  # 預設為 64，改為 32 或 16
   ```

2. **使用 CPU 模式：**
   ```powershell
   # 切換到 CPU 虛擬環境
   .\.venv-cpu\Scripts\Activate.ps1
   ```

### Q4: 建立 embeddings 時處理速度很慢

**問題：** CPU 模式處理速度較慢。

**解決方法：**

1. **使用關鍵字過濾：** 只處理需要的 CVE（如 `SNMP`），而不是全部

2. **縮小年份範圍：**
   ```
   Enter year(s): 2020-2025
   ```
   只處理近 5 年的資料

3. **考慮使用 GPU：** 如果有可用的 NVIDIA GPU，速度可提升 10-40 倍

### Q5: Web 介面無法連線

**問題：** 瀏覽器無法開啟 http://127.0.0.1:7860

**檢查步驟：**

1. **確認服務已啟動：**
   終端機應顯示 `Running on local URL: http://127.0.0.1:7860`

2. **檢查 port 是否被佔用：**
   ```powershell
   netstat -ano | findstr :7860
   ```
   如果被佔用，可以修改 `.env` 中的 `GRADIO_SERVER_PORT`

3. **防火牆設定：**
   確保 Windows 防火牆允許 Python 存取網路

### Q6: CVE 資料路徑錯誤

**問題：** 系統找不到 CVE 資料。

**錯誤訊息：**
```
[ERROR] CVE V5 path not found: ../cvelistV5/cves
```

**解決方法：**

1. **確認 CVE 專案已 clone：**
   ```powershell
   ls ..\cvelistV5
   ls ..\cvelist
   ```

2. **使用絕對路徑：**
   編輯 `.env`，改用絕對路徑：
   ```ini
   CVE_V5_PATH=C:/Users/你的用戶名/Source/cvelistV5/cves
   CVE_V4_PATH=C:/Users/你的用戶名/Source/cvelist
   ```
   注意：Windows 路徑中的反斜線 `\` 要改成斜線 `/`

### Q7: 想要更新 CVE 資料

**問題：** CVE 資料庫會持續更新，如何獲取最新資料？

**解決方法：**

```powershell
# 更新 V5 資料
cd ..\cvelistV5
git pull
cd ..\RAG_LLM_CVE-main

# 更新 V4 資料
cd ..\cvelist
git pull
cd ..\RAG_LLM_CVE-main

# 重新建立 embeddings
python .\cli\build_embeddings.py
```

### Q8: 想要同時查詢 SNMP 和其他主題的 CVE

**問題：** 已經建立了 SNMP 的 embeddings，想要新增其他主題（如 SSH）的資料。

**解決方法：**

使用 `add_to_embeddings.py` 增量新增：

```powershell
python .\cli\add_to_embeddings.py

# 選擇選項 2 (CVE data by year)
# 輸入年份：all
# 輸入關鍵字：SSH
```

這會將 SSH 相關的 CVE 新增到現有的資料庫中，而不會刪除 SNMP 的資料。

---

## 下一步

現在你已經成功建立了開發環境並啟動了 Web 介面，可以：

1. **了解系統架構：** 閱讀 [docs/ARCHITECTURE.md](./ARCHITECTURE.md) 了解技術細節
2. **查看更新紀錄：** 閱讀 [docs/PROGRESS.md](./PROGRESS.md) 了解最新功能
3. **自訂配置：** 調整 `.env` 檔案以最佳化系統效能

---

**文件版本：** 1.0
**最後更新：** 2025-10-30
**適用版本：** RAG_LLM_CVE v1.0+
