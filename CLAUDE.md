# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Policy

**CRITICAL: Never commit or push code changes to the git repository unless explicitly requested by the user.**

## Documentation Update Policy

**IMPORTANT: When the user requests to "update documentation" or "update files", unless explicitly specified otherwise, this refers to updating the markdown files in the project root directory:**
- `CLAUDE.md` - Project documentation and user guide (this file)
- `ARCHITECTURE.md` - System architecture and technical details
- `PROGRESS.md` - Completed changes and upcoming features

**Do NOT update external documentation or create new files unless explicitly requested.**

## Project Overview

RAG-based CVE validation system for security operations centers (SOCs). Validates CVE usage in threat intelligence reports using Meta's Llama 3.2-1B-Instruct model combined with retrieval-augmented generation to reduce LLM hallucinations.

## Prerequisites

- Python 3.10+
- **Hardware**: Choose based on your needs
  - **CPU-only**: 8-16GB RAM (slow but works)
  - **GTX 1660 Ti (6GB)**: 10-20x faster, needs CUDA 11.8
  - **RTX 4060+ (12GB)**: 20-40x faster, needs CUDA 12.4 (recommended)
- Hugging Face account with Llama model access approval
- Run `huggingface-cli login` before first use
- External CVE JSON feeds: `../cvelist/<year>` (v4 schema) and `../cvelistV5/cves/<year>` (v5 schema)

## Installation

### Automated Setup (Recommended)

Use PowerShell scripts in `scripts/` directory. These scripts create isolated virtual environments and handle PyTorch installation automatically.

**CPU-only** (no GPU required):
```powershell
.\scripts\setup-cpu.ps1
```
Creates `.venv-cpu` with PyTorch CPU version (~200MB download)

**CUDA 11.8** (for GTX 1660 Ti):
```powershell
# First install CUDA Toolkit 11.8:
# https://developer.nvidia.com/cuda-11-8-0-download-archive

.\scripts\setup-cuda118.ps1
```
Creates `.venv-cuda118` with PyTorch + CUDA 11.8 (~2.5GB download)

**CUDA 12.4** (for RTX 4060+, recommended):
```powershell
# First install CUDA Toolkit 12.4:
# https://developer.nvidia.com/cuda-12-4-0-download-archive

.\scripts\setup-cuda124.ps1
```
Creates `.venv-cuda124` with PyTorch + CUDA 12.4 (~2.5GB download)

**Activating environments**:
```powershell
# CPU
.\.venv-cpu\Scripts\Activate.ps1

# CUDA 11.8
.\.venv-cuda118\Scripts\Activate.ps1

# CUDA 12.4
.\.venv-cuda124\Scripts\Activate.ps1
```

**Switching between environments**: Just deactivate current environment and activate another.

### Manual Installation

If you prefer manual setup:

```bash
# Step 1: Install PyTorch (choose one)
# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install dependencies
pip install -r requirements.txt
```

## Quick Start Workflow

### Step 1: Setup Environment
Choose based on your hardware (see Installation section above):
```powershell
# Example for RTX 4060
.\scripts\setup-cuda124.ps1
.\.venv-cuda124\Scripts\Activate.ps1
```

### Step 2: Login to Hugging Face
```bash
huggingface-cli login
# Enter your token when prompted
```

### Step 3: Build Embedding Database (one-time setup)
```bash
# Default (recommended: fast speed, pkl format)
python localEmbedding.py
# When prompted, enter PDF path and output file name (without extension)
# Outputs: CVEEmbeddings.pkl (~2-5 minutes on GPU, 10-20 minutes on CPU)

# Speed levels
python localEmbedding.py --speed=normal    # Baseline quality (float32, slower)
python localEmbedding.py --speed=fast      # Recommended (default): float16, 1.5-2x faster
python localEmbedding.py --speed=fastest   # Maximum speed: float16 + larger chunks, 2-3x faster

# Output formats (affects file size and read speed)
python localEmbedding.py --extension=csv      # Text format, largest (~95 MB)
python localEmbedding.py --extension=pkl      # Pickle (default), balanced (~33 MB)
python localEmbedding.py --extension=parquet  # Optimal: smallest (~24 MB), fastest read, requires pyarrow
python localEmbedding.py --extension=chroma   # Vector database, no server required, best for large datasets

# Combined example
python localEmbedding.py --speed=fastest --extension=parquet
```

**Format comparison**:
- `csv`: Largest file (~95 MB), slowest read, maximum compatibility
- `pkl` (default): Balanced size/speed (~33 MB), Python-native
- `parquet`: Smallest file (~24 MB), fastest read, requires `pip install pyarrow`
- `chroma`: Vector database (directory-based), optimized queries, no server, best for large datasets

### Step 4: (Optional) Extract CVE Reference Text
```bash
# Extract current year from V5 (default, fastest, shows progress bar)
python extractCVE.py

# Extract specific year from V5
python extractCVE.py --year=2024

# Extract from V4 only
python extractCVE.py --schema=v4

# Extract from both V5 and V4 (with deduplication, slower)
python extractCVE.py --schema=all

# Extract all years from both schemas
python extractCVE.py --year=all --schema=all

# Enable detailed logging (for debugging)
python extractCVE.py --verbose

# Output file naming:
# - Single year: CVEDescription2024.txt
# - Multiple years: CVEDescription2023-2024.txt
```

**Verbose mode**:
- Default: Shows progress bar with file count
- `--verbose` or `-v`: Shows detailed file-by-file logging

**Schema options**:
- `--schema=v5` (default): V5 only, fastest
- `--schema=v4`: V4 only
- `--schema=all`: Both V5 and V4 with deduplication (V5 takes priority)

### Step 5: Analyze Threat Intelligence Reports
```bash
# Default (uses fast speed, full mode, v5→v4 fallback schema, pkl embeddings)
python theRag.py

# Demo mode (faster, limited to 10 pages)
python theRag.py --mode=demo

# Speed levels (optimize LLM performance)
python theRag.py --speed=normal   # Baseline, maximum precision (FP32)
python theRag.py --speed=fast     # Recommended (default): FP16 + optimized cache
python theRag.py --speed=fastest  # Maximum speed: +low temperature +SDPA

# Embedding file format (must match localEmbedding.py output)
python theRag.py --extension=csv      # Read CVEEmbeddings.csv
python theRag.py --extension=pkl      # Read CVEEmbeddings.pkl (default)
python theRag.py --extension=parquet  # Read CVEEmbeddings.parquet (fastest file-based)
python theRag.py --extension=chroma   # Read CVEEmbeddings/ (vector database, optimized)

# Use specific CVE schema
python theRag.py --schema=v5      # V5 only
python theRag.py --schema=v4      # V4 only
python theRag.py --schema=all     # V5→V4 fallback (default)

# Combine all parameters
python theRag.py --mode=full --speed=fastest --extension=parquet --schema=v5
```

**Speed options** (see ARCHITECTURE.md for details):
- `--speed=normal`: Baseline with chunk-aware filtering, FP32 precision
  - GPU (GTX 1660 Ti): Option 2 ~4-5 min (7x faster than original)
  - CPU: Option 2 ~4-5 min (7x faster than original)
- `--speed=fast` (default): +FP16 +reduced cache clearing
  - GPU (GTX 1660 Ti): Option 2 ~3 min (11x faster than original) ⭐
  - CPU: Option 2 ~4-5 min (similar to normal, FP16/cache ineffective on CPU)
- `--speed=fastest`: +lower temperature +SDPA attention
  - GPU (GTX 1660 Ti): Option 2 ~2 min (16x faster than original)
  - CPU: Option 2 ~4 min (5-10% faster than normal, minor gains from temperature/SDPA)

**Note**: All speed levels provide 7x speedup from chunk-aware CVE filtering. GPU benefits more from fast/fastest optimizations (FP16, cache management). CPU users can use any speed level, but normal or fast recommended as fastest provides minimal additional benefit.

**Schema options**:
- `--schema=v5`: V5 only (fastest, requires V5 feeds)
- `--schema=v4`: V4 only (requires V4 feeds)
- `--schema=all` (default): V5→V4 fallback (backward compatible)

**When prompted**: Enter the PDF filename you want to analyze

**What happens during loading** (can take 1-8 minutes depending on hardware):
1. **Phase 1**: Load Llama model (~2.5GB, 10-60 sec)
2. **Phase 2**: Extract text from PDF (10 sec)
3. **Phase 3**: Extract CVEs with regex (instant)
4. **Phase 4**: Lookup CVE metadata from JSON files (1-10 sec)
5. **Phase 5**: For missing CVEs, ask LLM for recommendations (longest step, 10 sec - 4 min)

**After loading completes**, interactive menu appears:
```
1. Summarize report (LLM generates executive summary)
2. Validate CVE usage (compares report vs. official descriptions)
3. Custom Q&A (ask questions about the report)
4. Exit
```

## Mode Comparison

| Feature | Demo Mode (`--mode=demo`) | Full Mode (default) |
|---------|---------------------------|---------------------|
| **Use Case** | Quick testing, limited hardware | Production analysis |
| **PDF Pages** | First 10 pages only | All pages |
| **Text Processing** | Truncate to 2000 chars | Intelligent chunking (1500 tokens) |
| **Embedding Rows** | First 1000 rows | All rows |
| **Retrieval Results** | Top-3 | Top-5 |
| **Token Generation** | 64-256 tokens | 150-700 tokens |
| **Memory Usage** | 4-6 GB RAM / 3.5-4 GB VRAM | 6-8 GB RAM / 4-5 GB VRAM |
| **Optimizations** | FP16 + CUDNN deterministic | Auto-precision |
| **Missing CVE Context** | First 500 chars | 1500-char window around mention |

## Architecture Details

### CVE Extraction (Optimized)
- **Direct regex pattern matching** on PDF text: `CVE-\d{4}-\d{4,7}`
- No LLM involvement in extraction (fast and accurate)
- Legacy LLM-based extraction preserved in code for reference

### CVE Lookup Strategy
1. **JSON File Lookup with configurable schema**:
   - `--schema=v5`: Only `../cvelistV5/cves/<year>/<prefix>/CVE-<year>-<id>.json` (v5 schema)
   - `--schema=v4`: Only `../cvelist/<year>/<prefix>/CVE-<year>-<id>.json` (v4 schema)
   - `--schema=all` (default): Try v5 first, fallback to v4 (backward compatible)
   - Auto-detects schema and extracts: `cveId`, `vendor`, `product`, `description`
2. **Fallback for Missing CVEs**:
   - Demo mode: Uses first 500 chars of report
   - Full mode: Extracts context window around CVE (up to 2000 chars)
   - Asks Llama to paraphrase usage in 2 sentences
   - Retrieves similar CVEs using `asking_llama_for_advice()`

### Embedding & Retrieval (Optimized)
- **Global SentenceTransformer**: Loaded once at startup (not per-call)
- Model: `all-mpnet-base-v2` (768 dimensions)
- Loads `CVEEmbeddings.csv` with precomputed embeddings
- Returns top-3 (demo) or top-5 (full) chunks via dot product scoring
- Feeds retrieved context + query to Llama for CVE recommendation

### Memory Management & Performance
- **All inference calls use `torch.no_grad()`** for efficiency
- **CUDA cache cleared** after each chunk in full mode
- **Periodic GC** during PDF processing in demo mode
- **Context length limits** prevent OOM errors
- **Auto device detection**: Seamlessly switches between CUDA/GPU based on environment
- **Offline operation**: Works without network after first model download

### Typical Performance (10-page PDF, 8 CVEs, 3 missing)
| Hardware | Phase 5 (Missing CVEs) | Option 1 (Summary) | Total First Run |
|----------|------------------------|---------------------|-----------------|
| **CPU** (i5/i7) | 4 min | 3 min | ~7-8 min |
| **GTX 1660 Ti** | 24 sec | 18 sec | ~1 min (**90% faster**) |
| **RTX 4060** | 15 sec | 12 sec | ~40 sec (**95% faster**) |

## File Paths & Dependencies

### Expected Directory Structure
```
RAG_LLM_CVE/
├── theRag.py            # Main RAG application (dual-mode)
├── localEmbedding.py    # Generate embedding database
├── extractCVE.py        # Optional: export CVE descriptions
├── cleanupLlamaCache.py # Clean Llama model cache
├── requirements.txt     # Python dependencies (includes pyarrow, chromadb)
├── CVEEmbeddings.pkl    # Generated by localEmbedding.py (default format)
│                        # (or .csv / .parquet / CVEEmbeddings/ dir for chroma)
├── samples/             # Sample reports for testing (gitignored)
├── scripts/             # Environment setup scripts
│   ├── setup-cpu.ps1    # CPU-only environment
│   ├── setup-cuda118.ps1 # CUDA 11.8 environment
│   └── setup-cuda124.ps1 # CUDA 12.4 environment
├── .venv-cpu/           # Virtual environment (CPU, gitignored)
├── .venv-cuda118/       # Virtual environment (CUDA 11.8, gitignored)
└── .venv-cuda124/       # Virtual environment (CUDA 12.4, gitignored)
../
├── cvelist/
│   ├── 2023/            # v4 CVE JSON feeds (fallback)
│   ├── 2024/
│   └── <year>/          # Organized by year
└── cvelistV5/
    └── cves/
        ├── 2023/        # v5 CVE JSON feeds (primary)
        ├── 2024/
        └── <year>/      # Organized by year
```

### Key File Interactions
- `theRag.py` reads from:
  - User-provided PDF (via `fitz.open`)
  - `CVEEmbeddings.{extension}` (embedding database, based on `--extension` parameter)
    - `--extension=csv` → `CVEEmbeddings.csv` (file)
    - `--extension=pkl` (default) → `CVEEmbeddings.pkl` (file)
    - `--extension=parquet` → `CVEEmbeddings.parquet` (file)
    - `--extension=chroma` → `CVEEmbeddings/` (directory with vector database)
  - CVE JSON feeds (based on `--schema` parameter):
    - `--schema=v5`: Only `../cvelistV5/cves/<year>/<prefix>/CVE-*.json`
    - `--schema=v4`: Only `../cvelist/<year>/<prefix>/CVE-*.json`
    - `--schema=all` (default): Try v5 first, fallback to v4
- `localEmbedding.py` writes to:
  - `<user_input>.{extension}` (based on `--extension` parameter)
  - Default: `CVEEmbeddings.pkl` (file)
  - Chroma: `<user_input>/` (directory with vector database)
  - **Speed parameter** affects processing: normal (baseline), fast (default, 1.5-2x), fastest (2-3x)
- `extractCVE.py` reads from:
  - `../cvelistV5/cves/<year>/` (V5 schema, default)
  - `../cvelist/<year>/` (V4 schema, optional)
  - **Schema selection** via `--schema`: v5 (default, fastest), v4, or all (with deduplication)
  - **Year selection** via `--year`: current year (default), specific year, multiple years, or all
  - **Verbose mode** via `--verbose`: Shows detailed file-by-file logging (default: progress bar)
  - Only `--schema=all` enables deduplication (V5 priority over V4)
- `extractCVE.py` writes to:
  - `CVEDescription<year>.txt` (single year)
  - `CVEDescription<min>-<max>.txt` (multiple years)

## Model Configuration

### Llama 3.2-1B-Instruct Settings
- **Device**: Auto-detects CUDA/CPU (`device_map="auto"`)
- **Model initialization**:
  - Demo mode: `torch_dtype=torch.float16`, `low_cpu_mem_usage=True`
  - Full mode: `torch_dtype="auto"`
- **Generation params**:
  - `temperature=0.3`, `top_p=0.9`, `do_sample=True`
  - `pad_token_id=tokenizer.eos_token_id` (all calls)
- **Token limits**:
  - Demo mode: 64-256 tokens
  - Full mode: 150-700 tokens
- **Memory management**:
  - All inference wrapped in `torch.no_grad()`
  - Demo mode: CUDNN deterministic mode enabled
  - `cleanup_model()` on exit
- **Offline capable**: Models cached in `~/.cache/huggingface/hub/`
  - First run: ~2.9 GB download (Llama + mpnet)
  - Subsequent runs: fully offline

### Embedding Model
- `all-mpnet-base-v2` from SentenceTransformers
- 768-dimensional embeddings
- **Globally initialized once** at startup (performance optimization)
- Used for both corpus building (`localEmbedding.py`) and query encoding (`theRag.py`)

## Important Notes

- **Sync CVE Feeds**: Ensure `../cvelistV5/` and `../cvelist/` are up-to-date to minimize "Could Not Find" fallbacks
- **Embedding File Dependency**: `CVEEmbeddings.{extension}` must exist before running `theRag.py`
  - Generate with `localEmbedding.py --extension={extension}`
  - `--extension` parameter must match between localEmbedding.py and theRag.py
  - Default: `.pkl` (balanced size/speed)
- **Format-specific Requirements**:
  - Parquet: Requires `pyarrow` (installed in requirements.txt)
  - Chroma: Requires `chromadb` (installed in requirements.txt), creates directory instead of file
- **CVE Path Format**: Second set of CVE ID formatted as `Nxxx` (e.g., `4` → `0xxx`, `12345` → `12xxx`) via `format_second_set()`
- **Network Requirements**: First run downloads ~2.9 GB; subsequent runs fully offline
- **Memory Requirements**:
  - Demo mode: ~4 GB RAM minimum
  - Full mode: ~8 GB RAM recommended (CPU) or ~4 GB VRAM (GPU)
