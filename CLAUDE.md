# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Policy

**CRITICAL: Never commit or push code changes to the git repository unless explicitly requested by the user.**

## Documentation Update Policy

**IMPORTANT: When the user requests to "update documentation" or "update files", unless explicitly specified otherwise, this refers to updating the following markdown files:**
- `CLAUDE.md` - Project documentation and user guide (this file, in project root)
- `docs/ARCHITECTURE.md` - System architecture and technical details
- `docs/PROGRESS.md` - Completed changes and upcoming features

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
.\scripts\windows\Setup-CPU.ps1
```
Creates `.venv-cpu` with PyTorch CPU version (~200MB download)

**CUDA 11.8** (for GTX 1660 Ti):
```powershell
# First install CUDA Toolkit 11.8:
# https://developer.nvidia.com/cuda-11-8-0-download-archive

.\scripts\windows\Setup-CUDA118.ps1
```
Creates `.venv-cuda118` with PyTorch + CUDA 11.8 (~2.5GB download)

**CUDA 12.4** (for RTX 4060+, recommended):
```powershell
# First install CUDA Toolkit 12.4:
# https://developer.nvidia.com/cuda-12-4-0-download-archive

.\scripts\windows\Setup-CUDA124.ps1
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
.\scripts\windows\Setup-CUDA124.ps1
.\.venv-cuda124\Scripts\Activate.ps1
```

### Step 2: Login to Hugging Face
```bash
huggingface-cli login
# Enter your token when prompted
```

### Step 3: Build Embedding Database (one-time setup)
```bash
# Default (recommended: fast speed, pkl format, 30% overlap)
python cli/build_embeddings.py
# When prompted, enter PDF path and output file name (without extension)
# Outputs: cve_embeddings.pkl (~2-5 minutes on GPU, 10-20 minutes on CPU)

# Speed levels
python cli/build_embeddings.py --speed=normal    # Baseline quality (float32, slower)
python cli/build_embeddings.py --speed=fast      # Recommended (default): float16, 1.5-2x faster
python cli/build_embeddings.py --speed=fastest   # Maximum speed: float16 + larger chunks, 2-3x faster

# Output formats (affects file size and read speed)
python cli/build_embeddings.py --extension=csv      # Text format, largest (~95 MB)
python cli/build_embeddings.py --extension=pkl      # Pickle (default), balanced (~33 MB)
python cli/build_embeddings.py --extension=parquet  # Optimal: smallest (~24 MB), fastest read, requires pyarrow
python cli/build_embeddings.py --extension=chroma   # Vector database, no server required, best for large datasets

# Combined example
python cli/build_embeddings.py --speed=fastest --extension=parquet
```

**Format comparison**:
- `csv`: Largest file (~95 MB), slowest read, maximum compatibility
- `pkl` (default): Balanced size/speed (~33 MB), Python-native
- `parquet`: Smallest file (~24 MB), fastest read, requires `pip install pyarrow`
- `chroma`: Vector database (directory-based), optimized queries, no server, best for large datasets

**Chunk overlap** (configured via `.env`):
- Default: 30% overlap for PDF documents (improves retrieval accuracy by 13-17%)
- CVE data: 0% overlap (atomic descriptions don't need overlap)
- To adjust: Edit `CHUNK_OVERLAP_RATIO` in `.env` (0.0 = no overlap, 0.5 = 50% overlap)
- Trade-off: +40% storage/time cost for +13-17% accuracy improvement

### Step 4: (Optional) Extract CVE Reference Text
```bash
# Extract current year from V5 (default, fastest, shows progress bar)
python cli/extract_cve.py

# Extract specific year from V5
python cli/extract_cve.py --year=2024

# Extract from V4 only
python cli/extract_cve.py --schema=v4

# Extract from both V5 and V4 (with deduplication, slower)
python cli/extract_cve.py --schema=all

# Extract all years from both schemas
python cli/extract_cve.py --year=all --schema=all

# Enable detailed logging (for debugging)
python cli/extract_cve.py --verbose

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
python cli/validate_report.py

# Demo mode (faster, limited to 10 pages)
python cli/validate_report.py --mode=demo

# Speed levels (optimize LLM performance)
python cli/validate_report.py --speed=normal   # Baseline, maximum precision (FP32)
python cli/validate_report.py --speed=fast     # Recommended (default): FP16 + optimized cache
python cli/validate_report.py --speed=fastest  # Maximum speed: +low temperature +SDPA

# Embedding file format (must match localEmbedding.py output)
python cli/validate_report.py --extension=csv      # Read cve_embeddings.csv
python cli/validate_report.py --extension=pkl      # Read cve_embeddings.pkl (default)
python cli/validate_report.py --extension=parquet  # Read cve_embeddings.parquet (fastest file-based)
python cli/validate_report.py --extension=chroma   # Read cve_embeddings/ (vector database, optimized)

# Use specific CVE schema
python cli/validate_report.py --schema=v5      # V5 only
python cli/validate_report.py --schema=v4      # V4 only
python cli/validate_report.py --schema=all     # V5→V4 fallback (default)

# Combine all parameters
python cli/validate_report.py --mode=full --speed=fastest --extension=parquet --schema=v5
```

**Speed options** (see docs/ARCHITECTURE.md for details):
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

### Step 6: Web UI (Gradio Interface)

The project includes two web interfaces with Claude Projects-style layout:

#### Phase 1: Pure Python (web_ui.py)
```bash
# Launch Phase 1 web interface (port 7860)
python web/web_ui.py

# With custom configuration
python web/web_ui.py --server-name=0.0.0.0 --server-port=7860

# Share publicly (generates temporary URL)
python web/web_ui.py --share
```

**Features**:
- Chat interface with RAG-based question answering
- Upload PDF reports for validation (Summary/Validate/Q&A)
- Add PDFs to knowledge base permanently
- View and manage knowledge base sources
- Manual conversation history management (last 10 rounds)

#### Phase 2: LangChain (web_ui_langchain.py)
```bash
# Launch Phase 2 web interface (port 7861)
python web/web_ui_langchain.py

# Both versions can run simultaneously for comparison
```

**Features**:
- Same UI as Phase 1
- LangChain ConversationalRetrievalChain for RAG workflow
- Automatic memory management (ConversationBufferWindowMemory)
- Standardized LangChain abstractions

#### Web UI Layout
```
┌──────────────────────────────────────────────────────┐
│  Left Column (7/12)         │ Right Column (5/12)     │
│                             │                         │
│  💬 Chat History             │ ⚙️ Analysis Settings    │
│  [Conversation Display]     │  Speed:  [fast ▼]       │
│                             │  Mode:   [full ▼]       │
│  Your message:              │  Schema: [all ▼]        │
│  [________________] [Send]  │                         │
│                             │ ───────────────────     │
│  📎 Upload Report            │ 📚 Knowledge Base       │
│     for Validation          │  [+ Add Files]          │
│                             │                         │
│                             │  Current Sources:       │
│                             │  • CVEpdf2024.pdf       │
│                             │    (7,261 chunks)       │
│                             │  [🔄 Refresh]           │
└──────────────────────────────────────────────────────┘
```

#### Usage Examples

**Ask questions about CVEs**:
```
User: "What is CVE-2024-1234?"
AI: [Retrieves from knowledge base and responds with context]
```

**Validate uploaded reports**:
1. Click upload button → Select PDF report
2. Type "validate" in chat to validate CVE usage
3. Or type "summarize" for executive summary
4. Or type "add to kb" to add to knowledge base

**Manage knowledge base**:
- Use right panel "Add Files" to expand knowledge base
- View all sources with chunk counts
- Click refresh to update statistics

#### Recent Updates

✅ **Phase 2 (LangChain) - Response Quality Fixed** (2025-01-18):
- **Previous issues** (now resolved):
  - Frequent "I don't know" responses despite relevant KB content
  - Occasional hangs with unresponsive frontend
- **Root cause**: Inconsistent query paths after hybrid search implementation
- **Solution**: Refactored `query()` method to use unified approach consistent with Phase 1
- **Status**: Both Phase 1 and Phase 2 now provide comparable response quality
- **Testing**: Real-world validation recommended (see `docs/PROGRESS.md` for technical details)

#### Current Limitations

**Chat File Upload (Left Panel)**:
- **Single file mode**: Uploading a new file replaces the previous one
- Files are deleted after sending message or manual removal
- No persistence across conversation turns
- **Planned improvement**: Multi-file conversation context (see `docs/PROGRESS.md` - "Upcoming Features")
  - Future: Retain multiple files in conversation session
  - Future: Generate temporary embeddings without persisting to Chroma
  - Future: Query both permanent KB and session files simultaneously

**Knowledge Base Upload (Right Panel)**:
- Files are permanently added with embeddings stored in Chroma
- No deletion after use
- Supports multiple files (no single-file limitation)

#### Comparison: Phase 1 vs Phase 2

| Feature | Phase 1 (Pure Python) | Phase 2 (LangChain) |
|---------|----------------------|---------------------|
| **Port** | 7860 | 7861 |
| **History Management** | Manual (deque, 10 rounds) | Automatic (ConversationBufferWindowMemory) |
| **RAG Workflow** | Custom logic | ConversationalRetrievalChain |
| **Embeddings** | Direct SentenceTransformer | HuggingFaceEmbeddings wrapper |
| **Vector Store** | Direct Chroma queries | LangChain Chroma wrapper |
| **Code Complexity** | More control, more code | Abstraction, less code |
| **Use Case** | Learning internals, debugging | Standardized workflow, faster development |

**When to use each**:
- **Phase 1**: Fine-grained control, understanding RAG internals, custom prompts
- **Phase 2**: Rapid prototyping, standardized workflow, LangChain ecosystem integration

### Step 7: Incremental Knowledge Base Updates

Use `cli/addToEmbeddings.py` to add new documents to an existing knowledge base:

```bash
# Add single PDF
python cli/add_to_embeddings.py --pdf=new_report.pdf

# Add multiple PDFs
python cli/add_to_embeddings.py --pdf="report1.pdf,report2.pdf,report3.pdf"

# Add CVE data by year
python cli/add_to_embeddings.py --cve-year=2024

# Specify target database (must already exist)
python cli/add_to_embeddings.py --pdf=report.pdf --target=cve_embeddings.pkl

# With custom speed and format
python cli/add_to_embeddings.py --pdf=report.pdf --speed=fast --extension=chroma
```

**Note**: The target embedding database must already exist (created by `cli/build_embeddings.py`). This tool only adds incremental updates.

## Configuration System

### Environment Variables (.env)

The project uses `.env` for configuration. Copy `.env.example` to `.env` and customize:

```bash
# Paths
CHROMA_DB_PATH=./cve_embeddings.chroma
CVE_V5_PATH=../cvelistV5/cves
CVE_V4_PATH=../cvelist

# Models
LLAMA_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
EMBEDDING_MODEL_NAME=all-mpnet-base-v2

# Defaults
DEFAULT_SPEED=fast              # normal, fast, fastest
DEFAULT_MODE=full               # demo, full
DEFAULT_SCHEMA=all              # v5, v4, all

# Web UI
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False

# Advanced
CONVERSATION_HISTORY_LENGTH=10
RETRIEVAL_TOP_K=5
LLM_TEMPERATURE=0.3
LLM_TOP_P=0.9
CHUNK_SIZE=10
EMBEDDING_BATCH_SIZE=64
EMBEDDING_PRECISION=float16
VERBOSE_LOGGING=False
```

### Configuration Loading (config.py)

All scripts automatically load configuration from `.env`:

```python
from config import (
    LLAMA_MODEL_NAME,
    DEFAULT_SPEED,
    GRADIO_SERVER_PORT,
    # ... other settings
)
```

**Priority**: Environment variables override defaults in `config.py`.

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
- Loads `cve_embeddings.csv` with precomputed embeddings
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
├── cli/                 # Command-line tools
│   ├── validate_report.py # Main RAG application
│   ├── build_embeddings.py # Generate embedding database
│   ├── add_to_embeddings.py # Incremental knowledge base updates
│   ├── extract_cve.py   # Optional: export CVE descriptions
│   └── cleanup_cache.py # Clean Llama model cache
├── core/                # Shared modules (Phase 1 & 2)
│   ├── __init__.py
│   ├── models.py        # Llama model loading wrapper
│   ├── embeddings.py    # SentenceTransformer wrapper
│   ├── chroma_manager.py # Chroma CRUD operations
│   ├── cve_lookup.py    # CVE JSON file queries
│   └── pdf_processor.py # PDF text extraction
├── rag/                 # RAG implementations
│   ├── __init__.py
│   ├── pure_python.py   # Phase 1: Manual implementation
│   └── langchain_impl.py # Phase 2: LangChain wrapper
├── web/                 # Web interfaces
│   ├── web_ui.py        # Phase 1: Pure Python (port 7860)
│   └── web_ui_langchain.py # Phase 2: LangChain (port 7861)
├── docs/                # Documentation
│   ├── ARCHITECTURE.md  # System architecture and technical details
│   └── PROGRESS.md      # Completed changes and upcoming features
├── embeddings/          # Prebuilt embedding files (tracked)
│   ├── cve_embeddings.csv
│   ├── cve_embeddings.pkl
│   ├── cve_embeddings.parquet
│   └── cve_embeddings.chroma/
├── samples/             # Sample PDF files (tracked)
│   ├── CVEDocument.pdf
│   └── CVEpdf2024.pdf
├── scripts/             # Environment setup scripts
│   └── windows/         # Windows PowerShell scripts
│       ├── Setup-CPU.ps1    # CPU-only environment
│       ├── Setup-CUDA118.ps1 # CUDA 11.8 environment
│       └── Setup-CUDA124.ps1 # CUDA 12.4 environment
├── config.py            # Unified configuration (loads from .env)
├── .env.example         # Configuration template (committed)
├── .env                 # Local configuration (gitignored)
├── requirements.txt     # Base dependencies (includes pyarrow, chromadb, langchain)
├── CLAUDE.md            # Project documentation (this file)
├── README.md            # Project overview and quick start
├── LICENSE.md           # License information
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

#### CLI Tools
- **`cli/validate_report.py`** (CLI RAG application):
  - Reads: User PDF, `cve_embeddings.{extension}`, CVE JSON feeds
  - Uses: `core/` modules, embedding database
  - Outputs: Interactive menu with Summary/Validate/Q&A options

- **`cli/build_embeddings.py`** (Initial embedding generation):
  - Reads: User-provided PDF corpus
  - Writes: `cve_embeddings.{extension}` (csv/pkl/parquet/chroma)
  - Speed: normal (baseline), fast (default, 1.5-2x), fastest (2-3x)

- **`cli/addToEmbeddings.py`** (Incremental updates):
  - Reads: Existing `cve_embeddings.{extension}`, new PDFs
  - Writes: Updated `cve_embeddings.{extension}` with new documents
  - Note: Target database must already exist

- **`cli/extractCVE.py`** (CVE reference extraction):
  - Reads: `../cvelistV5/cves/<year>/` (v5) or `../cvelist/<year>/` (v4)
  - Writes: `CVEDescription<year>.txt` or `CVEDescription<min>-<max>.txt`
  - Supports: v5/v4 schema selection, year ranges, deduplication

#### Web Interfaces
- **`web/web_ui.py`** (Phase 1: Pure Python):
  - Uses: `rag/pure_python.py`, `core/` modules
  - Port: 7860
  - Features: Chat, validation, knowledge base management

- **`web/web_ui_langchain.py`** (Phase 2: LangChain):
  - Uses: `rag/langchain_impl.py`, `core/` modules
  - Port: 7861
  - Features: Same as Phase 1, with LangChain automatic memory

#### Core Modules (Shared)
- **`core/models.py`**: Llama model initialization and generation
- **`core/embeddings.py`**: SentenceTransformer embedding operations
- **`core/chroma_manager.py`**: Vector database CRUD (add, query, delete, stats)
- **`core/cve_lookup.py`**: CVE JSON file parsing and batch lookup
- **`core/pdf_processor.py`**: PDF text extraction with PyMuPDF

#### RAG Implementations
- **`rag/pure_python.py`**: Manual RAG with deque-based history (Phase 1)
- **`rag/langchain_impl.py`**: LangChain chains and memory (Phase 2)

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
- Used for both corpus building (`cli/build_embeddings.py`) and query encoding (`cli/validate_report.py`)

## Important Notes

- **Sync CVE Feeds**: Ensure `../cvelistV5/` and `../cvelist/` are up-to-date to minimize "Could Not Find" fallbacks
- **Embedding File Dependency**: `cve_embeddings.{extension}` must exist before running `cli/validate_report.py`
  - Generate with `cli/localEmbedding.py --extension={extension}`
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
