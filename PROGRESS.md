# Progress

This file tracks completed changes and upcoming features for the project.

## [2025-01] Embedding Optimization & File Format Support

### Added (localEmbedding.py)
- **Three-level speed control** via `--speed` parameter:
  - `normal`: Baseline quality (float32, batch_size=32, chunk_size=10)
  - `fast`: Recommended default (float16, batch_size=64, chunk_size=10, 1.5-2x faster)
  - `fastest`: Maximum speed (float16, batch_size=128, chunk_size=20, 2-3x faster)
- **Batch encoding optimization**: Process all chunks at once instead of one-by-one (10-20x faster on GPU, 3-5x on CPU)
- **Multiple output formats** via `--extension` parameter:
  - `csv`: Text format, largest (~95 MB), slowest read, maximum compatibility
  - `pkl`: Pickle (default), balanced (~33 MB), Python-native
  - `parquet`: Parquet, smallest (~24 MB), fastest read (5-10x), requires pyarrow
- **Automatic device selection**: Initializes SentenceTransformer directly on correct device (CPU/CUDA)
- **Progress reporting**: Real-time progress bar during batch encoding

### Added (theRag.py)
- **Embedding format support** via `--extension` parameter:
  - Reads `CVEEmbeddings.{extension}` based on parameter
  - Supports csv, pkl (default), parquet formats
  - File existence check with helpful error messages
- **Format-specific loading**: Optimized readers for each format

### Added (extractCVE.py)
- **Verbose mode** via `--verbose` or `-v` flag:
  - Default: Shows progress bar with file count
  - Verbose: Shows detailed file-by-file logging
- **Progress bar integration** using tqdm (non-verbose mode)

### Changed
- **localEmbedding.py**: Default output format changed to `.pkl` (from `.csv`)
- **theRag.py**: Default embedding format changed to `.pkl` (from `.csv`)
- **requirements.txt**: Added `pyarrow` for parquet support
- User input flow: Now asks for base filename without extension

### Performance Impact (localEmbedding.py)
- **Batch encoding speedup**:
  - GPU (GTX 1660 Ti): 10-20x faster than one-by-one encoding
  - CPU: 3-5x faster than one-by-one encoding
- **File size reduction**:
  - pkl: 65% smaller than csv (~33 MB vs 95 MB)
  - parquet: 75% smaller than csv (~24 MB vs 95 MB)
- **Read speed improvement** (theRag.py):
  - pkl: 3-5x faster than csv
  - parquet: 5-10x faster than csv

### Technical Details
- Batch encoding uses `SentenceTransformer.encode()` with configurable `batch_size`
- Float16 precision reduces memory bandwidth by 50% with negligible quality loss
- Float16 conversion applied post-encoding via `embeddings.astype(np.float16)`
- Larger chunk sizes (20 vs 10) reduce total chunk count by ~50%
- Parquet format uses columnar storage with snappy compression

### Verified Configurations
✅ **Tested and verified working (2025-01-13)**:
- `localEmbedding.py --speed=fast --extension=pkl` → CVEEmbeddings.pkl (~33 MB)
- `localEmbedding.py --speed=fastest --extension=parquet` → CVEEmbeddings.parquet (~24 MB)
- `theRag.py --speed=fast --extension=pkl` → Successfully loaded and processed
- `theRag.py --speed=fastest --extension=parquet` → Successfully loaded and processed
- All combinations tested on CUDA 11.8 (GTX 1660 Ti) without issues

### Bug Fixes
- Fixed `SentenceTransformer.encode()` dtype parameter issue (not supported)
- Changed to post-encoding conversion: `embeddings.astype(np.float16)`
- Updated setup scripts to use `python -m pip` instead of direct `pip.exe` calls
  - Resolves shebang issues after venv rename (venv-* → .venv-*)
  - More reliable and follows Python best practices

## [2025-01] Multi-Level Speed Optimization

### Added
- **Three-level speed control** via `--speed` parameter:
  - `normal`: Baseline with chunk-aware filtering (FP32, frequent cache clearing)
  - `fast`: Recommended default (FP16, reduced cache clearing)
  - `fastest`: Maximum speed (FP16, reduced cache, low temperature, SDPA)
- SDPA (Scaled Dot-Product Attention) support for fastest mode
- Configurable cache clearing frequency (every 1 or 3 chunks)
- Dynamic temperature control (0.3 for normal/fast, 0.1 for fastest)

### Performance Impact
- **Option 2 (CVE Validation)**:
  - Original: 34.4 minutes (2.6 tok/s)
  - normal: 4.5 minutes (7.6x faster)
  - fast: 3.1 minutes (11x faster) ⭐ **default**
  - fastest: 2.1 minutes (16x faster)

- **Option 1 (Summarization)** & **Option 3 (Q&A)**:
  - 30-60% faster with fast/fastest modes

### Changed
- Default speed level is now `fast` (was implicit normal)
- FP16 enabled by default for better GPU utilization
- Cache clearing optimized to every 3 chunks (from every chunk)

## [2025-01] Chunk-Aware CVE Filtering

### Added
- Intelligent CVE filtering in Option 2 (CVE Validation)
- Per-chunk CVE extraction using regex
- Dynamic system prompt generation with only relevant CVEs

### Performance Impact
- Reduced input tokens from 3,934 → ~1,900 per chunk (50% reduction)
- Generation speed improved from 2.6 tok/s → 15-20 tok/s (6-8x faster)
- Memory bandwidth usage reduced by 50%

### Technical Details
- Parses CVE descriptions into dictionary for O(1) lookup
- Extracts CVEs from each chunk using `extract_cves_regex()`
- Filters CVE descriptions to only include mentioned CVEs
- 200-token overlap between chunks catches cross-references

### Changed
- `menu_option_2()` full mode now uses chunk-aware filtering
- No changes to Option 1 (Summarization) or Option 3 (Q&A)
- Demo mode unchanged (already uses truncation)

## [2025-01] CUDA Environment Updates

### Added
- CUDA 12.4 support for RTX 4060+ GPUs
- Setup script: `scripts/setup-cuda124.ps1`
- Virtual environment: `.venv-cuda124`
- CUDA migration guide: Documented upgrade path from 12.1 → 12.4

### Changed
- Updated CLAUDE.md with CUDA 12.4 as recommended version
- Renamed CUDA 12.1 references to CUDA 11.8 for GTX 1660 Ti compatibility
- Corrected GPU references (GTX 1650 → GTX 1660 Ti in documentation)

### Performance
- RTX 4060: 20-40x faster than CPU
- GTX 1660 Ti: 10-20x faster than CPU

## [2024-12] CVE Schema Unification

### Added
- Flexible schema support via `--schema` parameter:
  - `v5`: CVE 5.0 schema only (fastest)
  - `v4`: CVE 4.0 schema only
  - `all`: V5→V4 fallback (default, backward compatible)
- Auto-detection of v4/v5 JSON schema formats
- Unified field extraction for both schemas

### Changed
- `load_cve_record()` now supports schema parameter
- `extract_cve_fields()` detects schema automatically
- `extractCVE.py` supports schema selection and deduplication
- Default behavior: Try V5 first, fallback to V4 if not found

### Migration
- Renamed `extractCVE4.py` → `extractCVE.py`
- Removed hardcoded year references
- Updated all CVE list paths to support both schemas

## [2024-12] File Organization

### Added
- `ARCHITECTURE.md`: Technical details and system architecture (contains optimization guide)
- `PROGRESS.md`: This file, tracking completed changes and upcoming features

### Changed
- Renamed `cveEmbeddings.csv` → `CVEEmbeddings.csv` for naming consistency
- Standardized virtual environment naming: `venv-*` → `.venv-*` (dot prefix)
- All setup scripts now create dot-prefixed venvs

### Removed
- Intermediate optimization files (theRagFast.py, theRagDebug.py, theRag_original.py)
- Redundant README files (DEBUG_README.md, FAST_README.md, SPEED_LEVELS_README.md)
- CUDA_MIGRATION.md (content merged into ARCHITECTURE.md)

### Git Changes
- Updated .gitignore to support both `venv-*` and `.venv-*` patterns
- Consolidated documentation structure

## [2024-11] Core Optimizations

### Added
- Direct regex CVE extraction (no LLM overhead)
- Global SentenceTransformer initialization (loaded once at startup)
- Unified `torch.no_grad()` wrapper for all inference calls
- Context length limits for missing CVE processing

### Performance Impact
- CVE extraction: Instant (was 10-30 seconds with LLM)
- Memory: Reduced by initializing SentenceTransformer once
- Generation efficiency: All calls use no_grad context

### Changed
- CVE extraction pattern: `CVE-\d{4}-\d{4,7}`
- Missing CVE context window: 1500 chars (full mode), 500 chars (demo mode)
- All `model.generate()` calls include `pad_token_id=tokenizer.eos_token_id`

## [2024-11] Dual-Mode Architecture

### Added
- **Demo mode** (`--mode=demo`): Memory-optimized for limited hardware
  - First 10 pages only
  - Text truncation (1000-2000 chars)
  - FP16 precision
  - Limited embedding rows (1000)
  - Token generation: 64-256 tokens

- **Full mode** (default): Complete feature set
  - All pages processed
  - Intelligent chunking (1500 tokens, 200 overlap)
  - Auto precision
  - Complete embedding database
  - Token generation: 150-700 tokens

### Performance
- Demo mode: ~4-6 GB RAM / ~3.5-4 GB VRAM
- Full mode: ~6-8 GB RAM / ~4-5 GB VRAM

## Project History

### Initial Release [2024-10]
- RAG-based CVE validation system
- Llama 3.2-1B-Instruct integration
- SentenceTransformer embeddings (all-mpnet-base-v2)
- Three analysis options: Summarize, Validate CVE, Q&A
- CVE JSON feed support (V4 schema)
- PyMuPDF text extraction
- CPU and CUDA support

### Repository Structure
```
RAG_LLM_CVE/
├── theRag.py              # Main application (optimized)
├── localEmbedding.py      # Generate embedding database
├── extractCVE.py          # Export CVE descriptions (optional)
├── cleanupLlamaCache.py   # Clean model cache
├── CLAUDE.md              # Project documentation and user guide
├── ARCHITECTURE.md        # Technical details and system architecture
├── PROGRESS.md            # This file (completed changes and upcoming features)
├── requirements.txt       # Python dependencies
├── scripts/               # Environment setup scripts
│   ├── setup-cpu.ps1
│   ├── setup-cuda118.ps1
│   └── setup-cuda124.ps1
└── .venv-*/               # Virtual environments (gitignored)
```

## Upcoming Features

### Planned Optimizations
- [ ] Parallel CVE lookups (if memory permits)
- [ ] Progress bars for long-running operations
- [ ] Batch processing for multiple PDFs
- [ ] Input validation for CSV existence and format

### Under Consideration
- [ ] Web interface for easier access
- [ ] Support for additional LLM models
- [ ] Real-time CVE feed updates
- [ ] Export analysis results to JSON/HTML

## Notes

- All performance benchmarks based on GTX 1660 Ti (6GB VRAM)
- Times may vary based on PDF size, CVE count, and hardware
- GPU acceleration provides 10-40x speedup depending on model
- Memory usage scales with PDF size and embedding database size
