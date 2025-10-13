# Progress

This file tracks completed changes and upcoming features for the project.

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
