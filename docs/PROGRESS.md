# Progress

This file tracks completed changes and upcoming features for the project.

## [2025-10] Phase 1: Web UI and Knowledge Base Enhancement

### Added (Core Modules)
- **config.py**: Unified configuration system with `.env` support
  - Path configuration (Chroma DB, CVE feeds, temp uploads)
  - Model names (Llama, SentenceTransformer)
  - Default parameters (speed, mode, schema, precision)
  - RAG configuration (history length, retrieval top-k, chunk settings)
  - Web UI configuration (port, share, server name)
  - Validation functions and debug utilities
- **core/models.py**: Llama model loading and management
  - LlamaModel class with FP16, SDPA, low memory support
  - Backward compatible utility functions
  - Automatic device selection
  - cleanup() for resource management
- **core/embeddings.py**: SentenceTransformer wrapper
  - EmbeddingModel class with batch encoding
  - retrieve_top_k() for similarity search
  - Precision conversion (float32/float16)
  - Backward compatible utility functions
- **core/chroma_manager.py**: Chroma vector database CRUD
  - ChromaManager class for database operations
  - add_documents() with metadata support
  - query() with filtering
  - delete_by_source() for source removal
  - get_stats() and list_sources() for management
  - Metadata schema: source_type, source_name, added_date, chunk_index, precision
- **core/cve_lookup.py**: CVE JSON parsing (V4/V5)
  - lookup_cve() for single CVE lookup
  - batch_lookup_cves() for multiple CVEs
  - extract_cves_regex() for regex extraction
  - format_cve_description() for output formatting
- **core/pdf_processor.py**: PDF text extraction
  - PDFProcessor class with PyMuPDF
  - extract_text() and extract_text_by_pages()
  - extract_cve_context() for context windows

### Added (RAG Implementation)
- **rag/pure_python.py**: Conversation-aware RAG (Phase 1)
  - ConversationHistory class (fixed-size deque, last 10 rounds)
  - PureRAG class with full RAG workflow
  - query() with history-aware context retrieval
  - summarize_report() for executive summaries
  - validate_cve_usage() for CVE correctness checking
  - answer_question_about_report() for Q&A
  - process_report_for_cve_validation() for PDF + CVE lookup

### Added (Tools and Web UI)
- **add_to_embeddings.py**: Incremental knowledge base updates
  - Add PDFs: `--source=pdf --files=report.pdf,report2.pdf`
  - Add CVE data: `--source=cve --year=2024 --schema=v5`
  - Configurable chunk size and batch size
  - Automatic metadata tagging
  - Progress bars and error handling
- **web/web_ui.py**: Gradio web interface (Phase 1)
  - Claude Projects-style layout (left chat + right settings/KB)
  - Conversational AI with 10-round history
  - Upload PDF for validation (summarize/validate/add to KB)
  - Knowledge base management (add/view/refresh)
  - Real-time statistics display
  - Analysis settings (speed/mode/schema dropdowns)
  - Auto-launch browser on startup

### Changed
- **.gitignore**: Added `.env` and `temp_uploads/` exclusions
- **requirements.txt**: Added `python-dotenv` and `gradio`
- Repository structure: Added `core/`, `rag/`, `web/` directories

### Repository Structure (Updated)
```
RAG_LLM_CVE/
├── core/                  # Shared modules (Phase 1)
│   ├── models.py          # Llama loading
│   ├── embeddings.py      # SentenceTransformer
│   ├── chroma_manager.py  # Chroma CRUD
│   ├── cve_lookup.py      # CVE parsing
│   └── pdf_processor.py   # PDF extraction
├── rag/                   # RAG implementations
│   └── pure_python.py     # Phase 1: Pure Python RAG
├── web/                   # Web interfaces
│   └── webUI.py           # Phase 1: Gradio UI
├── validate_report.py     # CLI application (original)
├── build_embeddings.py    # Generate embeddings
├── add_to_embeddings.py   # Incremental updates (new)
├── extract_cve.py         # Export CVE descriptions
├── config.py              # Configuration loader (new)
├── .env.example           # Config template (new)
├── .env                   # Local config (gitignored)
├── FEATURE_PLAN.md        # Planning document (new)
├── CLAUDE.md              # User guide
├── ARCHITECTURE.md        # Technical details
└── PROGRESS.md            # This file
```

### Usage Examples
```bash
# Web UI (recommended for demos)
python web/web_ui.py

# Add PDFs to knowledge base
python cli/add_to_embeddings.py --source=pdf --files=report1.pdf,report2.pdf

# Add CVE data to knowledge base
python cli/add_to_embeddings.py --source=cve --year=2024 --schema=v5

# Original CLI (still works)
python cli/validate_report.py --speed=fast --extension=chroma
```

### Phase 1 Goals Achieved
✅ Core modules created and tested (all imports successful)
✅ Configuration system with .env support
✅ Incremental knowledge base updates (PDFs and CVE data)
✅ Conversation-aware RAG implementation
✅ Gradio web UI with Claude Projects-style layout
✅ Knowledge base management (add/view sources)
✅ Real-time statistics and refresh

## [2025-10] Phase 2: LangChain Implementation

### Added (LangChain RAG)
- **rag/langchain_impl.py**: LangChain-based RAG system
  - LangChainRAG class with automatic initialization
  - HuggingFacePipeline wrapper for Llama model
  - HuggingFaceEmbeddings for SentenceTransformer
  - ConversationBufferWindowMemory (k=10 rounds, automatic management)
  - ConversationalRetrievalChain for standardized RAG workflow
  - Chroma vectorstore integration via LangChain
  - query() with automatic conversation history
  - summarize_report(), validate_cve_usage(), answer_question_about_report()
  - process_report_for_cve_validation() for PDF processing
  - add_document_to_kb() with automatic embedding generation
  - get_kb_stats() and delete_source() for KB management

### Added (LangChain Web UI)
- **web/web_ui_langchain.py**: Gradio interface using LangChain
  - Same Claude Projects-style layout as Phase 1
  - Uses LangChainRAG for backend
  - Automatic memory management (no manual history tracking)
  - Port 7861 (Phase 1 uses 7860) - both can run simultaneously
  - Clear labeling as "LangChain" version

### Key Differences: Phase 1 vs Phase 2

| Feature | Phase 1 (Pure Python) | Phase 2 (LangChain) |
|---------|----------------------|---------------------|
| **Conversation History** | Manual (deque with sliding window) | Automatic (ConversationBufferWindowMemory) |
| **RAG Workflow** | Custom query + retrieval logic | ConversationalRetrievalChain |
| **Embeddings** | Direct SentenceTransformer usage | HuggingFaceEmbeddings abstraction |
| **Vector Store** | Direct Chroma client queries | LangChain Chroma wrapper |
| **Code Complexity** | Lower-level control, more code | Higher-level abstraction, less code |
| **Flexibility** | Full control over every step | Standardized patterns |
| **Learning Curve** | Understand RAG internals | Understand LangChain APIs |
| **Port** | 7860 | 7861 |

### Usage Examples
```bash
# Phase 1 (Pure Python, port 7860)
python web/web_ui.py

# Phase 2 (LangChain, port 7861)
python web/web_ui_langchain.py

# Both can run simultaneously for A/B comparison
```

### Repository Structure (Final)
```
RAG_LLM_CVE/
├── core/                  # Shared modules
│   ├── models.py          # Llama loading
│   ├── embeddings.py      # SentenceTransformer
│   ├── chroma_manager.py  # Chroma CRUD
│   ├── cve_lookup.py      # CVE parsing
│   └── pdf_processor.py   # PDF extraction
├── rag/                   # RAG implementations
│   ├── pure_python.py     # Phase 1: Manual implementation
│   └── langchain_impl.py  # Phase 2: LangChain (new)
├── web/                   # Web interfaces
│   ├── web_ui.py          # Phase 1: Pure Python
│   └── web_ui_langchain.py # Phase 2: LangChain (new)
├── validate_report.py     # CLI application (original)
├── build_embeddings.py    # Generate embeddings
├── add_to_embeddings.py   # Incremental updates
├── extract_cve.py         # Export CVE descriptions
├── config.py              # Configuration loader
├── .env.example           # Config template
└── FEATURE_PLAN.md        # Planning document
```

### Phase 2 Goals Achieved
✅ LangChain RAG implementation with chains and memory
✅ ConversationalRetrievalChain integration
✅ Automatic conversation history management
✅ Gradio web UI using LangChain backend
✅ Coexistence with Phase 1 (both can run simultaneously)
✅ Standardized LangChain patterns throughout
✅ All imports tested and working

### Performance Comparison (To Be Tested)
- [ ] Memory usage: Phase 1 vs Phase 2
- [ ] Response latency: Phase 1 vs Phase 2
- [ ] Conversation quality: Phase 1 vs Phase 2
- [ ] Code maintainability assessment

### Recommendations
- **Phase 1 (pure_python.py)**: Best for learning RAG internals, maximum control
- **Phase 2 (langchain_impl.py)**: Best for production, standardized patterns, community support
- **Both**: Keep both for comparison and fallback options

## [2025-01] Embedding Optimization & File Format Support

### Added (build_embeddings.py)
- **Three-level speed control** via `--speed` parameter:
  - `normal`: Baseline quality (float32, batch_size=32, chunk_size=10)
  - `fast`: Recommended default (float16, batch_size=64, chunk_size=10, 1.5-2x faster)
  - `fastest`: Maximum speed (float16, batch_size=128, chunk_size=20, 2-3x faster)
- **Batch encoding optimization**: Process all chunks at once instead of one-by-one (10-20x faster on GPU, 3-5x on CPU)
- **Multiple output formats** via `--extension` parameter:
  - `csv`: Text format, largest (~95 MB), slowest read, maximum compatibility
  - `pkl`: Pickle (default), balanced (~33 MB), Python-native
  - `parquet`: Parquet, smallest (~24 MB), fastest read (5-10x), requires pyarrow
  - `chroma`: Vector database (directory-based), optimized queries, no server required, best for large datasets
- **Automatic device selection**: Initializes SentenceTransformer directly on correct device (CPU/CUDA)
- **Progress reporting**: Real-time progress bar during batch encoding

### Added (validate_report.py)
- **Embedding format support** via `--extension` parameter:
  - Reads `CVEEmbeddings.{extension}` (or directory for chroma) based on parameter
  - Supports csv, pkl (default), parquet, chroma formats
  - File/directory existence check with helpful error messages
- **Format-specific loading**: Optimized readers for each format
  - Chroma: Direct vector database queries (no need to load all embeddings into memory)

### Added (extract_cve.py)
- **Verbose mode** via `--verbose` or `-v` flag:
  - Default: Shows progress bar with file count
  - Verbose: Shows detailed file-by-file logging
- **Progress bar integration** using tqdm (non-verbose mode)

### Changed
- **build_embeddings.py**: Default output format changed to `.pkl` (from `.csv`)
- **validate_report.py**: Default embedding format changed to `.pkl` (from `.csv`)
- **requirements.txt**: Added `pyarrow` for parquet support, `chromadb` for vector database support
- User input flow: Now asks for base filename without extension (for file-based formats)
- Chroma format creates directory instead of file (e.g., `cve_embeddings/` instead of `cve_embeddings.pkl`)

### Performance Impact (build_embeddings.py)
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
- `build_embeddings.py --speed=fast --extension=pkl` → cve_embeddings.pkl (~33 MB)
- `build_embeddings.py --speed=fastest --extension=parquet` → cve_embeddings.parquet (~24 MB)
- `validate_report.py --speed=fast --extension=pkl` → Successfully loaded and processed
- `validate_report.py --speed=fastest --extension=parquet` → Successfully loaded and processed
- All combinations tested on CUDA 11.8 (GTX 1660 Ti) without issues

✅ **Chroma integration (2025-01-14)**:
- Added chromadb to requirements.txt
- `build_embeddings.py --extension=chroma` → Creates cve_embeddings/ directory with vector database
- `validate_report.py --extension=chroma` → Direct query from Chroma database (no memory loading)
- Persistent client mode (no server required)

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
├── validate_report.py     # Main application (optimized)
├── build_embeddings.py    # Generate embedding database
├── extract_cve.py         # Export CVE descriptions (optional)
├── cleanup_cache.py       # Clean model cache
├── CLAUDE.md              # Project documentation and user guide
├── ARCHITECTURE.md        # Technical details and system architecture
├── PROGRESS.md            # This file (completed changes and upcoming features)
├── requirements.txt       # Python dependencies
├── scripts/               # Environment setup scripts
│   └── windows/           # Windows PowerShell scripts
│       ├── Setup-CPU.ps1
│       ├── Setup-CUDA118.ps1
│       └── Setup-CUDA124.ps1
└── .venv-*/               # Virtual environments (gitignored)
```

## Upcoming Features

### Planned Optimizations
- [ ] **Chunk overlap for PDF embeddings** (High Priority)
  - Add `--overlap` parameter to `build_embeddings.py`
  - Default: 30% overlap for PDF documents (0% for CVE data)
  - Expected impact: +13-17% retrieval accuracy, +40% storage/time cost
  - Rationale: Prevents splitting important context across chunk boundaries
  - Implementation: Replace `split()` with `split_with_overlap()` in PDF processing
  - Cost-benefit: Acceptable trade-off for security reports where accuracy is critical
- [ ] Parallel CVE lookups (if memory permits)
- [ ] Progress bars for long-running operations
- [ ] Batch processing for multiple PDFs
- [ ] Input validation for CSV existence and format

### Under Consideration
- [ ] Web interface for easier access
- [ ] Support for additional LLM models
- [ ] Real-time CVE feed updates
- [ ] Export analysis results to JSON/HTML
- [ ] Configurable overlap ratio per document type (PDF vs CVE)

## Notes

- All performance benchmarks based on GTX 1660 Ti (6GB VRAM)
- Times may vary based on PDF size, CVE count, and hardware
- GPU acceleration provides 10-40x speedup depending on model
- Memory usage scales with PDF size and embedding database size
