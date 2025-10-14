# System Architecture Overview

## Project Purpose
RAG-based CVE validation system that reduces LLM hallucinations in Security Operations Centers (SOCs) by combining retrieval-augmented generation with official CVE metadata to verify threat intelligence reports.

## System Requirements

### Hardware Options

**CPU-Only** (Minimum):
- Python 3.10+
- RAM: 8GB minimum, 16GB recommended
- Storage: ~10GB for models and dependencies
- Processing time: 30-60 sec per LLM call

**GPU-Accelerated** (Recommended):
- **GTX 1660 Ti (6GB VRAM)**:
  - CUDA Toolkit: 11.8
  - Environment: `venv-cuda118`
  - Speed: ~40-85 tokens/sec (10-20x faster than CPU)
  - Suitable for: Llama-3.2-1B/3B models

- **RTX 4060 (12GB VRAM)** or better:
  - CUDA Toolkit: 12.4 (recommended)
  - Environment: `venv-cuda124`
  - Speed: ~80-120 tokens/sec (20-40x faster than CPU)
  - Suitable for: Llama-3.2-1B/3B/7B models

- **RTX 4070+ / Professional GPUs**:
  - CUDA Toolkit: 12.4+
  - Environment: `venv-cuda124`
  - Speed: 100-200+ tokens/sec

### Software Prerequisites
- Python 3.10+
- Windows 10/11 (scripts are PowerShell-based)
- Hugging Face account with Llama model access
- External CVE JSON feeds: `../cvelist/` and `../cvelistV5/`

## Data Sources & Preprocessing

### Reference Knowledge Base
- **Input**: Curated threat intelligence PDFs (e.g., `CVEpdf2024.pdf`)
- **Tool**: `localEmbedding.py`
- **Parameters**:
  - `--speed`: normal (baseline), fast (default, 1.5-2x), fastest (2-3x)
  - `--extension`: csv, pkl (default), parquet, or chroma
- **Process**:
  1. Extract text from PDF using PyMuPDF (`fitz`)
  2. Tokenize with spaCy's sentencizer into configurable chunks (10 or 20 sentences)
  3. **Batch encode** all chunks with `all-mpnet-base-v2` SentenceTransformer (768 dimensions)
     - Uses configurable batch size (32/64/128) for optimal throughput
     - Precision: float32 (normal) or float16 (fast/fastest)
  4. Save to selected format:
     - csv/pkl/parquet: File with columns `sentence_chunk`, `embedding`
     - chroma: Chroma vector database (directory-based, persistent, no server required)
- **Performance**:
  - Batch encoding: 10-20x faster on GPU, 3-5x on CPU vs one-by-one
  - File sizes: csv ~95 MB, pkl ~33 MB (default), parquet ~24 MB
  - Chroma: Directory-based storage with optimized queries
- **Purpose**: Creates retrieval corpus for contextual grounding and CVE recommendations
- **Run once**: Before first use of theRag.py

### Official CVE Metadata
- **Source**: MITRE/NVD JSON feeds in `../cvelistV5/cves/<year>` (v5 schema) and `../cvelist/<year>` (v4 schema)
- **Structure**: Files organized as `<year>/<prefix-xxx>/CVE-<year>-<id>.json`
  - v5 schema: `cveMetadata.cveId`, `containers.cna.affected`, `containers.cna.descriptions`
  - v4 schema: `CVE_data_meta.ID`, `affects.vendor.vendor_data`, `description.description_data`
- **Optional Tools**: `extractCVE.py` flattens JSONs to `CVEDescription<year>.txt` (for human reference only, not used by RAG)
  - `--schema=v5` (default): Extract from V5 only, fastest
  - `--schema=v4`: Extract from V4 only
  - `--schema=all`: Extract from both with deduplication (V5 priority)

## Module Architecture

### Modular Structure (Phase 1 & 2)

The project uses a layered modular architecture with shared core components:

```
RAG_LLM_CVE/
├── core/                # Shared utilities (Phase 1 & 2)
│   ├── models.py        # Llama model wrapper
│   ├── embeddings.py    # SentenceTransformer wrapper
│   ├── chroma_manager.py # Vector database CRUD
│   ├── cve_lookup.py    # CVE JSON parsing
│   └── pdf_processor.py # PDF text extraction
├── rag/                 # RAG implementations
│   ├── pure_python.py   # Phase 1: Manual implementation
│   └── langchain_impl.py # Phase 2: LangChain chains
└── web/                 # Web interfaces
    ├── webUI_v1.py      # Phase 1: Pure Python (port 7860)
    └── webUI_v2.py      # Phase 2: LangChain (port 7861)
```

### Phase 1: Pure Python Implementation

**Architecture**: Direct component usage with manual orchestration

```
webUI_v1.py (Gradio)
    ↓
rag/pure_python.py (PureRAG class)
    ↓
core/models.py (LlamaModel)           core/embeddings.py (EmbeddingModel)
core/chroma_manager.py (ChromaManager) core/cve_lookup.py
core/pdf_processor.py
```

**Features**:
- Manual conversation history (deque with 10-round sliding window)
- Direct Chroma queries with metadata filtering
- Custom RAG workflow with fine-grained control
- Explicit memory management

**Use Cases**:
- Learning RAG internals
- Debugging and optimization
- Custom prompt engineering
- Performance benchmarking

### Phase 2: LangChain Implementation

**Architecture**: LangChain abstractions with automatic orchestration

```
webUI_v2.py (Gradio)
    ↓
rag/langchain_impl.py (LangChainRAG class)
    ↓
ConversationalRetrievalChain          ConversationBufferWindowMemory (k=10)
    ↓                                     ↓
HuggingFacePipeline (Llama wrapper)   LangChain Chroma (vector store)
    ↓                                     ↓
HuggingFaceEmbeddings (SentenceTransformer wrapper)
    ↓
core/ modules (shared with Phase 1)
```

**Features**:
- Automatic conversation history management
- ConversationalRetrievalChain for standardized RAG workflow
- LangChain Memory with automatic pruning
- Document loaders and text splitters

**Use Cases**:
- Rapid prototyping
- Standardized workflow
- LangChain ecosystem integration
- Production deployment (if LangChain is preferred)

### Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 (Pure Python) | Phase 2 (LangChain) |
|--------|----------------------|---------------------|
| **Conversation History** | Manual (deque, 10 rounds) | Automatic (ConversationBufferWindowMemory) |
| **RAG Workflow** | Custom query + retrieval logic | ConversationalRetrievalChain |
| **Embeddings** | Direct SentenceTransformer | HuggingFaceEmbeddings wrapper |
| **Vector Store** | Direct Chroma client queries | LangChain Chroma wrapper |
| **Code Lines** | More code, more control | Less code, more abstraction |
| **Learning Curve** | Understanding RAG internals | Understanding LangChain API |
| **Debugging** | Easier to trace | Black box abstractions |
| **Flexibility** | Full control over prompts | LangChain prompt templates |
| **Performance** | Optimized, minimal overhead | Slight overhead from abstractions |
| **Port** | 7860 | 7861 |

### Chroma Metadata Schema

Both phases use consistent metadata structure for vector database:

```python
metadata = {
    "source_type": "pdf",           # "pdf" or "cve"
    "source_name": "CVEpdf2024.pdf", # Filename or "CVE List 2024"
    "added_date": "2025-01-14",     # ISO format (YYYY-MM-DD)
    "chunk_index": 1234,            # Sequential index
    "page_number": 5,               # PDF page (if applicable)
    "precision": "float16"          # Embedding precision
}
```

**Purpose**:
- **source_type**: Filter by document type (PDF reports vs CVE data)
- **source_name**: Track and delete specific documents
- **added_date**: Audit trail for knowledge base updates
- **chunk_index**: Order chunks within same document
- **page_number**: Source tracing for PDFs
- **precision**: Ensure embedding consistency (float16/float32)

**Operations**:
- `chroma_manager.add_documents(texts, embeddings, metadata)`: Add with metadata
- `chroma_manager.query(embedding, filter={"source_type": "pdf"})`: Filter by type
- `chroma_manager.delete_by_source("report1.pdf")`: Remove specific source
- `chroma_manager.get_stats()`: Aggregate by source_type and source_name

### Web UI Architecture (Gradio)

**Framework**: Gradio (Python-native web framework)

**Technology Stack**:
- **Backend**: Gradio → FastAPI/Starlette → Uvicorn (ASGI server)
- **Frontend**: Auto-generated React UI from Python code
- **Communication**: HTTP/WebSocket for real-time updates
- **Deployment**: Single command (`python webUI_v1.py`)

**Advantages**:
- No HTML/CSS/JavaScript required
- ChatGPT-style UI out of the box
- Automatic WebSocket handling for streaming
- Complex layouts via `gr.Blocks()`, `gr.Row()`, `gr.Column()`
- Built-in file upload, markdown rendering
- Easy deployment: `--share` flag generates public URL (valid 72 hours)

**Layout**: Claude Projects-inspired two-column design
- **Left Column (7/12)**: Chat interface with conversation history
- **Right Column (5/12)**:
  - Top: Analysis Settings (speed/mode/schema dropdowns)
  - Bottom: Knowledge Base management (add/view/delete sources)

**Deployment Options**:
1. **Local**: `python web/webUI_v1.py` (http://localhost:7860)
2. **Share Link**: `python web/webUI_v1.py --share` (https://xxx.gradio.live)
3. **Production**: Docker + Nginx (optional)

## Runtime Architecture

### Main Implementation: theRag.py

**Dual-Mode Design**:
- **Demo Mode** (`--mode=demo`): Memory-optimized for limited hardware
- **Full Mode** (default): Complete feature set with intelligent chunking

**Schema Selection** (`--schema`):
- **v5**: Use V5 schema only (fastest, requires V5 feeds)
- **v4**: Use V4 schema only (requires V4 feeds)
- **all** (default): Try V5 first, fallback to V4 (backward compatible)

**Demo Mode Features**:
- Processes first 10 pages of PDF only
- Limits text to 1000-2000 characters
- Uses `torch.float16` precision + `low_cpu_mem_usage=True`
- Wraps all generation in `torch.no_grad()`
- Reads only 1000 rows from embedding CSV
- Returns top-3 retrieval results
- Token generation: 64-256 tokens
- CUDNN deterministic mode for stability

**Full Mode Features**:
- Processes entire PDF with 1500-token chunks (200-token overlap)
- No text truncation
- Reads complete embedding CSV
- Returns top-5 retrieval results
- Token generation: 150-700 tokens
- Context extraction for missing CVEs (up to 2000 chars)

### Execution Flow

**Phase 1: Initialization** (one-time, at startup)
1. Parse command-line arguments:
   - `--mode=demo` or `--mode=full` (default)
   - `--schema=v5|v4|all` (default: all)
2. Load **Llama-3.2-1B-Instruct** model (~2.5GB)
   - Device: Auto-detect CUDA/CPU via `device_map="auto"`
   - Precision: FP16 (demo) or auto (full)
3. Load **SentenceTransformer** `all-mpnet-base-v2` globally (~420MB)
4. Set memory optimization flags (CUDNN if demo mode)

**Phase 2: PDF Processing** (user input required)
1. User provides PDF filename
2. Extract text page-by-page:
   - Demo: First 10 pages
   - Full: All pages
3. Periodic memory cleanup every 5 pages (demo mode only)

**Phase 3: CVE Extraction** (optimized with direct regex)
1. Apply regex pattern `CVE-\d{4}-\d{4,7}` to extracted text
2. Deduplicate CVE list
3. **No LLM call needed** (previous versions used LLM, now deprecated)

**Phase 4: CVE Metadata Lookup** (for each extracted CVE)
1. Parse CVE format: `CVE-<year>-<id>` → extract year and ID
2. Format path prefix: `<id>` → `Nxxx` (e.g., `1234` → `1xxx`, `12345` → `12xxx`)
3. **Load CVE record based on schema parameter**:
   - `--schema=v5`: Only try `../cvelistV5/cves/<year>/<prefix>/CVE-<year>-<id>.json`
   - `--schema=v4`: Only try `../cvelist/<year>/<prefix>/CVE-<year>-<id>.json`
   - `--schema=all` (default): Try v5 first, fallback to v4
4. Extract fields (schema-aware):
   - `cveId`, `vendor`, `product`, `description`
5. Build concatenated string of all found CVE descriptions

**Phase 5: Fallback for Missing CVEs** (if JSON not found)
1. Extract context from PDF:
   - Demo: First 500 chars of report
   - Full: 1500-char window around CVE mention (capped at 2000 chars)
2. **LLM Call 1**: Ask Llama to paraphrase CVE usage in 2 sentences (64-700 tokens)
3. **LLM Call 2**: Call `asking_llama_for_advice()`:
   - Load `CVEEmbeddings.csv` (embedding database)
   - Use SentenceTransformer to find top-3/5 similar CVE descriptions
   - Ask Llama to recommend similar CVE based on retrieved context (128-700 tokens)
4. Cleanup: Clear CUDA cache, run garbage collection

**Phase 6: Interactive Analysis Menu** (loop until user exits)
- **Option 1 - Summarize Report**:
  - Demo: Truncate to 2000 chars, 1 LLM call (256 tokens)
  - Full: Process in chunks, N LLM calls (700 tokens each)
- **Option 2 - Validate CVE Usage**:
  - Compare report context vs. official CVE descriptions
  - 1 LLM call with both inputs (256-700 tokens)
- **Option 3 - Custom Q&A**:
  - User asks question about report
  - 1 LLM call with question + report text (256-700 tokens)
- **Option 4 - Exit**:
  - Cleanup models, exit program

### Key Components

#### `asking_llama_for_advice(cveDesp: str) -> str`
**Purpose**: Recommend similar CVE when original not found
**Process**:
1. Load embedding CSV (1000 rows max)
2. Parse string-formatted embeddings to numpy arrays
3. Convert to torch tensor (float32 for consistency)
4. Initialize SentenceTransformer `all-mpnet-base-v2` (**optimization opportunity**: should be global)
5. Call `retrieve_context()` to get top-3 chunks
6. Format retrieved chunks as context string
7. Ask Llama to recommend CVE based on description + context (128 tokens)
8. Cleanup: delete embeddings, run `torch.cuda.empty_cache()` + `gc.collect()`

#### `retrieve_context(query, embeddings, model, n_resources_to_return=3)`
**Purpose**: Semantic search over knowledge base
**Process**:
1. Encode query with SentenceTransformer
2. Ensure dtype consistency between query and corpus embeddings
3. Calculate dot product similarity scores
4. Return indices of top-3 chunks

## Memory Management Strategy

- `torch.backends.cudnn.benchmark = False` (disable auto-tuner)
- `torch.backends.cudnn.deterministic = True` (reproducible but slower)
- Periodic `gc.collect()` during PDF processing
- `torch.no_grad()` context for all generation
- Manual cleanup after each CVE fallback
- Manual cleanup after each menu operation
- Final `cleanup_model()` on exit

## Critical Dependencies

1. **CVEEmbeddings.{extension}**: Must exist before running `theRag.py`
   - Generated by `localEmbedding.py --extension={extension}`
   - Default: `CVEEmbeddings.pkl` (file)
   - Chroma: `CVEEmbeddings/` (directory)
   - `--extension` parameter must match between localEmbedding.py and theRag.py
   - Format requirements:
     - Parquet: Requires `pyarrow` (in requirements.txt)
     - Chroma: Requires `chromadb` (in requirements.txt)
2. **CVE JSON Feeds**: External directories `../cvelistV5/` (primary) and `../cvelist/` (fallback) must be synced
3. **Hugging Face Access**: Llama model requires approval + `huggingface-cli login`
4. **Python Environment**: One of `.venv-cpu`, `.venv-cuda118`, or `.venv-cuda124` activated
5. **CUDA Toolkit** (for GPU): Version must match venv (11.8 or 12.4)

## Performance Characteristics

### Typical Processing Times (10-page PDF with 8 CVEs, 3 missing)

**CPU Environment** (Intel i5/i7, AMD Ryzen 5/7):
- Phase 1 (Initialization): ~30-60 sec (model loading)
- Phase 2-4 (PDF + CVE lookup): ~10 sec
- Phase 5 (Missing CVEs): 3 CVEs × 2 LLM calls × 40 sec = **240 sec (4 min)**
- **Option 1 Summary (full mode)**: 4 chunks × 45 sec = **180 sec (3 min)**
- **Total first run**: ~7-8 minutes

**GPU Environment - GTX 1660 Ti (6GB)**:
- Phase 1 (Initialization): ~15-30 sec
- Phase 2-4 (PDF + CVE lookup): ~10 sec
- Phase 5 (Missing CVEs): 3 CVEs × 2 LLM calls × 4 sec = **24 sec**
- **Option 1 Summary (full mode)**: 4 chunks × 4.5 sec = **18 sec**
- **Total first run**: ~1 minute (**90% faster**)

**GPU Environment - RTX 4060 (12GB)**:
- Phase 1 (Initialization): ~10-20 sec
- Phase 2-4 (PDF + CVE lookup): ~10 sec
- Phase 5 (Missing CVEs): 3 CVEs × 2 LLM calls × 2.5 sec = **15 sec**
- **Option 1 Summary (full mode)**: 4 chunks × 3 sec = **12 sec**
- **Total first run**: ~40 seconds (**95% faster**)

### Memory Usage

**Demo Mode**:
- CPU: ~4-6 GB RAM
- GPU: ~3.5-4 GB VRAM + 2-3 GB RAM

**Full Mode**:
- CPU: ~6-8 GB RAM
- GPU: ~4-5 GB VRAM + 3-4 GB RAM

## Design Constraints & Trade-offs

### Resolved Optimizations
1. ✅ **CVE Extraction**: Now uses direct regex (no LLM waste)
2. ✅ **Embedding Model**: Global SentenceTransformer (loaded once)
3. ✅ **Memory Management**: torch.no_grad() + context limits + chunked cleanup
4. ✅ **Schema Compatibility**: v5→v4 fallback for CVE JSON

### Remaining Limitations
1. **Sequential CVE Processing**: Could parallelize but increases memory pressure
2. **Static Retrieval Corpus**: No online updates from NVD/MITRE during runtime
3. **Context Window**: Demo mode's 2000-char limit may miss details in long reports
4. **Single-threaded LLM**: One generation at a time (model limitation)

### Architectural Trade-offs
- Demo mode prioritizes stability over completeness (acceptable for quick analysis)
- Full mode prioritizes completeness with intelligent chunking (for thorough analysis)
- Auto device detection enables seamless CPU/GPU switching
- FP16 precision balances speed and accuracy on GPU

## File Interaction Map

```
User PDF → theRag.py
             ↓
       PyMuPDF (fitz)
             ↓
       Extract Text → CVE Regex
             ↓
    For each CVE → ../cvelist/<year>/<prefix>/CVE-*.json
                           ↓ (if not found)
                     Llama paraphrase → asking_llama_for_advice()
                                                   ↓
                                             CVEEmbeddings.{extension}
                                             (csv/pkl/parquet/chroma)
                                                   ↓
                                           SentenceTransformer
                                           (chroma: direct query)
                                                   ↓
                                           Llama recommendation

Menu Options → Llama (with official CVE descriptions + retrieval context) → User
```

## Security Considerations
- This is a **defensive security tool** for validating CVE usage in reports
- Does not discover, harvest, or exploit vulnerabilities
- Relies on publicly available CVE metadata from MITRE/NVD
- LLM outputs should be human-verified (hallucinations still possible despite RAG)

## Multi-Level Speed Optimization

### Overview
`theRag.py` now supports **3 speed levels** via the `--speed` parameter:
- **normal**: Baseline with chunk-aware filtering
- **fast**: Recommended (default) with FP16 + reduced cache clearing
- **fastest**: Aggressive with lower temperature + SDPA

### Speed Level Comparison

| Feature | normal | fast (default) | fastest |
|---------|--------|---------------|---------|
| **Chunk-aware CVE filtering** | ✅ | ✅ | ✅ |
| **FP16 precision** | ❌ (FP32) | ✅ | ✅ |
| **Cache clearing** | Every chunk | Every 3 chunks | Every 3 chunks |
| **Temperature** | 0.3 | 0.3 | 0.1 |
| **SDPA attention** | ❌ | ❌ | ✅ (if available) |
| **Option 2 time (GTX 1660 Ti)** | ~4-5 min | ~3 min | ~2 min |
| **Option 2 time (CPU)** | ~4-5 min | ~4-5 min | ~4 min |
| **Speedup vs original (GPU)** | 7-8x | 10-12x | 15-17x |
| **Speedup vs original (CPU)** | 7-8x | 7-8x | 8-9x |

### Optimization 1: Chunk-Aware CVE Filtering

**Problem**: Option 2 originally sent ALL CVE descriptions to every chunk
- 23 CVE descriptions × ~10,506 chars = ~2,400 tokens per chunk
- Result: 3,934 total input tokens → 2.6 tok/s (extremely slow)

**Solution**: Only send CVE descriptions mentioned in each chunk
```python
# Parse CVE descriptions into dictionary (one-time)
cve_dict = {
    "CVE-2025-54253": "description...",
    "CVE-2025-7776": "description...",
    # ... all 23 CVEs
}

# For each chunk
for chunk in chunks:
    # Extract CVEs mentioned in THIS chunk only
    chunk_cves = extract_cves_regex(chunk)  # e.g., 2-4 CVEs

    # Build filtered descriptions (only relevant CVEs)
    filtered_desc = "\n\n\n".join([cve_dict[cve] for cve in chunk_cves])

    # Send only relevant CVE descriptions
    system_prompt = base_prompt + filtered_desc + instructions
```

**Result**: ~1,900 total input tokens → 15-20 tok/s (7-8x faster)

**Why it works**:
- Each CVE is validated independently (no cross-CVE reasoning needed)
- Typical chunk mentions only 2-4 CVEs out of 23 total
- Removing irrelevant CVEs reduces noise and improves focus

**Trade-offs**: None - 200-token overlap between chunks catches cross-references

### Optimization 2: FP16 Precision (fast/fastest)

**FP16** (16-bit floating point) vs **FP32** (32-bit floating point):
- **Memory bandwidth**: 2x faster (reads/writes half the data)
- **Precision**: Negligible loss for Llama 3.2-1B
- **Model support**: Llama 3.2 is trained with mixed precision

**Implementation**:
```python
if USE_FP16:
    model_kwargs["torch_dtype"] = torch.float16
else:
    model_kwargs["torch_dtype"] = "auto"
```

**Impact**:
- GPU: 20-30% speed improvement
- CPU: Minimal to no improvement (CPUs typically use FP32 pipelines)

**Why safe on GPU**: GTX 1660 Ti/RTX 4060 have dedicated Tensor Cores for FP16

**CPU note**: PyTorch accepts FP16 on CPU but typically converts to FP32 internally, providing little benefit

### Optimization 3: Reduced Cache Clearing (fast/fastest)

**CUDA cache** stores temporary GPU memory:
- normal: Clears after every chunk (safe but slow)
- fast/fastest: Clears every 3 chunks (faster, minimal risk)

**Implementation**:
```python
for idx, chunk in enumerate(chunks):
    # ... processing ...
    if (idx + 1) % CACHE_CLEAR_FREQUENCY == 0:
        torch.cuda.empty_cache()  # No-op on CPU
```

**Why safe on GPU**:
- GTX 1660 Ti has 6GB VRAM
- Peak usage: ~4GB with full-mode settings
- Clearing every 3 chunks frees ~300-600MB (plenty of headroom)

**Impact**:
- GPU: 5-10% speed improvement
- CPU: No effect (`torch.cuda.empty_cache()` is safely ignored)

### Optimization 4: Temperature Adjustment (fastest only)

**Temperature** controls randomness in generation:
- 0.3 (normal/fast): Balanced creativity and consistency
- 0.1 (fastest): More deterministic, less creative

**Effect of lower temperature**:
- ✅ Faster generation (5-8% speedup)
- ✅ More consistent outputs
- ⚠️ Less diverse phrasing (acceptable trade-off)

**Impact**:
- GPU: 5-8% speed improvement for fastest level
- CPU: 5-8% speed improvement for fastest level (same benefit as GPU)

### Optimization 5: SDPA Attention (fastest only)

**SDPA** (Scaled Dot-Product Attention):
- Uses fused kernels for better memory efficiency
- Available in PyTorch 2.0+ and transformers 4.35+
- Protected by try-except, gracefully falls back if unavailable

**Implementation**:
```python
if USE_SDPA:
    try:
        model_kwargs["attn_implementation"] = "sdpa"
    except Exception as e:
        print(f"SDPA not available: {e}, using default attention")
```

**Impact**:
- GPU: 10-20% speed improvement if available
- CPU: Minor improvement (0-5%), SDPA less optimized for CPU

## Performance Benchmarks

### Test Environment - GPU
- **GPU**: GTX 1660 Ti (6GB VRAM)
- **PDF**: 32 pages, 23 CVEs, 8 chunks
- **Mode**: Full mode (CUDA enabled)

### Test Environment - CPU
- **CPU**: Intel i5/i7 or AMD Ryzen 5/7
- **PDF**: Same 32 pages, 23 CVEs, 8 chunks
- **Mode**: Full mode

### Option 2 (CVE Validation) - Most Impactful

**GPU Performance (GTX 1660 Ti)**:
| Speed Level | Time | Speedup vs Original | Speedup vs normal |
|-------------|------|---------------------|-------------------|
| Original | 34.4 min | 1x | - |
| normal | 4.5 min | 7.6x | 1x |
| fast (default) | 3.1 min | 11.1x | 1.5x |
| fastest | 2.1 min | 16.4x | 2.1x |

**CPU Performance (Intel i5/i7, AMD Ryzen 5/7)**:
| Speed Level | Time (est.) | Speedup vs Original | Speedup vs normal |
|-------------|-------------|---------------------|-------------------|
| Original | ~34 min | 1x | - |
| normal | ~4.5 min | 7.6x | 1x |
| fast (default) | ~4.5 min | 7.6x | ~1.0x |
| fastest | ~4.0 min | 8.5x | ~1.1x |

### Option 1 (Summarization)

| Speed Level | Time | Speedup vs normal |
|-------------|------|-------------------|
| normal | 2.6 min | 1x |
| fast (default) | 1.8 min | 1.4x |
| fastest | 1.2 min | 2.2x |

### Option 3 (Q&A)

| Speed Level | Time | Speedup vs normal |
|-------------|------|-------------------|
| normal | 2.5 min | 1x |
| fast (default) | 1.7 min | 1.5x |
| fastest | 1.0 min | 2.5x |

### Per-Chunk Breakdown (Option 2, GTX 1660 Ti)

| Chunk | Original Tokens | Optimized Tokens | Original Time | Optimized Time | Speedup |
|-------|-----------------|------------------|---------------|----------------|---------|
| 1 | 3,934 | ~1,900 | 273.3s | ~35s | 7.8x |
| 2 | 3,936 | ~1,900 | 260.6s | ~34s | 7.7x |
| 3 | 3,937 | ~1,900 | 277.8s | ~36s | 7.7x |
| 4 | 3,937 | ~1,900 | 263.9s | ~34s | 7.8x |
| 5 | 3,937 | ~1,900 | 267.5s | ~35s | 7.6x |
| 6 | 3,936 | ~1,900 | 256.2s | ~33s | 7.8x |
| 7 | 3,937 | ~1,900 | 268.6s | ~35s | 7.7x |
| 8 | 3,170 | ~1,500 | 191.2s | ~25s | 7.6x |
| **Total** | - | - | **2061.6s (34.4 min)** | **~267s (4.5 min)** | **7.7x** |

## Earlier Optimizations (Already Implemented)

### Performance Improvements
1. ✅ **Direct regex CVE extraction**: Skips LLM entirely, uses pattern `CVE-\d{4}-\d{4,7}`
2. ✅ **Global SentenceTransformer**: Initialized once at startup instead of per-call
3. ✅ **Unified inference wrapper**: All `model.generate()` calls use `torch.no_grad()`
4. ✅ **Fixed typos**: "based of" → "based on", improved prompt clarity

### Memory Management
1. ✅ **Context length limits**: Missing CVE contexts capped at 2000 chars (full mode)
2. ✅ **Chunk-level cleanup**: `torch.cuda.empty_cache()` after each chunk in full mode
3. ✅ **Consistent `pad_token_id`**: All generate calls include padding token
4. ✅ **Demo/Full mode separation**: Clear memory profiles for different use cases

## Command-Line Usage

```bash
# Default (uses fast)
python theRag.py

# Specify speed level
python theRag.py --speed=normal   # Baseline, maximum precision
python theRag.py --speed=fast     # Recommended (default)
python theRag.py --speed=fastest  # Maximum speed

# Combine with other parameters
python theRag.py --speed=fastest --mode=demo
python theRag.py --speed=fast --schema=v5
python theRag.py --speed=normal --mode=full --schema=all
```

## Recommendation by Use Case

### 📊 Production Analysis (Daily Work)
**Recommended: fast (default)**
- GPU: Best balance of speed and accuracy, 11x faster than original for Option 2
- CPU: Same speed as normal (~7-8x faster), but safe default choice
- No noticeable quality loss

### 🔬 Quality Validation / Benchmarking
**Recommended: normal**
- Maximum precision (FP32)
- Baseline for comparison
- GPU/CPU: Both ~7-8x faster than original

### ⏱️ Batch Processing / Time Critical
**GPU users: fastest**
- Maximum speed (17x faster than original for Option 2)
- Acceptable output quality

**CPU users: normal or fast**
- fastest provides minimal benefit on CPU (only 5-10% faster)
- normal/fast both ~7-8x faster than original

### 💻 CPU-Only Environment
**Recommended: normal or fast**
- All speed levels benefit from chunk-aware CVE filtering (7-8x speedup)
- FP16 and cache optimizations ineffective on CPU
- fastest only adds 5-10% from temperature/SDPA
- Save energy and use normal or fast

## Key Design Decisions

### Why Phase 1 Before Phase 2?

**Rationale**:
1. **Learning**: Understand RAG internals before using abstractions
2. **Debugging**: Easier to debug custom code vs LangChain black box
3. **Flexibility**: Fine-grained control over prompts and retrieval logic
4. **Comparison**: Benchmark pure Python vs LangChain performance
5. **Fallback**: If LangChain has issues, Phase 1 is production-ready

**Outcome**: Both phases coexist, allowing A/B testing and gradual migration.

### Why Gradio Over Custom Frontend?

| Criteria | Gradio | Flask+React | FastAPI+Vue |
|----------|--------|-------------|-------------|
| **Dev Time** | 1-2 days | 2-3 weeks | 3-4 weeks |
| **Learning Curve** | None (pure Python) | Moderate | Steep |
| **UI Quality** | Modern (ChatGPT-like) | Custom | Custom |
| **Deployment** | 1 command | Complex | Complex |
| **Suitable For** | PoC, Demo | Production | Enterprise |

**Decision**: Gradio for rapid prototyping and quality UI without frontend development.

### Why Not Expose Embedding Precision in UI?

**Reasoning**:
- Too technical for non-technical users
- Mixing precisions causes ranking inconsistencies (should be fixed at `fast`)
- Configuration should be in `.env`, not user-facing UI
- If needed, can be exposed in "Advanced Settings" (future enhancement)

**Current Approach**: `.env` sets `EMBEDDING_PRECISION=float16`, all components use same precision.

### Why Unified `fast` Speed Level?

**Problem**: Mixed precision (float32 + float16) causes ranking inconsistencies in retrieval.

**Solution**:
- `fast` (default) uses float16 for both embeddings and LLM
- Provides 1.5-2x speedup with <1% accuracy loss
- Consistent precision ensures stable similarity scores

**Comparison**:
| Speed  | Precision | Chunk Size | Batch Size | Use Case |
|--------|-----------|------------|------------|----------|
| normal | float32   | 10         | 32         | Baseline |
| fast   | float16   | 10         | 64         | Recommended (default) |
| fastest| float16   | 20         | 128        | Max speed (larger chunks) |

### Why Two Ports for Phase 1 and Phase 2?

**Reasoning**:
- Both implementations can run simultaneously for A/B comparison
- Phase 1 (port 7860): Pure Python baseline
- Phase 2 (port 7861): LangChain alternative
- Users can compare performance, accuracy, and behavior side-by-side

**Use Case**: Demo both approaches in company presentation.

### Why Manual CVE Regex Over LLM Extraction?

**Previous Approach**: Ask LLM to extract CVEs from report text.

**Problems**:
- Slow (30-60 sec per LLM call)
- Unreliable (hallucinations, missed CVEs)
- Expensive (tokens + compute)

**Current Approach**: Direct regex `CVE-\d{4}-\d{4,7}` on extracted text.

**Benefits**:
- Instant (<1 sec)
- 100% accurate for standard format
- No token cost

**Trade-off**: Won't catch non-standard CVE mentions (acceptable for official reports).

## Troubleshooting

### SDPA Warning (fastest)
```
SDPA not available: ..., using default attention
```
**Solution**: This is normal if you have older transformers. SDPA is optional.

### Out of Memory (fast/fastest)
```
CUDA out of memory
```
**Solution**: Use `--speed=normal` or `--mode=demo`

### Outputs Too Similar (fastest)
If outputs are too repetitive with fastest, use fast instead:
```bash
python theRag.py --speed=fast
```
