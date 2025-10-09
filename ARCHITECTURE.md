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

- **RTX 3060 (12GB VRAM)** or better:
  - CUDA Toolkit: 12.1 (recommended)
  - Environment: `venv-cuda121`
  - Speed: ~80-120 tokens/sec (20-40x faster than CPU)
  - Suitable for: Llama-3.2-1B/3B/7B models

- **RTX 4060+ / Professional GPUs**:
  - CUDA Toolkit: 12.1+
  - Environment: `venv-cuda121`
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
- **Process**:
  1. Extract text from PDF using PyMuPDF (`fitz`)
  2. Tokenize with spaCy's sentencizer into 10-sentence chunks
  3. Generate embeddings using `all-mpnet-base-v2` SentenceTransformer (768 dimensions)
  4. Save to CSV (e.g., `test6.csv`) with columns: `sentence_chunk`, `embedding`
- **Purpose**: Creates retrieval corpus for contextual grounding and CVE recommendations
- **Run once**: Before first use of theRag.py

### Official CVE Metadata
- **Source**: MITRE/NVD JSON feeds in `../cvelistV5/cves/2024` (v5 schema, primary) and `../cvelist/2024` (v4 schema, fallback)
- **Structure**: Files organized as `<year>/<prefix-xxx>/CVE-<year>-<id>.json`
  - v5 schema: `cveMetadata.cveId`, `containers.cna.affected`, `containers.cna.descriptions`
  - v4 schema: `CVE_data_meta.ID`, `affects.vendor.vendor_data`, `description.description_data`
- **Optional Tools**: `extractCVE.py` flattens JSONs to `CVEDescription2024.txt` (for human reference only, not used by RAG)

## Runtime Architecture

### Main Implementation: theRag.py

**Dual-Mode Design**:
- **Demo Mode** (`--mode=demo`): Memory-optimized for limited hardware
- **Full Mode** (default): Complete feature set with intelligent chunking

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
1. Parse command-line arguments (`--mode=demo` or default full)
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
3. **Try v5 schema first**: `../cvelistV5/cves/<year>/<prefix>/CVE-<year>-<id>.json`
4. **Fallback to v4 schema**: `../cvelist/<year>/<prefix>/CVE-<year>-<id>.json`
5. Extract fields (schema-aware):
   - `cveId`, `vendor`, `product`, `description`
6. Build concatenated string of all found CVE descriptions

**Phase 5: Fallback for Missing CVEs** (if JSON not found)
1. Extract context from PDF:
   - Demo: First 500 chars of report
   - Full: 1500-char window around CVE mention (capped at 2000 chars)
2. **LLM Call 1**: Ask Llama to paraphrase CVE usage in 2 sentences (64-700 tokens)
3. **LLM Call 2**: Call `asking_llama_for_advice()`:
   - Load `test6.csv` (embedding database)
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

1. **test6.csv**: Must exist before running `theRag.py`; generated by `localEmbedding.py`
2. **CVE JSON Feeds**: External directories `../cvelistV5/` (primary) and `../cvelist/` (fallback) must be synced
3. **Hugging Face Access**: Llama model requires approval + `huggingface-cli login`
4. **Python Environment**: One of `venv-cpu`, `venv-cuda118`, or `venv-cuda121` activated
5. **CUDA Toolkit** (for GPU): Version must match venv (11.8 or 12.1)

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

**GPU Environment - RTX 3060 (12GB)**:
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
                                             test6.csv (embeddings)
                                                   ↓
                                           SentenceTransformer
                                                   ↓
                                           Llama recommendation

Menu Options → Llama (with official CVE descriptions + retrieval context) → User
```

## Security Considerations
- This is a **defensive security tool** for validating CVE usage in reports
- Does not discover, harvest, or exploit vulnerabilities
- Relies on publicly available CVE metadata from MITRE/NVD
- LLM outputs should be human-verified (hallucinations still possible despite RAG)

## Implemented Optimizations

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

## Future Optimization Targets
1. Add input validation for CSV file existence and format
2. Parallelize CVE lookups (if memory permits)
3. Add progress bars for long-running operations
4. Consider batch processing for multiple PDFs
