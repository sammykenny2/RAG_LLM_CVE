# Feature Planning Document - Web UI & Knowledge Base Enhancement

**Branch**: `feature/enhanced-knowledge-base`
**Created**: 2025-01-14
**Status**: Planning Phase

---

## 1. Project Goals

### Primary Objectives
1. **Web-based Chat Interface**: Claude Projects-style UI for RAG interactions
2. **Incremental Knowledge Base**: Add PDFs and CVE data to existing embeddings
3. **Report Validation**: Integrate existing theRag.py functionality into web UI
4. **Knowledge Base Management**: View, add, and delete sources

### Demo Requirements
- **Audience**: Company-wide (technical and non-technical)
- **Core Features**:
  - Simple conversational AI (like ChatGPT)
  - CVE validation demonstration (Summary/Validate/Q&A)
  - Knowledge base expansion capability

---

## 2. Architecture Overview

### 2.1 Phased Implementation Strategy

#### Phase 1: Pure Python Implementation (Week 1)
```
webUI_v1.py                 # Gradio interface (pure Python)
├─ Uses core/ modules       # Shared utilities
└─ Uses rag/pure_python.py  # Manual memory management
```

**Features**:
- ✅ Manual conversation history (last 10 rounds)
- ✅ Direct Chroma queries
- ✅ theRag.py logic extraction to reusable functions
- ✅ Gradio-based UI (no LangChain)

#### Phase 2: LangChain Implementation (Week 2)
```
webUI_v2.py                    # Gradio interface (LangChain)
├─ Uses core/ modules          # Same shared utilities
└─ Uses rag/langchain_impl.py  # LangChain chains
```

**Features**:
- ✅ LangChain Memory management
- ✅ ConversationalRetrievalChain
- ✅ LangChain Retrievers with metadata filtering
- ✅ Document Loaders and Text Splitters

### 2.2 Module Structure

```
RAG_LLM_CVE/
├── core/                        # Shared modules (Phase 1 & 2)
│   ├── __init__.py
│   ├── models.py                # Llama model loading
│   ├── embeddings.py            # SentenceTransformer operations
│   ├── chroma_manager.py        # Chroma CRUD operations
│   ├── cve_lookup.py            # CVE JSON file queries
│   └── pdf_processor.py         # PDF text extraction
│
├── rag/                         # RAG implementations
│   ├── __init__.py
│   ├── pure_python.py           # Phase 1: Manual implementation
│   └── langchain_impl.py        # Phase 2: LangChain wrapper
│
├── cli/                         # Command-line tools
│   ├── theRag.py                # Original CLI (uses core/)
│   ├── theRag_lc.py             # LangChain CLI (Phase 2, optional)
│   ├── localEmbedding.py        # Initial embedding generation
│   └── addToEmbeddings.py       # Incremental additions (NEW)
│
├── web/                         # Web interfaces
│   ├── webUI_v1.py              # Phase 1: Pure Python
│   └── webUI_v2.py              # Phase 2: LangChain
│
├── config.py                    # Unified configuration (uses .env)
├── .env.example                 # Environment template
├── .env                         # Local config (gitignored)
├── requirements.txt             # Base dependencies
└── requirements_langchain.txt   # Phase 2 dependencies
```

---

## 3. UI Design (Claude Projects Style)

### 3.1 Layout Structure

```
┌──────────────────────────────────────────────────────────────┐
│  🔒 CVE RAG Assistant                          [Minimize]     │
├──────────────────────────────┬───────────────────────────────┤
│ Left Column (7/12)           │ Right Column (5/12)            │
│                              │                                │
│ ┌──────────────────────────┐ │ ┌─────────────────────────┐   │
│ │ 💬 Chat History          │ │ │ ⚙️ Instructions          │   │
│ │                          │ │ │ ─────────────────       │   │
│ │ [Conversation Display]   │ │ │ 📊 Analysis Settings    │   │
│ │                          │ │ │                         │   │
│ └──────────────────────────┘ │ │ Speed:  [fastest ▼]     │   │
│                              │ │ Mode:   [full ▼]        │   │
│ ┌──────────────────────────┐ │ │ Schema: [all ▼]         │   │
│ │ Your message:            │ │ │                         │   │
│ │ [________________]       │ │ └─────────────────────────┘   │
│ └──────────────────────────┘ │                                │
│                              │ ───────────────────────────   │
│ [📎] [⚙️] [🕐]      [Send]   │                                │
│  ↑                           │ ┌─────────────────────────┐   │
│  └─ Upload PDF to Validate   │ │ 📚 Knowledge Base       │   │
│                              │ │ ─────────────────       │   │
│                              │ │ [+ Add Files]           │   │
│                              │ │  ↑                      │   │
│                              │ │  └─ Add to RAG          │   │
│                              │ │                         │   │
│                              │ │ Current Sources:        │   │
│                              │ │ ┌─────────────────────┐ │   │
│                              │ │ │☑ CVEpdf2024.pdf [🗑️]│ │   │
│                              │ │ │  7,261 chunks        │ │   │
│                              │ │ │☑ report1.pdf   [🗑️]│ │   │
│                              │ │ │  856 chunks          │ │   │
│                              │ │ └─────────────────────┘ │   │
│                              │ │ [Delete Selected]       │   │
│                              │ │ Total: 8,117 chunks     │   │
│                              │ └─────────────────────────┘   │
└──────────────────────────────┴───────────────────────────────┘
```

### 3.2 User Interaction Flows

#### Flow 1: Ask Questions (RAG Query)
```
User: Types "What is CVE-2024-1234?"
  ↓
System: Retrieves from Knowledge Base (right panel sources)
  ↓
AI: Responds with answer + source citation
  ↓
Chat history updates (maintains 10 rounds)
```

#### Flow 2: Upload Report for Validation
```
User: Clicks [📎] → Uploads report.pdf
  ↓
AI: "📄 Received report.pdf. What would you like to do?"
  ↓
Displays action buttons:
┌─────────────────────────────────┐
│ [📝 Summarize Report]            │
│ [✅ Validate CVE Usage]          │
│ [❓ Q&A Mode]                     │
│ [🧠 Add to Knowledge Base]      │
└─────────────────────────────────┘
  ↓
User selects action
  ↓
System executes (using Analysis Settings from right panel)
```

#### Flow 3: Expand Knowledge Base
```
User: Clicks [+ Add Files] in right panel → Uploads new_report.pdf
  ↓
System: Incremental embedding (using addToEmbeddings.py logic)
  ↓
AI: "✅ Added new_report.pdf to Knowledge Base (1,234 chunks)"
  ↓
Right panel source list updates automatically
```

#### Flow 4: Delete Source
```
User: Checks ☑ CVEpdf2024.pdf → Clicks [Delete Selected]
  ↓
System: Confirms "Delete CVEpdf2024.pdf (7,261 chunks)?"
  ↓
User confirms
  ↓
System: Removes from Chroma (metadata filter: source_name)
  ↓
Right panel updates
```

---

## 4. Technical Specifications

### 4.1 Configuration Management (.env)

**Files**:
- `.env.example` - Template (committed to git)
- `.env` - Local configuration (gitignored)
- `config.py` - Python configuration loader

**Key Configuration Parameters**:
```bash
# Paths
CHROMA_DB_PATH=./CVEEmbeddings.chroma
CVE_V5_PATH=../cvelistV5/cves
CVE_V4_PATH=../cvelist

# Models
LLAMA_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
EMBEDDING_MODEL=all-mpnet-base-v2

# Defaults
DEFAULT_SPEED=fast              # normal, fast, fastest
DEFAULT_MODE=full               # demo, full
DEFAULT_SCHEMA=all              # v5, v4, all
DEFAULT_EMBEDDING_FORMAT=chroma # csv, pkl, parquet, chroma

# Web UI
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=False

# Advanced
CONVERSATION_HISTORY_LENGTH=10
RETRIEVAL_TOP_K=5
```

### 4.2 Embedding Strategy

**Unified Precision**: `fast` (float16)

**Rationale**:
- ❌ Mixed precision (float32 + float16) causes ranking inconsistencies
- ✅ float16 provides 1.5-2x speedup with <1% accuracy loss
- ✅ Consistent precision ensures stable similarity scores
- ✅ `fast` balances speed and quality (chunk_size=10, batch_size=64)

**Comparison**:
| Speed  | Precision | Chunk Size | Batch Size | Use Case |
|--------|-----------|------------|------------|----------|
| normal | float32   | 10         | 32         | Baseline |
| fast   | float16   | 10         | 64         | ✅ Recommended |
| fastest| float16   | 20         | 128        | Max speed (larger chunks) |

### 4.3 Chroma Metadata Schema

```python
metadata = {
    "source_type": "pdf",           # "pdf" or "cve"
    "source_name": "CVEpdf2024.pdf", # Filename or "CVE List 2024"
    "added_date": "2025-01-14",     # ISO format
    "chunk_index": 1234,            # Sequential index
    "page_number": 5,               # PDF page (if applicable)
    "precision": "float16"          # Embedding precision
}
```

### 4.4 Frontend Technology Stack

**Framework**: Gradio (Python-native)

**Advantages**:
- ✅ No HTML/CSS/JavaScript required
- ✅ Beautiful ChatGPT-style UI out of the box
- ✅ Automatic WebSocket handling for real-time updates
- ✅ Complex layouts via `gr.Blocks()`, `gr.Row()`, `gr.Column()`
- ✅ Built-in file upload, markdown rendering, streaming
- ✅ Easy deployment: `python webUI_v1.py --share` (generates public URL)

**Deployment Options**:
1. **Local**: `python webUI_v1.py`
2. **Share Link**: `demo.launch(share=True)` → https://xxx.gradio.live (7 days)
3. **Production**: Docker + Nginx (optional)

---

## 5. Implementation Tasks

### 5.1 Phase 1: Pure Python (Priority)

#### Core Modules (Shared)
- [ ] `core/models.py` - Llama model loading wrapper
- [ ] `core/embeddings.py` - SentenceTransformer wrapper
- [ ] `core/chroma_manager.py` - Chroma CRUD operations
  - [ ] `add_documents()` - Incremental addition with metadata
  - [ ] `query()` - Semantic search with filtering
  - [ ] `get_stats()` - Source statistics (for right panel)
  - [ ] `delete_by_source()` - Remove by source_name
- [ ] `core/cve_lookup.py` - CVE JSON file parser (extract from theRag.py)
- [ ] `core/pdf_processor.py` - PDF text extraction (PyMuPDF)

#### RAG Implementation
- [ ] `rag/pure_python.py` - Manual RAG with history management
  - [ ] `query()` - RAG query with 10-round history
  - [ ] `validate_report()` - theRag.py Option 2 logic
  - [ ] `summarize_report()` - theRag.py Option 1 logic
  - [ ] `qa_mode()` - theRag.py Option 3 logic

#### CLI Tools
- [ ] Refactor `theRag.py` to use `core/` and `rag/pure_python.py`
- [ ] Implement `addToEmbeddings.py`
  - [ ] `--source=pdf --files=a.pdf,b.pdf` - Add multiple PDFs
  - [ ] `--source=cve --year=2024 --month=1-6` - Add CVE data

#### Web UI
- [ ] `web/webUI_v1.py` - Gradio interface
  - [ ] Left panel: Chat interface with history display
  - [ ] Left panel: Upload button (asks for action)
  - [ ] Right panel: Instructions (Analysis Settings)
  - [ ] Right panel: Knowledge Base (source list + add/delete)
  - [ ] Action handler: Summarize/Validate/Q&A/Add to KB
  - [ ] Real-time source statistics update

#### Configuration
- [ ] `.env.example` - Configuration template
- [ ] `config.py` - Unified config loader with validation
- [ ] Update `.gitignore` - Add `.env`

### 5.2 Phase 2: LangChain (Future)

- [ ] `requirements_langchain.txt` - LangChain dependencies
- [ ] `rag/langchain_impl.py` - LangChain-based RAG
  - [ ] Use `ConversationalRetrievalChain`
  - [ ] Use `ConversationBufferWindowMemory` (k=10)
  - [ ] Use `Chroma.as_retriever()` with metadata filtering
  - [ ] Use `PyPDFLoader` and `RecursiveCharacterTextSplitter`
- [ ] `web/webUI_v2.py` - LangChain version of web UI
- [ ] `cli/theRag_lc.py` - LangChain CLI (optional)
- [ ] Compare performance and accuracy: Phase 1 vs Phase 2

### 5.3 Documentation
- [ ] Update `CLAUDE.md` - Add web UI usage instructions
- [ ] Update `ARCHITECTURE.md` - Add module structure and UI architecture
- [ ] Update `PROGRESS.md` - Document new features
- [ ] Create `WEB_UI_GUIDE.md` - Detailed user guide with screenshots

---

## 6. Key Design Decisions

### 6.1 Why Phase 1 Before Phase 2?

**Rationale**:
1. **Learning**: Understand RAG internals before abstraction
2. **Debugging**: Easier to debug custom code vs LangChain black box
3. **Flexibility**: Fine-grained control over prompts and retrieval
4. **Comparison**: Benchmark pure Python vs LangChain performance
5. **Fallback**: If LangChain has issues, Phase 1 is production-ready

### 6.2 Why Gradio Over Custom Frontend?

| Criteria | Gradio | Flask+React | FastAPI+Vue |
|----------|--------|-------------|-------------|
| **Dev Time** | 1-2 days | 2-3 weeks | 3-4 weeks |
| **Learning Curve** | None (pure Python) | Moderate | Steep |
| **UI Quality** | Modern (ChatGPT-like) | Custom | Custom |
| **Deployment** | 1 command | Complex | Complex |
| **Suitable For** | ✅ PoC, Demo | Production | Enterprise |

**Decision**: Gradio for PoC speed and quality.

### 6.3 Why Not Show Embedding Precision in UI?

**Reasoning**:
- ❌ Too technical for non-technical users
- ❌ Mixing precisions causes problems (should be fixed at `fast`)
- ✅ Configuration should be in `.env`, not user-facing UI
- ✅ If needed, can be exposed in "Advanced Settings" (future)

### 6.4 Instructions Panel Configuration

**Option A** (Current Plan): Only Analysis Settings
```
⚙️ Instructions
─────────────
Speed:  [fastest ▼]
Mode:   [full ▼]
Schema: [all ▼]
```

**Option B** (Alternative): Split into Two Sections
```
⚙️ Instructions
─────────────
📊 Analysis Settings (for validation)
  Speed:  [fastest ▼]
  Mode:   [full ▼]
  Schema: [all ▼]

📚 Embedding Settings (for knowledge base)
  Speed:  [fast (fixed)]
  Format: [chroma (fixed)]
```

**Decision**: Start with Option A (simpler), add Option B if user requests.

---

## 7. Testing Strategy

### 7.1 Unit Tests (Optional for PoC)
- `core/chroma_manager.py` - Add, query, delete operations
- `core/cve_lookup.py` - V4/V5 schema parsing

### 7.2 Integration Tests
- [ ] Upload PDF → Add to KB → Query about it → Verify response
- [ ] Upload report → Validate CVE → Check results match CLI
- [ ] Delete source → Verify removed from Chroma and UI
- [ ] 10+ round conversation → Check history truncation

### 7.3 User Acceptance Testing
- [ ] Non-technical user can ask questions
- [ ] Technical user can validate reports
- [ ] Knowledge base management is intuitive
- [ ] No crashes on edge cases (empty PDF, corrupted file, etc.)

---

## 8. Deployment Plan

### 8.1 Local Demo (Week 1)
```bash
# Setup
git checkout feature/enhanced-knowledge-base
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your paths

# Run
python web/webUI_v1.py
# Opens http://localhost:7860
```

### 8.2 Company Demo (Week 2)
```bash
# Option 1: Local network
python web/webUI_v1.py --server-name=0.0.0.0 --server-port=7860
# Access via http://<your-ip>:7860

# Option 2: Public share link (easiest)
python web/webUI_v1.py --share
# Generates https://xxx.gradio.live (valid 72 hours)
```

### 8.3 Production Deployment (Future)
- Docker container
- Nginx reverse proxy
- SSL certificate
- User authentication (Gradio supports basic auth)

---

## 9. Success Criteria

### Minimum Viable Demo (MVP)
- ✅ Chat with RAG (answer CVE questions)
- ✅ Upload and validate reports (Summary/Validate/Q&A)
- ✅ Add PDFs to knowledge base
- ✅ View knowledge base sources
- ✅ Stable for 30-minute demo

### Complete Feature Set
- ✅ All MVP features
- ✅ Delete sources from knowledge base
- ✅ Add CVE data by year range
- ✅ Source citation in responses
- ✅ Responsive UI (works on different screen sizes)
- ✅ Error handling (graceful failures)
- ✅ Loading indicators (for long operations)

### Phase 2 Success Criteria
- ✅ LangChain version working with same features
- ✅ Performance comparison documented
- ✅ Decision on which version to keep

---

## 10. Risk Mitigation

### Potential Issues

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Gradio learning curve** | Medium | Use official examples, start simple |
| **Chroma metadata bugs** | High | Test extensively, fallback to full reload |
| **Memory leaks (long sessions)** | Medium | Implement session timeout, periodic cleanup |
| **Large file uploads** | Medium | Set file size limits, show progress bars |
| **Concurrent users** | Low | PoC is single-user, document limitations |

### Fallback Plans

1. **If Gradio doesn't work**: Fall back to Streamlit (similar simplicity)
2. **If Phase 1 too slow**: Skip to Phase 2 (LangChain optimization)
3. **If demo has issues**: Keep CLI version ready as backup

---

## 11. Timeline Estimate

### Week 1: Phase 1 Implementation
- Days 1-2: Core modules + config setup
- Days 3-4: RAG implementation + CLI refactoring
- Days 5-6: Web UI development
- Day 7: Testing and bug fixes

### Week 2: Polish & Phase 2
- Days 1-2: Phase 1 refinement based on testing
- Days 3-5: Phase 2 LangChain implementation
- Days 6-7: Documentation and demo preparation

---

## 12. Open Questions / Decisions Needed

- [ ] **Source citation format**: Show in chat or separate panel?
- [ ] **File size limits**: Max PDF size (50MB? 100MB?)
- [ ] **Concurrent users**: Lock mechanism or per-session Chroma?
- [ ] **Streaming responses**: Show generation in real-time?
- [ ] **Custom instructions**: Allow users to add custom analysis prompts?
- [ ] **Export results**: Download validation reports as PDF/JSON?

---

## 13. References

- Gradio Documentation: https://gradio.app/docs
- LangChain Documentation: https://python.langchain.com/docs
- Chroma Documentation: https://docs.trychroma.com
- Claude Projects UI Reference: https://claude.ai/projects

---

**Document Status**: Complete, ready for implementation
**Next Action**: Begin Phase 1 core module development
