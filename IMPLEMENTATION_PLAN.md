# Multi-File Conversation Context - Implementation Plan v2

**Feature Branch**: `feature/multi-file-conversation-v2`
**Created**: 2025-01-19
**Status**: Planning Phase

---

## 📋 Executive Summary

Re-implement multi-file conversation context feature with improved architecture based on lessons learned from the first attempt (reverted in commit 097f0f2).

**Goal**: Enable users to upload multiple PDF files in a single chat session and ask questions that reference content across all uploaded files.

**Key Improvements Over v1**:
1. ✅ **Fixed score priority issue**: Session files prioritized over KB
2. ✅ **Incremental implementation**: Smaller, testable steps
3. ✅ **Better UI/UX planning**: Clear visual feedback
4. ✅ **Backward compatibility**: No breaking changes to existing features
5. ✅ **Comprehensive testing**: Each phase fully tested before proceeding

---

## 🚨 Lessons Learned from v1

### What Went Wrong
1. **Too aggressive timeline**: 2 hours for 2 PRs was too fast
2. **Architecture flaw**: Fixed score=1.0 for KB caused priority issues
3. **Insufficient UI/UX planning**: No clear user interaction design
4. **Lack of incremental validation**: Should have tested PR 1 thoroughly before PR 2

### What Went Right
1. ✅ Complete test coverage
2. ✅ Detailed documentation
3. ✅ Fast iteration and bug fixing
4. ✅ Brave decision to revert when issues found

### Improvements in v2
- **Smaller phases**: 6 phases instead of 4 PRs
- **Architecture first**: Fix score priority issue in design phase
- **UI/UX mockups**: Plan interaction flow before coding
- **Validation gates**: Each phase must pass tests before next phase
- **Conservative timeline**: Estimate 8-12 hours total (vs. 2.5 hours in v1)

---

## 🏗️ Architecture Design

### Core Components

```
┌─────────────────────────────────────────────────────┐
│                   Web UI                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ File Upload │→ │ SessionMgr   │→ │ File List  │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
│         │                                  │         │
└─────────┼──────────────────────────────────┼─────────┘
          │                                  │
          ▼                                  ▼
  ┌──────────────┐                  ┌──────────────┐
  │ RAG System   │◄─────────────────│ User Query   │
  │              │                  └──────────────┘
  │ query() with │
  │ session_mgr  │
  └──────┬───────┘
         │
         ├──→ Query Session Files (top_k=5, priority=HIGH)
         │    └─→ If results >= 5: Use session only
         │    └─→ Else: Supplement with KB
         │
         └──→ Query KB (top_k=remaining, priority=LOW)
              └─→ Only if session results < 5
```

### Key Architecture Decisions

#### 1. **Session-Scoped Chroma Collection**
```python
Collection name: f"session_{session_id}"
Location: {CHROMA_DB_PATH}/session_{session_id}/
Lifetime: Until cleanup() or timeout
```

**Pros**:
- Supports large files (no memory limits)
- Efficient vector search
- Consistent with existing architecture
- Easy cleanup

**Cons**:
- Disk I/O overhead
- Requires Chroma instance per session

**Decision**: Use session-scoped collection (same as v1)

#### 2. **Priority System: Session First, KB Supplement**

**Problem in v1**:
```python
# v1 approach (BROKEN)
kb_results = query_kb(top_k=3)      # score=1.0 (fixed)
session_results = query_session(top_k=2)  # score=0.0-1.0 (real)
merged = kb_results + session_results
merged.sort(key=lambda x: x['score'], reverse=True)
# Result: KB always wins due to fixed score=1.0 ❌
```

**Solution in v2**:
```python
# v2 approach (FIXED)
session_results = query_session(top_k=5)  # Priority 1
if len(session_results) >= 5:
    return session_results  # Use session only ✅
else:
    remaining = 5 - len(session_results)
    kb_results = query_kb(top_k=remaining)  # Priority 2
    return session_results + kb_results  # Session first ✅
```

**Benefits**:
- ✅ Session files always prioritized
- ✅ KB only used when session insufficient
- ✅ No score comparison issues
- ✅ Clear, predictable behavior

#### 3. **File Metadata Schema**

```python
# Stored in Chroma metadata
{
    "source_type": "session",           # or "permanent" for KB
    "source_name": "report_A.pdf",      # filename
    "session_id": "abc123",             # session identifier
    "chunk_index": 42,                  # chunk number
    "added_date": "2025-01-19T10:30:00",
    "file_size_mb": 2.5,                # original file size
    "precision": "float16"              # embedding precision
}
```

#### 4. **Session Lifecycle**

```
User opens Web UI
    ↓
Generate session_id = uuid.uuid4()
    ↓
Initialize SessionManager(session_id)
    ↓
User uploads files → add_file() → Store in session collection
    ↓
User asks questions → query() → Dual-source retrieval
    ↓
User closes browser OR timeout (1 hour)
    ↓
cleanup() → Delete collection + temp files
```

---

## 📅 Implementation Phases

### **Phase 1: Planning & Architecture** ⏱️ 1-2 hours

**Objectives**:
- ✅ Create this implementation plan
- ✅ Design improved architecture (fix score issue)
- ✅ Create UI/UX mockups
- ✅ Update PROGRESS.md with plan

**Deliverables**:
- `IMPLEMENTATION_PLAN.md` (this file)
- UI/UX mockup (ASCII art or description)
- Updated PROGRESS.md

**Validation**:
- [ ] Plan reviewed and approved
- [ ] Architecture addresses v1 issues
- [ ] UI/UX flow is clear

**Estimated Time**: 1-2 hours

---

### **Phase 2: SessionManager Core** ⏱️ 2-3 hours

#### Step 2.1: Basic SessionManager Class (45 min)
```python
# core/session_manager.py
class SessionManager:
    def __init__(self, session_id: str):
        """Initialize session with unique ID."""
        self.session_id = session_id
        self.files = {}  # {filename: file_info}
        self.temp_dir = Path(f"temp_uploads/session_{session_id}")

    def add_file(self, file_path: str) -> Dict:
        """Add file to session (no Chroma yet)."""
        # Validate file size
        # Copy to temp_dir
        # Extract text with PDFProcessor
        # Store file_info in self.files
        # Return file_info

    def list_files(self) -> List[Dict]:
        """List all files in session."""
        return list(self.files.values())

    def cleanup(self):
        """Delete temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
```

**Tests**:
- [ ] Initialize session
- [ ] Add file (text extraction only)
- [ ] List files
- [ ] Cleanup

#### Step 2.2: Chroma Integration (60 min)
```python
def add_file(self, file_path: str) -> Dict:
    # ... (Step 2.1 code)

    # NEW: Generate embeddings
    chunks = self._split_text_with_overlap(text)
    embeddings = self.embedder.encode(chunks)

    # NEW: Store in Chroma
    if not self.collection:
        self.collection = self.chroma_client.create_collection(
            name=f"session_{self.session_id}"
        )

    self.collection.add(
        ids=[f"{filename}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[metadata for _ in chunks]
    )

def query(self, question: str, top_k: int = 5) -> List[Dict]:
    """Query session files only."""
    query_embedding = self.embedder.encode([question])[0]
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return self._format_results(results)
```

**Tests**:
- [ ] Add file with embeddings
- [ ] Query session files
- [ ] Verify results format

#### Step 2.3: File Removal & Config (45 min)
```python
def remove_file(self, filename: str) -> int:
    """Remove file from session."""
    if filename not in self.files:
        return 0

    # Delete from Chroma
    deleted = self.collection.delete(
        where={"source_name": filename}
    )

    # Delete from temp_dir
    file_path = self.temp_dir / filename
    file_path.unlink(missing_ok=True)

    # Remove from tracking
    del self.files[filename]

    return deleted
```

**Config Updates** (.env.example, config.py):
```bash
# Session Management
SESSION_MAX_FILES=5
SESSION_MAX_FILE_SIZE_MB=10
SESSION_TIMEOUT_HOURS=1
```

**Tests**:
- [ ] Remove file
- [ ] Verify Chroma deletion
- [ ] Verify temp file deletion
- [ ] Config loading

**Phase 2 Deliverables**:
- ✅ `core/session_manager.py` (~300 lines)
- ✅ Config updates
- ✅ Unit tests (~150 lines)

**Phase 2 Validation Gate**:
- [ ] All unit tests pass
- [ ] SessionManager can add/query/remove files
- [ ] No memory leaks
- [ ] Cleanup works correctly

**Estimated Time**: 2-3 hours

---

### **Phase 3: RAG Integration** ⏱️ 2-3 hours

#### Step 3.1: Backward-Compatible query() Modification (60 min)

**rag/pure_python.py**:
```python
class PureRAG:
    def __init__(
        self,
        llama: LlamaModel,
        embedder: EmbeddingModel,
        chroma: ChromaManager,
        session_manager: Optional[SessionManager] = None  # NEW: Optional
    ):
        # ... existing code
        self.session_manager = session_manager  # NEW

    def query(
        self,
        question: str,
        include_history: bool = True,
        return_sources: bool = False
    ) -> Union[str, Dict]:
        """
        Query with optional dual-source retrieval.

        If session_manager is None: Use KB only (backward compatible)
        If session_manager exists: Prioritize session files, supplement with KB
        """
        # NEW: Dual-source retrieval
        if self.session_manager:
            context_chunks = self._dual_source_retrieval(question)
        else:
            context_chunks = self._hybrid_search(question)  # Existing KB-only

        # ... rest of existing code (unchanged)
```

**Tests**:
- [ ] query() with session_manager=None (backward compat)
- [ ] query() with session_manager (new behavior)

#### Step 3.2: Dual-Source Retrieval Logic (90 min)

```python
def _dual_source_retrieval(self, question: str, top_k: int = 5) -> List[str]:
    """
    Prioritize session files, supplement with KB if needed.

    Algorithm:
    1. Query session files (top_k=5)
    2. If session results >= 5: Return session only
    3. Else: Query KB for remaining slots
    4. Return session + KB (session first)
    """
    # Step 1: Query session files (priority 1)
    try:
        session_results = self.session_manager.query(
            question=question,
            top_k=top_k
        )
    except Exception as e:
        logger.error(f"Session query failed: {e}")
        session_results = []

    # Step 2: Check if session sufficient
    if len(session_results) >= top_k:
        return [r['text'] for r in session_results]

    # Step 3: Supplement with KB
    remaining = top_k - len(session_results)
    kb_results = self._hybrid_search(question, top_k=remaining)

    # Step 4: Combine (session first)
    session_texts = [r['text'] for r in session_results]
    return session_texts + kb_results
```

**Tests**:
- [ ] Session has 5+ results → KB not queried
- [ ] Session has 2 results → KB queried for 3
- [ ] Session has 0 results → KB queried for 5
- [ ] Session query error → Fallback to KB only

#### Step 3.3: Mirror in LangChainRAG (30 min)

**rag/langchain_impl.py**:
- Copy exact same logic from pure_python.py
- Maintain behavioral consistency

**Tests**:
- [ ] LangChainRAG mirrors PureRAG behavior

**Phase 3 Deliverables**:
- ✅ Modified RAG classes with dual-source retrieval
- ✅ Integration tests

**Phase 3 Validation Gate**:
- [ ] All tests pass
- [ ] Backward compatibility verified (session_manager=None)
- [ ] Dual-source logic correct (session priority)
- [ ] No performance regression

**Estimated Time**: 2-3 hours

---

### **Phase 4: Web UI - Basic Multi-File** ⏱️ 2-3 hours

#### Step 4.1: UI Layout & State Management (60 min)

**web/web_ui.py** changes:
```python
# Global state (before demo block)
session_id = str(uuid.uuid4())
session_manager = SessionManager(session_id)
rag_system = PureRAG(
    llama=llama_model,
    embedder=embedding_model,
    chroma=chroma_manager,
    session_manager=session_manager  # NEW
)
chat_uploaded_files = []  # List instead of single file

# UI Layout
with gr.Column(scale=7):
    chatbot = gr.Chatbot(...)

    # NEW: File list display
    with gr.Row():
        file_list_display = gr.HTML(value="No files uploaded", label="Session Files")

    with gr.Row():
        msg_input = gr.Textbox(...)
        send_btn = gr.Button(...)

    # NEW: File upload (accumulates instead of replacing)
    with gr.Row():
        chat_file_input = gr.File(label="📎 Upload PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(visible=False)
```

**Tests**:
- [ ] SessionManager initialized on page load
- [ ] File list display renders correctly
- [ ] Upload button visible

#### Step 4.2: File Upload Handler (Accumulation) (60 min)

```python
def handle_chat_file_upload(file):
    """Add file to session (accumulate, don't replace)."""
    if not file:
        return gr.update(), gr.update()

    global chat_uploaded_files

    try:
        # Validate file count
        if len(chat_uploaded_files) >= SESSION_MAX_FILES:
            return (
                gr.update(value=f"❌ Maximum {SESSION_MAX_FILES} files allowed"),
                gr.update()
            )

        # Add to SessionManager
        file_info = session_manager.add_file(file.name)

        # Add to tracking list
        chat_uploaded_files.append(file_info)

        # Update UI
        file_list_html = format_file_list(chat_uploaded_files)
        status_msg = f"✅ {file_info['name']} uploaded ({len(chat_uploaded_files)}/{SESSION_MAX_FILES})"

        return gr.update(value=status_msg), gr.update(value=file_list_html)

    except Exception as e:
        return gr.update(value=f"❌ Upload failed: {str(e)}"), gr.update()

def format_file_list(files: List[Dict]) -> str:
    """Format file list as HTML chips."""
    if not files:
        return "<div>No files uploaded</div>"

    chips = []
    for f in files:
        status_icon = "✅" if f['status'] == 'ready' else "⏳"
        chips.append(
            f'<span style="display:inline-block; margin:5px; padding:5px 10px; '
            f'background:#e8f4f8; border-radius:15px; font-size:14px;">'
            f'{status_icon} {f["name"]} ({f["chunks"]} chunks)'
            f'</span>'
        )
    return "<div>" + "".join(chips) + "</div>"
```

**Tests**:
- [ ] Upload single file
- [ ] Upload multiple files (accumulate)
- [ ] Reject file when limit reached
- [ ] File list display updates

#### Step 4.3: Chat Integration (60 min)

```python
def chat_respond(message: str, history: list):
    """Handle chat with multi-file context."""
    # ... existing intent detection ...

    # NEW: If session has files, prioritize session content
    if chat_uploaded_files:
        # Dual-source retrieval happens automatically in rag_system.query()
        response = rag_system.query(
            question=message,
            include_history=True
        )
    else:
        # KB-only (backward compatible)
        response = rag_system.query(
            question=message,
            include_history=True
        )

    # ... rest of existing code ...
```

**Tests**:
- [ ] Query with no files (KB only)
- [ ] Query with 1 file (session priority)
- [ ] Query with multiple files (all files searchable)
- [ ] Intent detection still works

**Phase 4 Deliverables**:
- ✅ Modified web_ui.py with multi-file upload
- ✅ File list display
- ✅ Basic UI tests

**Phase 4 Validation Gate**:
- [ ] Can upload multiple files
- [ ] Files accumulate (don't replace)
- [ ] File list displays correctly
- [ ] Queries search across all uploaded files
- [ ] Backward compatible (no files = KB only)

**Estimated Time**: 2-3 hours

---

### **Phase 5: Web UI - Advanced Features** ⏱️ 1-2 hours

#### Step 5.1: Individual File Removal (45 min)

```python
def format_file_list(files: List[Dict]) -> str:
    """Format file list with remove buttons."""
    if not files:
        return "<div>No files uploaded</div>"

    chips = []
    for i, f in enumerate(files):
        status_icon = "✅" if f['status'] == 'ready' else "⏳"
        chips.append(
            f'<span style="display:inline-block; margin:5px; padding:5px 10px; '
            f'background:#e8f4f8; border-radius:15px; font-size:14px;">'
            f'{status_icon} {f["name"]} ({f["chunks"]} chunks) '
            f'<button onclick="remove_file({i})">🗑️</button>'  # NEW
            f'</span>'
        )
    return "<div>" + "".join(chips) + "</div>"

def handle_remove_file(filename: str):
    """Remove file from session."""
    global chat_uploaded_files

    # Remove from SessionManager
    deleted_count = session_manager.remove_file(filename)

    # Remove from tracking list
    chat_uploaded_files = [f for f in chat_uploaded_files if f['name'] != filename]

    # Update UI
    file_list_html = format_file_list(chat_uploaded_files)
    status_msg = f"✅ Removed {filename} ({deleted_count} chunks deleted)"

    return gr.update(value=status_msg), gr.update(value=file_list_html)
```

**Tests**:
- [ ] Remove file from session
- [ ] Verify file list updates
- [ ] Verify queries no longer include removed file

#### Step 5.2: Session Cleanup on Page Reload (30 min)

```python
# Add cleanup handler
import atexit

def cleanup_session():
    """Cleanup session on page close."""
    global session_manager
    if session_manager:
        session_manager.cleanup()

atexit.register(cleanup_session)

# Or use Gradio's on_load event
demo.load(fn=lambda: session_manager.cleanup(), outputs=None)
```

**Tests**:
- [ ] Session cleaned up on reload
- [ ] Temp files deleted
- [ ] Chroma collection deleted

#### Step 5.3: File Status Indicators (15 min)

```python
# Add status field to file_info
file_info = {
    'name': filename,
    'status': 'uploading',  # uploading → processing → ready → error
    'chunks': 0,
    'error': None
}

# Update status during processing
file_info['status'] = 'processing'
# ... generate embeddings ...
file_info['status'] = 'ready'
file_info['chunks'] = len(chunks)
```

**Tests**:
- [ ] Status transitions correctly
- [ ] UI shows status icon

**Phase 5 Deliverables**:
- ✅ File removal functionality
- ✅ Session cleanup
- ✅ Status indicators

**Phase 5 Validation Gate**:
- [ ] Can remove individual files
- [ ] Session cleanup works
- [ ] Status indicators correct
- [ ] No resource leaks

**Estimated Time**: 1-2 hours

---

### **Phase 6: Documentation & Final Testing** ⏱️ 1-2 hours

#### Step 6.1: Update CLAUDE.md (30 min)

Add section:
```markdown
### Multi-File Conversation Context

**Feature**: Upload multiple PDF files in a single chat session and ask questions across all files.

**Usage**:
1. Upload first PDF: Files accumulate (don't replace previous)
2. Ask questions: System searches across all uploaded files
3. Upload more PDFs: Up to 5 files per session
4. Remove files: Click 🗑️ button on file chip

**How it works**:
- Session files prioritized over knowledge base
- Automatic cleanup on page reload
- Each session isolated (no interference)

**Limits**:
- Max 5 files per session
- Max 10 MB per file
- Session timeout: 1 hour
```

#### Step 6.2: Update PROGRESS.md (30 min)

Add section:
```markdown
## [2025-01] Multi-File Conversation Context v2

### Improvements Over v1
- Fixed score priority issue (session first, KB supplement)
- Incremental implementation with validation gates
- Better UI/UX with file list display
- Comprehensive testing at each phase

### Implementation
- Phase 1-6 completed
- All validation gates passed
- No known issues
```

#### Step 6.3: End-to-End Testing (30 min)

**Test Scenarios**:
1. [ ] Upload 1 file, ask question → Session-only response
2. [ ] Upload 3 files, ask cross-reference question → Multi-file response
3. [ ] Remove middle file → Other files unaffected
4. [ ] Hit 5-file limit → Upload rejected
5. [ ] Reload page → Session cleaned up
6. [ ] No files uploaded → KB-only (backward compat)

#### Step 6.4: Performance Testing (15 min)

**Metrics**:
- [ ] Query latency with session vs. without
- [ ] Memory usage per session
- [ ] Cleanup time

**Phase 6 Deliverables**:
- ✅ Updated documentation
- ✅ End-to-end tests passed
- ✅ Performance benchmarks

**Phase 6 Validation Gate**:
- [ ] All documentation updated
- [ ] All tests pass
- [ ] Performance acceptable
- [ ] Ready for merge to main

**Estimated Time**: 1-2 hours

---

## 🎯 Success Criteria

### Functional Requirements
- [ ] Users can upload up to 5 PDF files per session
- [ ] Queries search across all uploaded files (session priority)
- [ ] Users can remove individual files
- [ ] Session cleanup on page reload
- [ ] Backward compatible (no files = KB only)

### Non-Functional Requirements
- [ ] Query latency: <2x overhead with session_manager
- [ ] Memory usage: <500 MB per active session
- [ ] Cleanup time: <5 seconds
- [ ] No resource leaks

### Quality Requirements
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All end-to-end tests pass
- [ ] Code coverage >80%
- [ ] Documentation complete

---

## ⏱️ Timeline Estimate

| Phase | Estimated Time | Cumulative |
|-------|---------------|------------|
| Phase 1: Planning | 1-2 hours | 1-2 hours |
| Phase 2: SessionManager | 2-3 hours | 3-5 hours |
| Phase 3: RAG Integration | 2-3 hours | 5-8 hours |
| Phase 4: Web UI Basic | 2-3 hours | 7-11 hours |
| Phase 5: Web UI Advanced | 1-2 hours | 8-13 hours |
| Phase 6: Documentation | 1-2 hours | 9-15 hours |

**Total Estimate**: 9-15 hours (vs. 2.5 hours in v1)

**Conservative Estimate**: 15 hours (allow buffer for unexpected issues)

---

## 🚦 Validation Gates

Each phase must pass its validation gate before proceeding to the next phase.

### Phase 2 Gate
- [ ] All unit tests pass
- [ ] SessionManager can add/query/remove files
- [ ] No memory leaks
- [ ] Cleanup works correctly

### Phase 3 Gate
- [ ] All integration tests pass
- [ ] Backward compatibility verified
- [ ] Dual-source logic correct
- [ ] No performance regression

### Phase 4 Gate
- [ ] Can upload multiple files
- [ ] Files accumulate correctly
- [ ] File list displays correctly
- [ ] Queries work across files

### Phase 5 Gate
- [ ] Can remove individual files
- [ ] Session cleanup works
- [ ] Status indicators correct
- [ ] No resource leaks

### Phase 6 Gate
- [ ] All documentation updated
- [ ] All tests pass
- [ ] Performance acceptable
- [ ] Ready for merge

---

## 📝 Notes

### Decision Log
- **2025-01-19**: Created v2 implementation plan
- **Architecture**: Session-first, KB-supplement (fixes v1 score issue)
- **Timeline**: Conservative 15-hour estimate (vs. 2.5 hours in v1)
- **Approach**: Incremental with validation gates

### Open Questions
- Q: Should we support file formats other than PDF?
  - A: Not in v2. Keep scope focused.

- Q: Should session timeout be configurable per session?
  - A: Not in v2. Use global config only.

- Q: Should we show session files in KB panel?
  - A: No, keep separate. Session files are temporary.

### Future Enhancements (Not in v2)
- [ ] Support other file formats (DOCX, TXT, etc.)
- [ ] Per-session timeout configuration
- [ ] Session persistence across browser restarts
- [ ] Session sharing (collaborative mode)
- [ ] File preview before upload
- [ ] Drag-and-drop file upload
- [ ] Batch file upload

---

## 🔗 References

- Previous attempt: commit feac1cc to 5acf4fb (reverted in 097f0f2)
- Git history analysis: See conversation log for detailed timeline
- Architecture decisions: See "Architecture Design" section above

---

**Status**: ✅ Plan complete, ready to begin Phase 2

**Next Step**: Create core/session_manager.py (Phase 2, Step 2.1)
