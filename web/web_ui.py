"""
Gradio Web UI (Phase 1: Pure Python)
Two-column interface for RAG CVE validation system.

Layout:
- Left: Chat interface with upload button for validation
- Right Top: Instructions (Analysis Settings)
- Right Bottom: Knowledge Base management
"""

import gradio as gr
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import RAG and core modules
from rag.pure_python import PureRAG, ConversationHistory
from core.chroma_manager import ChromaManager
from core.embeddings import EmbeddingModel
from core.pdf_processor import PDFProcessor
from core.session_manager import SessionManager
import re
import numpy as np
import uuid

# Import configuration
from config import (
    GRADIO_SERVER_PORT,
    GRADIO_SHARE,
    GRADIO_SERVER_NAME,
    DEFAULT_SPEED,
    DEFAULT_MODE,
    DEFAULT_SCHEMA,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PRECISION,
    MAX_FILE_UPLOAD_SIZE_MB,
    CVE_V5_PATH,
    CVE_V4_PATH,
    ENABLE_SESSION_AUTO_EMBED,
    RETRIEVAL_TOP_K,
    KB_FILES_DIR,
    SESSION_FILES_BASE
)

# Import CVE processing utilities
from core.cve_lookup import extract_cve_fields
import json
from tqdm.auto import tqdm

# =============================================================================
# Utility functions
# =============================================================================

def split_sentences(text):
    """
    Split text into sentences using both English and Chinese punctuation.

    Supports:
    - Chinese punctuation: 。！？
    - English punctuation: . ! ?
    - Mixed content: handles both languages seamlessly

    Args:
        text: Input text to split

    Returns:
        list: List of sentences
    """
    # Pattern matches Chinese and English sentence-ending punctuation
    pattern = r'[。！？.!?]+[\s\n]*'
    sentences = re.split(pattern, text)
    # Filter out empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]

# =============================================================================
# Global state
# =============================================================================

# Initialize RAG system (singleton)
rag_system = None
chroma_manager = None
embedding_model = None
session_manager = None  # SessionManager for multi-file context
session_id = str(uuid.uuid4())  # Unique session ID

# Track current settings
current_speed = DEFAULT_SPEED
current_mode = DEFAULT_MODE

# Track uploaded files
chat_uploaded_file = None  # Left side: for chat validation (current file display)
chat_file_uploading = False  # Left side upload status
kb_file_processing = False  # Right side: KB processing lock (True = processing, False = ready)
settings_changing = False  # Right side: Settings change lock (True = changing, False = ready)

# Files directory structure (from config)
# KB files: files/knowledge_base/ (permanent, original filenames)
# Session files: files/sessions/session_{id}/ (temporary, UUID filenames)
KB_UPLOAD_DIR = Path(KB_FILES_DIR)
SESSION_UPLOAD_DIR = Path(SESSION_FILES_BASE) / f"session_{session_id}"

# Ensure upload directories exist
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SESSION_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Clean up leftover session files from previous sessions (avoid disk space waste)
# Note: Only clean this specific session's directory
if SESSION_UPLOAD_DIR.exists():
    import glob
    session_files = list(SESSION_UPLOAD_DIR.glob("*"))
    if session_files:
        print(f"[INFO] Cleaning up {len(session_files)} leftover file(s) from previous session...")
        for session_file in session_files:
            try:
                if session_file.is_file():
                    session_file.unlink()
            except Exception as e:
                print(f"[WARNING] Could not delete {session_file.name}: {e}")

def initialize_system(speed_level: str = DEFAULT_SPEED, force_reload: bool = False):
    """Initialize RAG system based on speed level."""
    global rag_system, chroma_manager, embedding_model, current_speed, session_manager

    if rag_system is not None and not force_reload:
        return f"⚠️ System already initialized with speed={current_speed}"

    # Clean up existing model if reloading
    if force_reload and rag_system is not None:
        print(f"Reloading model with new speed: {speed_level}")
        rag_system.cleanup()
        rag_system = None

    print("Initializing RAG system for web UI...")

    # Speed level configuration
    use_fp16 = speed_level in ['fast', 'fastest']
    use_sdpa = speed_level == 'fastest'

    # Initialize SessionManager for multi-file context (only if enabled)
    if session_manager is None and ENABLE_SESSION_AUTO_EMBED:
        session_manager = SessionManager(session_id=session_id)
        print(f"[OK] SessionManager initialized (session_id={session_id[:8]}...)")
    elif not ENABLE_SESSION_AUTO_EMBED:
        print("[INFO] SessionManager disabled (ENABLE_SESSION_AUTO_EMBED=False)")

    # Initialize RAG with session_manager
    rag_system = PureRAG(session_manager=session_manager)
    rag_system.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)

    # Get references for knowledge base management
    chroma_manager = rag_system.chroma
    embedding_model = rag_system.embedder

    # Update current speed
    current_speed = speed_level

    print(f"[OK] RAG system ready (speed={current_speed})")
    return f"✅ System initialized with speed={current_speed}"

def reload_model(new_speed: str) -> str:
    """
    Reload model with new speed settings.

    Args:
        new_speed: New speed level (normal/fast/fastest)

    Returns:
        str: Status message
    """
    try:
        result = initialize_system(speed_level=new_speed, force_reload=True)
        return result
    except Exception as e:
        return f"❌ Reload failed: {str(e)}"

def update_mode(new_mode: str) -> str:
    """
    Update mode setting (no reload needed).

    Args:
        new_mode: New mode (demo/full)

    Returns:
        str: Status message
    """
    global current_mode
    current_mode = new_mode
    return f"✅ Mode updated to: {current_mode}"

def get_current_status() -> str:
    """Get current system status as formatted string."""
    model_name = 'Not loaded'
    if rag_system and hasattr(rag_system, 'llama') and rag_system.llama:
        model_name = rag_system.llama.model_name

    status = f"""
    <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
        <h4>📊 Current Settings</h4>
        <p><b>Speed:</b> {current_speed}</p>
        <p><b>Mode:</b> {current_mode}</p>
        <p><b>Model:</b> {model_name}</p>
    </div>
    """
    return status

# =============================================================================
# Chat interface handlers
# =============================================================================

def detect_user_intent(message: str, has_file: bool) -> str:
    """
    Detect user intent from natural language message.

    Supports Chinese and English phrases for:
    - summarize: Chinese keywords for summary, summary, summarize
    - validate: Chinese keywords for validate, validate, verify, check

    Args:
        message: User message
        has_file: Whether user has uploaded a file

    Returns:
        str: Intent ('summarize', 'validate', or None)
    """
    if not has_file:
        return None

    message_lower = message.lower()

    # Summarize intent keywords (Chinese + English)
    summarize_keywords = [
        # Chinese
        '總結', '摘要', '概括', '整理', '歸納', '內容', '講什麼', '說什麼',
        '主要內容', '主旨', '重點', '要點', '大意',
        # English
        'summarize', 'summary', 'summarise', 'what is this', 'what does',
        'content', 'about', 'main point', 'key point', 'overview'
    ]

    # Validate intent keywords (Chinese + English)
    validate_keywords = [
        # Chinese
        '驗證', '檢驗', '檢查', '查核', '確認', '核實', '校驗', '審核',
        # English
        'validate', 'verify', 'check', 'correct', 'accuracy'
    ]

    # Check for summarize intent
    for keyword in summarize_keywords:
        if keyword in message_lower:
            return 'summarize'

    # Check for validate intent
    for keyword in validate_keywords:
        if keyword in message_lower:
            return 'validate'

    return None

def chat_respond(message: str, history: list):
    """
    Handle chat messages with RAG context retrieval.

    Generator function that yields intermediate states for better UX.

    Args:
        message: User message
        history: Gradio chat history (messages format: list of dicts with 'role' and 'content')

    Yields:
        tuple: (msg_input, msg_input_interactive, history, chat_file_status, remove_file_btn, send_btn,
                upload_file_btn, speed_dropdown, mode_dropdown, delete_btn, refresh_kb_btn,
                source_dropdown, add_file, kb_upload_btn)
    """
    global chat_uploaded_file, chat_file_uploading

    if not message.strip():
        yield "", gr.update(), history, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # Check if settings are being changed
    if settings_changing:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⏳ Please wait, settings are being changed..."})
        yield "", gr.update(), history, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # Check if KB is being processed
    if kb_file_processing:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⏳ Please wait, knowledge base is being updated..."})
        yield "", gr.update(), history, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # Check if file is still uploading
    if chat_file_uploading:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⏳ Please wait, file is still uploading..."})
        yield "", gr.update(), history, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # CRITICAL: Save file path to local variable and immediately clear global
    # This prevents race condition when user uploads new file during processing
    current_file = chat_uploaded_file
    chat_uploaded_file = None  # Clear immediately to allow new uploads

    # Build user message with file attachment if present
    user_content = message
    if current_file:
        # Extract original filename (remove UUID suffix from unique filename)
        # Format: original_name_UUID.ext → original_name.ext
        import re
        full_filename = Path(current_file).name
        match = re.match(r'(.+)_[0-9a-f]{8}(\.\w+)$', full_filename)
        if match:
            # Restore original filename for display
            display_name = match.group(1) + match.group(2)
        else:
            # Fallback to full filename if pattern doesn't match
            display_name = full_filename
        user_content = f"{message}\n\n📎 **Attached:** {display_name}"

    # Immediately show user message with "Thinking..." placeholder and disable critical controls
    # Note: Allow user to type next message and manage files while waiting for response
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": "💭 Thinking..."})

    # Determine remove_file_btn state based on session mode
    # In session mode: keep enabled if files exist, in non-session mode: disable (file already consumed)
    if ENABLE_SESSION_AUTO_EMBED and session_manager and len(session_manager.files) > 0:
        remove_btn_thinking_state = gr.update(interactive=True)
    else:
        remove_btn_thinking_state = gr.update(interactive=False)

    yield (
        "",  # msg_input (clear but keep interactive for typing next message)
        gr.update(interactive=True),   # msg_input (enable - allow typing during processing)
        history,  # history (with thinking message)
        "",  # chat_file_status (clear)
        remove_btn_thinking_state,  # remove_file_btn (disable in non-session mode, enable if session has files)
        gr.update(interactive=False),  # send_btn (disable - prevent sending)
        gr.update(interactive=True),   # upload_file_btn (enable - allow file management)
        gr.update(interactive=False),  # speed_dropdown (disable)
        gr.update(interactive=False),  # mode_dropdown (disable)
        gr.update(interactive=False),  # delete_btn (disable)
        gr.update(interactive=False),  # refresh_kb_btn (disable)
        gr.update(interactive=False),  # source_dropdown (disable)
        gr.update(interactive=False),  # add_file (disable)
        gr.update(interactive=False)   # kb_upload_btn (disable)
    )

    try:
        import os

        # Detect user intent using natural language processing
        intent = detect_user_intent(message, has_file=bool(current_file))

        # Handle file-specific actions based on detected intent
        if current_file:
            if intent == 'summarize':
                response = process_uploaded_report(current_file, action='summarize', mode=current_mode)
            elif intent == 'validate':
                response = process_uploaded_report(current_file, action='validate', schema=DEFAULT_SCHEMA, mode=current_mode)
            else:
                # In session mode, use RAG query on embedded content (much faster!)
                # Otherwise, fall back to Q&A on file content (slower, processes whole PDF)
                if ENABLE_SESSION_AUTO_EMBED and session_manager and len(session_manager.files) > 0:
                    # Use RAG query - file is already embedded in SessionManager
                    response = rag_system.query(
                        question=message,
                        include_history=True,
                        max_tokens=1000
                    )
                else:
                    # Non-session mode: Process PDF directly
                    response = process_uploaded_report(current_file, action='qa', question=message, mode=current_mode)
        else:
            # Normal RAG query (no file attached)
            response = rag_system.query(
                question=message,
                include_history=True,
                max_tokens=1000
            )

        # Delete uploaded file from disk if exists (use local variable, not global)
        if current_file and os.path.exists(current_file):
            try:
                os.remove(current_file)
            except Exception as e:
                print(f"Warning: Could not delete file {current_file}: {e}")

        # Update history with actual response and re-enable all UI elements
        # Note: Send button state depends on whether user typed something during processing
        history[-1]["content"] = response

        # In session mode, show file count and keep remove button enabled if files exist
        if ENABLE_SESSION_AUTO_EMBED and session_manager and len(session_manager.files) > 0:
            file_count = len(session_manager.files)
            session_status_html = f"""
            <div style='padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;'>
                <p style='margin: 0;'><b>📚 Session Files</b></p>
                <p style='margin: 5px 0 0 0; color: #155724;'>✅ {file_count} file{'s' if file_count != 1 else ''} in context</p>
            </div>
            """
            remove_btn_state = gr.update(interactive=True)
        else:
            session_status_html = ""
            remove_btn_state = gr.update(interactive=False)

        yield (
            "",  # msg_input (clear - user may have typed during processing, now cleared)
            gr.update(interactive=True),   # msg_input (enable - allow typing)
            history,  # history (with response)
            session_status_html,  # chat_file_status (show session files or clear)
            remove_btn_state,  # remove_file_btn (enable if session has files, disable otherwise)
            gr.update(interactive=False),  # send_btn (disable - input just cleared)
            gr.update(interactive=True),   # upload_file_btn (enable)
            gr.update(interactive=True),   # speed_dropdown (enable)
            gr.update(interactive=True),   # mode_dropdown (enable)
            gr.update(interactive=False),  # delete_btn (disable - safe default)
            gr.update(interactive=True),   # refresh_kb_btn (enable)
            gr.update(interactive=True, value=None),   # source_dropdown (enable but reset selection)
            gr.update(interactive=True),   # add_file (enable)
            gr.update(interactive=True)    # kb_upload_btn (enable)
        )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1]["content"] = error_msg

        # Delete uploaded file from disk if exists (even on error, use local variable)
        import os
        if current_file and os.path.exists(current_file):
            try:
                os.remove(current_file)
            except Exception as e:
                print(f"Warning: Could not delete file {current_file}: {e}")

        # In session mode, show file count and keep remove button enabled if files exist
        if ENABLE_SESSION_AUTO_EMBED and session_manager and len(session_manager.files) > 0:
            file_count = len(session_manager.files)
            session_status_html = f"""
            <div style='padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;'>
                <p style='margin: 0;'><b>📚 Session Files</b></p>
                <p style='margin: 5px 0 0 0; color: #155724;'>✅ {file_count} file{'s' if file_count != 1 else ''} in context</p>
            </div>
            """
            remove_btn_state = gr.update(interactive=True)
        else:
            session_status_html = ""
            remove_btn_state = gr.update(interactive=False)

        yield (
            "",  # msg_input (clear)
            gr.update(interactive=True),   # msg_input (enable - allow typing)
            history,  # history (with error)
            session_status_html,  # chat_file_status (show session files or clear)
            remove_btn_state,  # remove_file_btn (enable if session has files, disable otherwise)
            gr.update(interactive=False),  # send_btn (disable - input just cleared)
            gr.update(interactive=True),   # upload_file_btn (enable)
            gr.update(interactive=True),   # speed_dropdown (enable)
            gr.update(interactive=True),   # mode_dropdown (enable)
            gr.update(interactive=False),  # delete_btn (disable - safe default)
            gr.update(interactive=True),   # refresh_kb_btn (enable)
            gr.update(interactive=True, value=None),   # source_dropdown (enable but reset selection)
            gr.update(interactive=True),   # add_file (enable)
            gr.update(interactive=True)    # kb_upload_btn (enable)
        )

def handle_chat_file_upload(file):
    """
    Handle file upload for chat (left side).
    If a file already exists, delete it before uploading the new one (only support 1 file at a time).

    Args:
        file: Gradio File object (path string)

    Returns:
        tuple: (HTML status display, remove_btn update to enable)
    """
    global chat_uploaded_file, chat_file_uploading
    import shutil
    import time
    import os

    if file is None:
        return "", gr.update(interactive=False)

    try:
        # Set uploading status
        chat_file_uploading = True

        # Delete existing file if present (only support one file at a time)
        if chat_uploaded_file and os.path.exists(chat_uploaded_file):
            try:
                os.remove(chat_uploaded_file)
                print(f"[INFO] Deleted previous file: {Path(chat_uploaded_file).name}")
            except Exception as e:
                print(f"[WARNING] Could not delete previous file: {e}")

        # Get source file path
        source_path = Path(file)
        original_name = source_path.name

        # Show uploading status
        uploading_html = f"""
        <div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>
            <p style='margin: 0;'><b>📄 {original_name}</b></p>
            <p style='margin: 5px 0 0 0; color: #856404;'>🔄 Uploading...</p>
        </div>
        """

        # Simulate upload delay (for demo purposes, can remove in production)
        time.sleep(0.5)

        # Generate unique filename to prevent overwriting files being processed
        # Format: original_name_UUID.ext (e.g., "report_a1b2c3d4.pdf")
        import uuid
        file_stem = source_path.stem  # filename without extension
        file_ext = source_path.suffix  # extension including dot
        unique_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        unique_filename = f"{file_stem}_{unique_id}{file_ext}"
        dest_path = SESSION_UPLOAD_DIR / unique_filename

        shutil.copy2(source_path, dest_path)

        # Update global state
        chat_uploaded_file = str(dest_path)

        # Add file to SessionManager for multi-file context (if enabled)
        if session_manager and ENABLE_SESSION_AUTO_EMBED:
            try:
                file_info = session_manager.add_file(str(dest_path))
                print(f"[OK] File added to session: {original_name} (saved as {unique_filename}, {file_info['chunks']} chunks)")
            except Exception as e:
                print(f"[WARNING] Failed to add file to session: {e}")

        chat_file_uploading = False

        # Show success status with file count (display original name to user)
        file_count = len(session_manager.files) if (session_manager and ENABLE_SESSION_AUTO_EMBED) else 1
        success_html = f"""
        <div style='padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;'>
            <p style='margin: 0;'><b>📄 {original_name}</b></p>
            <p style='margin: 5px 0 0 0; color: #155724;'>✅ Ready (Session: {file_count} file{'s' if file_count != 1 else ''})</p>
        </div>
        """

        # Enable remove button
        return success_html, gr.update(interactive=True)

    except Exception as e:
        chat_file_uploading = False
        # Get file name if possible, otherwise use generic label
        try:
            display_name = Path(file).name if file else "File"
        except:
            display_name = "File"

        error_html = f"""
        <div style='padding: 10px; background-color: #f8d7da; border-radius: 5px; border-left: 4px solid #dc3545;'>
            <p style='margin: 0;'><b>📄 {display_name}</b></p>
            <p style='margin: 5px 0 0 0; color: #721c24;'>❌ Upload Error</p>
        </div>
        """
        return error_html, gr.update(interactive=False)

def handle_remove_chat_file():
    """
    Remove uploaded file from chat (left side).
    In session mode, removes the last file from SessionManager.

    Returns:
        tuple: (Status HTML or empty, button update based on remaining files)
    """
    global chat_uploaded_file, chat_file_uploading
    import os

    # In session mode, remove last file from SessionManager
    if ENABLE_SESSION_AUTO_EMBED and session_manager and len(session_manager.files) > 0:
        try:
            # Remove the most recent file
            last_file = list(session_manager.files.keys())[-1]
            session_manager.remove_file(last_file)
            print(f"[OK] Removed file from session: {last_file}")

            # Update status based on remaining files
            remaining_count = len(session_manager.files)
            if remaining_count > 0:
                status_html = f"""
                <div style='padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;'>
                    <p style='margin: 0;'><b>📚 Session Files</b></p>
                    <p style='margin: 5px 0 0 0; color: #155724;'>✅ {remaining_count} file{'s' if remaining_count != 1 else ''} in context</p>
                </div>
                """
                return status_html, gr.update(interactive=True)
            else:
                return "", gr.update(interactive=False)
        except Exception as e:
            print(f"Warning: Could not remove file from session: {e}")
            return "", gr.update(interactive=False)
    else:
        # Non-session mode: delete single file
        if chat_uploaded_file and os.path.exists(chat_uploaded_file):
            try:
                os.remove(chat_uploaded_file)
            except Exception as e:
                print(f"Warning: Could not delete file {chat_uploaded_file}: {e}")

        # Clear global state
        chat_uploaded_file = None
        chat_file_uploading = False

        # Return empty HTML to clear display and disable button
        return "", gr.update(interactive=False)

def process_uploaded_report(
    file,
    action: str,
    schema: str = DEFAULT_SCHEMA,
    mode: str = None,
    question: str = None
) -> str:
    """
    Process uploaded report based on user action.

    Args:
        file: Gradio File object or file path string
        action: 'summarize', 'validate', 'add', or 'qa'
        schema: CVE schema to use
        mode: Processing mode (demo/full), uses global current_mode if None
        question: User question (required for 'qa' action)

    Returns:
        str: Processing result
    """
    if file is None:
        return "⚠️ No file uploaded"

    # Use global mode if not specified
    mode = mode or current_mode

    # Mode-specific settings (for PDF extraction and validation only)
    # Note: Summary token limits are now controlled by .env (SUMMARY_* parameters)
    if mode == 'demo':
        max_pages = 10
        top_k = 3  # Demo mode uses fewer results for speed
        validation_tokens = 256
    else:  # full mode
        max_pages = None
        top_k = RETRIEVAL_TOP_K  # Use config value
        validation_tokens = 700

    try:
        # Handle both string paths and Gradio File objects
        if isinstance(file, str):
            file_path = Path(file)
        else:
            file_path = Path(file.name)

        if action == 'summarize':
            # Extract text and summarize (uses .env SUMMARY_* configuration)
            pdf_processor = PDFProcessor()
            text = pdf_processor.extract_text(file_path, max_pages=max_pages)

            try:
                summary = rag_system.summarize_report(text)  # Let RAG class use .env config
                return f"📝 Summary:\n\n{summary}"
            except RuntimeError as e:
                # Catch CUDA out-of-memory errors
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    return (
                        "❌ GPU Memory Error\n\n"
                        "GPU ran out of memory during summarization. This can happen when:\n"
                        "- ENABLE_SESSION_AUTO_EMBED=True (embedding model + LLM model on GPU)\n"
                        "- Large PDF files requiring multiple processing chunks\n"
                        "- GTX 1660 Ti (6GB VRAM) is at its limit\n\n"
                        "**Solutions:**\n"
                        "1. Set ENABLE_SESSION_AUTO_EMBED=False in .env (recommended for 6GB VRAM)\n"
                        "2. Use smaller PDF files or demo mode (--mode=demo)\n"
                        "3. Restart the web UI to clear all GPU memory\n"
                        "4. Use CPU mode by setting CUDA_DEVICE=-1 in .env (slower but stable)\n\n"
                        f"Technical details: {str(e)}"
                    )
                else:
                    raise

        elif action == 'validate':
            # Process report and validate CVE usage
            try:
                report_text, cves, cve_descriptions = rag_system.process_report_for_cve_validation(
                    str(file_path),
                    schema=schema,
                    max_pages=max_pages
                )
                validation = rag_system.validate_cve_usage(
                    report_text,
                    cve_descriptions,
                    max_tokens=validation_tokens
                )
                return f"✅ Validation Result:\n\n{validation}\n\n📋 Found CVEs: {', '.join(cves)}"
            except RuntimeError as e:
                # Catch CUDA out-of-memory errors
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    return (
                        "❌ GPU Memory Error\n\n"
                        "GPU ran out of memory during CVE validation. "
                        "Try setting ENABLE_SESSION_AUTO_EMBED=False in .env or using demo mode.\n\n"
                        f"Technical details: {str(e)}"
                    )
                else:
                    raise

        elif action == 'qa':
            # Answer question about report (uses .env QA_* configuration)
            if not question:
                return "⚠️ No question provided for Q&A"

            pdf_processor = PDFProcessor()
            text = pdf_processor.extract_text(file_path, max_pages=max_pages)

            try:
                answer = rag_system.answer_question_about_report(text, question)  # Let RAG class use .env config
                return f"💬 Answer:\n\n{answer}"
            except RuntimeError as e:
                # Catch CUDA out-of-memory errors
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    return (
                        "❌ GPU Memory Error\n\n"
                        "GPU ran out of memory during Q&A. "
                        "Try setting ENABLE_SESSION_AUTO_EMBED=False in .env or using demo mode.\n\n"
                        f"Technical details: {str(e)}"
                    )
                else:
                    raise

        elif action == 'add':
            # Add to knowledge base
            result = add_pdf_to_kb(file, file_path.name)
            return result

        else:
            return f"⚠️ Unknown action: {action}"

    except Exception as e:
        return f"❌ Processing error: {str(e)}"

# =============================================================================
# Knowledge base management
# =============================================================================

def handle_kb_file_upload(file):
    """
    Handle file upload for Knowledge Base (right side).
    Uploads file and immediately processes it to generate embeddings.

    New behavior:
    - Uses original filename (no UUID)
    - Permanently stores file in files/knowledge_base/
    - Locks upload during processing (kb_file_processing flag)

    Args:
        file: Gradio File object (path string)

    Returns:
        tuple: (status_html, updated_kb_display, updated_dropdown_choices, cleared_file_input, kb_upload_btn_state, send_btn_state)
    """
    global kb_file_processing
    import shutil

    if file is None:
        return "", format_kb_display(), gr.update(choices=get_source_names()), None, gr.update(interactive=True), gr.update(interactive=True)

    # Check if settings are being changed
    if settings_changing:
        return (
            "⚠️ Settings are being changed. Please wait...",
            format_kb_display(),
            gr.update(choices=get_source_names()),
            None,
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    # Check if already processing
    if kb_file_processing:
        return (
            "⚠️ Another file is being processed. Please wait...",
            format_kb_display(),
            gr.update(choices=get_source_names()),
            None,
            gr.update(interactive=False),
            gr.update(interactive=False)  # Also disable left side Send button
        )

    try:
        # Set processing lock
        kb_file_processing = True

        # Get source file path
        source_path = Path(file)
        original_name = source_path.name

        # Use original filename (permanent storage in KB)
        dest_path = KB_UPLOAD_DIR / original_name

        # Check if file already exists in KB
        if dest_path.exists():
            kb_file_processing = False
            return (
                f"❌ File '{original_name}' already exists in knowledge base. Please remove it first or use a different filename.",
                format_kb_display(),
                gr.update(choices=get_source_names()),
                None,
                gr.update(interactive=True),
                gr.update(interactive=True)  # Re-enable left side Send button
            )

        # Copy file to files/knowledge_base/ with original name
        shutil.copy2(source_path, dest_path)

        # Immediately process the file
        print(f"Adding {original_name} to knowledge base...")

        # Extract text
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text(dest_path)

        # Split into sentences (supports both English and Chinese)
        sentences = split_sentences(text)

        # Create chunks
        chunks = []
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = "".join(sentences[i:i+CHUNK_SIZE]).strip()
            if chunk:
                chunks.append(chunk)

        # Generate embeddings
        embeddings = embedding_model.encode(
            chunks,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            precision=EMBEDDING_PRECISION
        )

        # Convert to list
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Prepare metadata
        metadata = [
            {
                'source_type': 'pdf',
                'source_name': original_name,
                'added_date': datetime.now().isoformat(),
                'chunk_index': i,
                'precision': EMBEDDING_PRECISION
            }
            for i in range(len(chunks))
        ]

        # Add to Chroma
        n_added = chroma_manager.add_documents(
            texts=chunks,
            embeddings=embeddings_list,
            metadata=metadata
        )

        print(f"[OK] Added {n_added} chunks from {original_name} to knowledge base")

        # Release processing lock
        kb_file_processing = False

        # Return empty status (hide immediately), updated KB display, dropdown, and clear file input
        return "", format_kb_display(), gr.update(choices=get_source_names()), None, gr.update(interactive=True), gr.update(interactive=True)

    except Exception as e:
        print(f"[ERROR] Error adding {file.name if file else 'file'} to knowledge base: {str(e)}")

        # Delete the file on error (rollback)
        import os
        try:
            if 'dest_path' in locals() and dest_path.exists():
                os.remove(dest_path)
                print(f"[OK] Rolled back file after error: {original_name}")
        except Exception as cleanup_error:
            print(f"[WARNING] Could not delete file: {cleanup_error}")

        # Release processing lock
        kb_file_processing = False

        # Return empty status even on error (hide immediately), and clear file input
        return "", format_kb_display(), gr.update(choices=get_source_names()), None, gr.update(interactive=True), gr.update(interactive=True)

def get_kb_sources() -> list:
    """
    Get list of sources in knowledge base.

    Returns:
        list: List of source dicts
    """
    try:
        sources = chroma_manager.list_sources()
        return sources
    except Exception as e:
        print(f"Error getting sources: {e}")
        return []

def format_kb_display() -> str:
    """
    Format knowledge base sources for display.

    Returns:
        str: Formatted HTML string
    """
    try:
        stats = chroma_manager.get_stats()
        sources = chroma_manager.list_sources()

        # Build HTML with collapsible sources list
        html = f"""
        <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
            <h4>📊 Statistics</h4>
            <p><b>Total documents:</b> {stats['total_docs']}</p>
            <p><b>By type:</b> {dict(stats['by_source_type'])}</p>
            <br>
            <details>
                <summary style='cursor: pointer; font-weight: bold; font-size: 1.1em;'>
                    📚 Sources ({len(sources)})
                </summary>
                <ul style='list-style: none; padding-left: 0; margin-top: 10px;'>
        """

        for source in sources:
            icon = "📄" if source['type'] == 'pdf' else "🔖"
            date = source['added_date'][:10]  # Just date, not time
            html += f"<li>{icon} <b>{source['name']}</b> ({source['count']} chunks, added {date})</li>"

        html += """
                </ul>
            </details>
        </div>
        """

        return html

    except Exception as e:
        return f"<p>❌ Error loading sources: {str(e)}</p>"

def get_source_names() -> list:
    """
    Get list of source names for dropdown.

    Returns:
        list: List of source names
    """
    try:
        sources = chroma_manager.list_sources()
        return [source['name'] for source in sources]
    except Exception as e:
        print(f"Error getting source names: {e}")
        return []

def delete_source(source_name: str) -> tuple:
    """
    Delete a single source from knowledge base.
    Also deletes the original file from files/knowledge_base/ if it exists.

    Args:
        source_name: Name of source to delete

    Returns:
        tuple: (status_message, updated_kb_display, updated_dropdown_choices)
    """
    if not source_name:
        return "⚠️ No source selected", format_kb_display(), get_source_names()

    try:
        # Delete the source embeddings
        n_deleted = chroma_manager.delete_by_source(source_name)

        if n_deleted > 0:
            status = f"✅ Deleted {n_deleted} chunks from '{source_name}'"

            # Also delete the original file if it exists in KB files directory
            file_path = KB_UPLOAD_DIR / source_name
            if file_path.exists() and file_path.is_file():
                try:
                    import os
                    os.remove(file_path)
                    print(f"[OK] Deleted original file: {source_name}")
                except Exception as e:
                    print(f"[WARNING] Could not delete original file {source_name}: {e}")
        else:
            status = f"⚠️ Source '{source_name}' not found"

        # Get updated displays
        updated_display = format_kb_display()
        updated_choices = get_source_names()

        return status, updated_display, updated_choices

    except Exception as e:
        return f"❌ Error deleting source: {str(e)}", format_kb_display(), get_source_names()

def parse_year_input(year_input: str, schema: str) -> list:
    """
    Parse year input string into list of years.

    Args:
        year_input: Year input string (single, range, comma-separated, or 'all')
        schema: Schema selection ('v5', 'v4', 'all') - used for 'all' year detection

    Returns:
        list: List of years to process

    Raises:
        ValueError: If input format is invalid
    """
    year_input = year_input.strip()

    if year_input.lower() == 'all':
        # Scan directories based on schema selection
        from pathlib import Path
        import os

        def get_available_years(base_path):
            """Scan directory for available year folders."""
            if not os.path.exists(base_path):
                return []
            years = []
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path) and item.isdigit() and len(item) == 4:
                    years.append(int(item))
            return sorted(years)

        v5_years = get_available_years(str(CVE_V5_PATH)) if schema in ['v5', 'all'] else []
        v4_years = get_available_years(str(CVE_V4_PATH)) if schema in ['v4', 'all'] else []
        years = sorted(set(v5_years + v4_years))

        if not years:
            raise ValueError(f"No year directories found in CVE feeds for schema '{schema}'")

        return years

    elif '-' in year_input and ',' not in year_input:
        # Range format: 2023-2025
        try:
            start_year, end_year = year_input.split('-')
            start_year = int(start_year.strip())
            end_year = int(end_year.strip())
            if start_year > end_year:
                raise ValueError(f"Invalid range: start year {start_year} > end year {end_year}")
            return list(range(start_year, end_year + 1))
        except ValueError as e:
            raise ValueError(f"Invalid year range format: {year_input}. {str(e)}")

    elif ',' in year_input:
        # Comma-separated format: 2023,2024,2025
        try:
            years = [int(y.strip()) for y in year_input.split(',')]
            return years
        except ValueError:
            raise ValueError(f"Invalid year format: {year_input}. Use single year (2025), range (2023-2025), comma-separated (2023,2024,2025), or 'all'")

    else:
        # Single year
        try:
            return [int(year_input)]
        except ValueError:
            raise ValueError(f"Invalid year format: {year_input}. Use single year (2025), range (2023-2025), comma-separated (2023,2024,2025), or 'all'")

def add_cve_data_to_kb(
    year_input: str,
    schema: str,
    replace_mode: bool,
    filter_keyword: str,
    progress=gr.Progress()
) -> tuple:
    """
    Add CVE data to knowledge base with progress tracking.

    Args:
        year_input: Year input string (single, range, comma-separated, or 'all')
        schema: CVE schema ('v5', 'v4', or 'all')
        replace_mode: Whether to replace existing year data
        filter_keyword: Optional keyword filter (case-insensitive)
        progress: Gradio Progress object

    Returns:
        tuple: (status_message, updated_kb_display, updated_dropdown_choices)
    """
    if not year_input or not year_input.strip():
        return "⚠️ Please enter year(s)", format_kb_display(), get_source_names()

    try:
        # Parse year input
        progress(0, desc="Parsing year input...")
        years = parse_year_input(year_input, schema)

        status_msg = f"Processing years: {years}\nSchema: {schema}\nMode: {'Replace' if replace_mode else 'Add only'}\n"
        if filter_keyword:
            status_msg += f"Filter: '{filter_keyword}'\n"
        status_msg += "\n"

        # Initialize progress tracking
        total_years = len(years)
        total_added = 0

        for year_idx, year in enumerate(years):
            year_progress = year_idx / total_years
            progress(year_progress, desc=f"Processing year {year} ({year_idx + 1}/{total_years})")

            # If replace mode, delete existing year data first
            if replace_mode:
                progress(year_progress, desc=f"Deleting existing data for year {year}...")
                deleted_count = chroma_manager.delete_by_year(year, schema)
                if deleted_count > 0:
                    status_msg += f"🗑️ Year {year}: Deleted {deleted_count} existing documents\n"

            # Determine paths based on schema
            paths_to_check = []
            if schema in ['v5', 'all']:
                v5_year_path = CVE_V5_PATH / str(year)
                if v5_year_path.exists():
                    paths_to_check.append(('v5', v5_year_path))

            if schema in ['v4', 'all']:
                v4_year_path = CVE_V4_PATH / str(year)
                if v4_year_path.exists():
                    paths_to_check.append(('v4', v4_year_path))

            if not paths_to_check:
                status_msg += f"⚠️ Year {year}: No CVE data found\n"
                continue

            # Collect CVE data for this year
            year_cve_texts = []
            year_cve_metadata = []

            for schema_type, year_path in paths_to_check:
                progress(year_progress, desc=f"Scanning {schema_type.upper()} for year {year}...")

                subdirs = list(year_path.iterdir())
                for subdir_idx, subdir in enumerate(subdirs):
                    if not subdir.is_dir():
                        continue

                    # Update progress
                    subdir_progress = year_progress + (subdir_idx / len(subdirs)) * (1 / total_years) * 0.8
                    progress(subdir_progress, desc=f"Year {year} {schema_type.upper()}: {subdir.name}")

                    for json_file in subdir.glob("*.json"):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            cve_id, vendor, product, description = extract_cve_fields(data, json_file.stem)

                            cve_text = (
                                f"CVE Number: {cve_id}, "
                                f"Vendor: {vendor}, "
                                f"Product: {product}, "
                                f"Description: {description}"
                            )

                            # Apply keyword filter if specified
                            if filter_keyword and filter_keyword.lower() not in cve_text.lower():
                                continue

                            year_cve_texts.append(cve_text)
                            year_cve_metadata.append({
                                'source_type': 'cve',
                                'source_name': f"CVE_{year}_{schema_type}",
                                'cve_id': cve_id,
                                'added_date': datetime.now().isoformat(),
                                'chunk_index': len(year_cve_texts) - 1,
                                'precision': EMBEDDING_PRECISION
                            })

                        except Exception as e:
                            continue

            if not year_cve_texts:
                status_msg += f"⚠️ Year {year}: No CVE descriptions extracted\n"
                continue

            # Generate embeddings
            progress(year_progress + 0.8 / total_years, desc=f"Generating embeddings for {len(year_cve_texts)} CVEs...")
            embeddings = embedding_model.encode(
                year_cve_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                precision=EMBEDDING_PRECISION
            )

            # Convert to list format
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Add to Chroma
            progress(year_progress + 0.9 / total_years, desc=f"Adding to database...")
            n_added = chroma_manager.add_documents(
                texts=year_cve_texts,
                embeddings=embeddings_list,
                metadata=year_cve_metadata
            )

            total_added += n_added
            status_msg += f"✅ Year {year}: Added {n_added} CVE descriptions\n"

        progress(1.0, desc="Complete!")
        status_msg += f"\n{'='*60}\n"
        status_msg += f"✅ Total added: {total_added} CVE descriptions\n"
        status_msg += f"{'='*60}"

        # Get updated displays
        updated_display = format_kb_display()
        updated_choices = get_source_names()

        return status_msg, updated_display, updated_choices

    except Exception as e:
        error_msg = f"❌ Error adding CVE data: {str(e)}"
        return error_msg, format_kb_display(), get_source_names()

# =============================================================================
# Gradio interface
# =============================================================================

def create_interface():
    """Create and configure Gradio interface."""

    with gr.Blocks(title="RAG CVE Validation System", theme=gr.themes.Soft(), css="""
        /* Remove blue outline/border from disabled components */
        .disabled, [disabled], :disabled {
            outline: none !important;
            border-color: transparent !important;
            box-shadow: none !important;
        }
        /* Specifically target Gradio input components when disabled */
        .gr-textbox:disabled, .gr-textbox[disabled],
        .gr-dropdown:disabled, .gr-dropdown[disabled],
        .gr-file:disabled, .gr-file[disabled] {
            outline: none !important;
            border-color: transparent !important;
            box-shadow: none !important;
        }
        /* Remove focus outline from disabled components */
        *:disabled:focus, *[disabled]:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        /* Target Gradio block containers that contain disabled elements */
        .block:has(textarea:disabled),
        .block:has(input:disabled),
        .block:has(select:disabled),
        #msg_input:has(textarea:disabled) {
            border-color: transparent !important;
            outline: none !important;
            box-shadow: none !important;
        }
        /* Remove generating state blue border */
        .generating {
            border-color: transparent !important;
        }
    """) as demo:
        # Title
        gr.Markdown("# 🛡️ RAG CVE Validation System")
        gr.Markdown("Conversational AI for threat intelligence report analysis")

        with gr.Row():
            # Left column: Chat interface (7/12 width)
            with gr.Column(scale=7):
                gr.Markdown("### 💬 Conversation")

                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_copy_button=True,
                    type='messages',  # Use OpenAI-style message format
                    elem_id="chatbot"
                )

                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about CVEs, security reports, or upload a file for validation...",
                    lines=2,
                    show_label=False,
                    elem_id="msg_input"
                )

                # Action buttons row
                with gr.Row():
                    upload_file_btn = gr.UploadButton(
                        "➕ Add File",
                        file_types=[".pdf"],
                        file_count="single",
                        size="sm",
                        scale=1
                    )
                    remove_file_btn = gr.Button("🗑️", size="sm", scale=0, min_width=40, interactive=False)
                    send_btn = gr.Button("Send →", size="sm", scale=1, variant="primary", elem_id="send_btn", interactive=False)
                    with gr.Column(scale=8):
                        pass  # Spacer

                # File upload status display
                chat_file_status = gr.HTML(value="")

            # Right column: Settings and Knowledge Base (5/12 width)
            with gr.Column(scale=5):
                # Current Status Display
                gr.Markdown("### ⚙️ Analysis Settings")
                with gr.Group():
                    status_display = gr.HTML(
                        value=get_current_status(),
                        label="Current Settings"
                    )

                    speed_dropdown = gr.Dropdown(
                        choices=['normal', 'fast', 'fastest'],
                        value=DEFAULT_SPEED,
                        label="Speed",
                        info="⚠️ Changing speed will reload the model (takes 1-2 min)"
                    )
                    mode_dropdown = gr.Dropdown(
                        choices=['demo', 'full'],
                        value=DEFAULT_MODE,
                        label="Mode",
                        info="demo: 10 pages, 256 tokens | full: All pages, 700 tokens"
                    )

                gr.Markdown("---")

                # Knowledge Base panel
                gr.Markdown("### 📚 Knowledge Base")
                with gr.Group():
                    # Sources display (with collapsible list in HTML)
                    kb_display = gr.HTML(
                        value=format_kb_display(),
                        label="Sources"
                    )
                    with gr.Row():
                        refresh_kb_btn = gr.Button("🔄 Refresh", size="sm", scale=1)

                    # Delete section
                    source_dropdown = gr.Dropdown(
                        choices=[],
                        label="Remove",
                        interactive=True,
                        value=None
                    )
                    delete_btn = gr.Button("🗑️ Delete Selected Source", size="sm", variant="stop", interactive=False)

                    # Add section
                    add_file = gr.File(
                        label="Add",
                        file_types=[".pdf"],
                        file_count="single",
                        type="filepath"
                    )
                    kb_upload_btn = gr.UploadButton(
                        "➕ Add File To Knowledge Base",
                        file_types=[".pdf"],
                        file_count="single",
                        size="sm"
                    )
                    kb_file_status = gr.HTML(value="", visible=False)

        # Event handlers

        def handle_refresh_kb():
            """Refresh knowledge base display and source dropdown."""
            updated_display = format_kb_display()
            updated_sources = get_source_names()
            # Disable delete button since dropdown will be reset to None
            return updated_display, gr.update(choices=updated_sources, value=None), gr.update(interactive=False)

        def handle_delete_source(source_name):
            """Handle delete source button click."""
            if not source_name:
                return format_kb_display(), gr.update(), gr.update(interactive=False)

            status, updated_display, updated_sources = delete_source(source_name)
            print(status)  # Print to console instead of UI
            # After deletion, reset dropdown to None and disable delete button
            return updated_display, gr.update(choices=updated_sources, value=None), gr.update(interactive=False)

        def handle_source_dropdown_change(source_name):
            """Handle source dropdown selection change - enable/disable delete button."""
            # Enable delete button only if a source is selected
            if source_name:
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)

        def handle_speed_change(new_speed):
            """Handle speed dropdown change - reload model."""
            global settings_changing

            # Lock all operations
            settings_changing = True

            # First yield: Disable all buttons and show processing status
            processing_status = """
            <div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>
                <p style='margin: 0;'><b>⚙️ Changing Speed Setting...</b></p>
                <p style='margin: 5px 0 0 0;'>Please wait while the model is reloading...</p>
            </div>
            """
            yield (
                processing_status,
                gr.update(interactive=False),  # add_file disabled
                gr.update(interactive=False),  # kb_upload_btn disabled
                gr.update(interactive=False),  # delete_btn disabled
                gr.update(interactive=False)   # send_btn disabled
            )

            # Perform reload
            reload_model(new_speed)
            updated_status = get_current_status()

            # Unlock operations
            settings_changing = False

            # Second yield: Re-enable buttons and show updated status
            yield (
                updated_status,
                gr.update(interactive=True),  # add_file enabled
                gr.update(interactive=True),  # kb_upload_btn enabled
                gr.update(interactive=True if source_dropdown.value else False),  # delete_btn depends on selection
                gr.update(interactive=True if msg_input.value and msg_input.value.strip() else False)  # send_btn depends on input
            )

        def handle_mode_change(new_mode):
            """Handle mode dropdown change - no reload needed."""
            global settings_changing

            # Lock all operations
            settings_changing = True

            # First yield: Disable all buttons and show processing status
            processing_status = """
            <div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>
                <p style='margin: 0;'><b>⚙️ Changing Mode Setting...</b></p>
                <p style='margin: 5px 0 0 0;'>Updating mode configuration...</p>
            </div>
            """
            yield (
                processing_status,
                gr.update(interactive=False),  # add_file disabled
                gr.update(interactive=False),  # kb_upload_btn disabled
                gr.update(interactive=False),  # delete_btn disabled
                gr.update(interactive=False)   # send_btn disabled
            )

            # Perform mode change
            update_mode(new_mode)
            updated_status = get_current_status()

            # Unlock operations
            settings_changing = False

            # Second yield: Re-enable buttons and show updated status
            yield (
                updated_status,
                gr.update(interactive=True),  # add_file enabled
                gr.update(interactive=True),  # kb_upload_btn enabled
                gr.update(interactive=True if source_dropdown.value else False),  # delete_btn depends on selection
                gr.update(interactive=True if msg_input.value and msg_input.value.strip() else False)  # send_btn depends on input
            )

        def handle_msg_input_change(msg, history):
            """Handle message input change - enable/disable send button based on input and processing state."""
            # Disable if no input
            if not msg or not msg.strip():
                return gr.update(interactive=False)

            # Check if settings are being changed
            if settings_changing:
                return gr.update(interactive=False)  # Keep disabled during settings change

            # Check if KB is currently processing
            if kb_file_processing:
                return gr.update(interactive=False)  # Keep disabled during KB processing

            # Check if LLM is currently processing (last message is "Thinking...")
            if history and len(history) > 0:
                last_message = history[-1]
                if last_message.get('role') == 'assistant' and '💭 Thinking...' in last_message.get('content', ''):
                    return gr.update(interactive=False)  # Keep disabled during LLM processing

            # Has input and no processing - enable
            return gr.update(interactive=True)

        # Connect events - chat
        msg_input.submit(
            chat_respond,
            [msg_input, chatbot],
            [msg_input, msg_input, chatbot, chat_file_status, remove_file_btn, send_btn,
             upload_file_btn, speed_dropdown, mode_dropdown, delete_btn, refresh_kb_btn,
             source_dropdown, add_file, kb_upload_btn]
        )
        send_btn.click(
            chat_respond,
            [msg_input, chatbot],
            [msg_input, msg_input, chatbot, chat_file_status, remove_file_btn, send_btn,
             upload_file_btn, speed_dropdown, mode_dropdown, delete_btn, refresh_kb_btn,
             source_dropdown, add_file, kb_upload_btn]
        )
        msg_input.change(handle_msg_input_change, [msg_input, chatbot], [send_btn])

        # File upload handlers - left side (chat)
        upload_file_btn.upload(
            handle_chat_file_upload,
            inputs=[upload_file_btn],
            outputs=[chat_file_status, remove_file_btn]
        )
        remove_file_btn.click(
            handle_remove_chat_file,
            outputs=[chat_file_status, remove_file_btn]
        )

        # File upload handlers - right side (KB)
        # Both drag-and-drop (add_file) and button click (kb_upload_btn) trigger the same handler
        add_file.upload(
            handle_kb_file_upload,
            inputs=[add_file],
            outputs=[kb_file_status, kb_display, source_dropdown, add_file, kb_upload_btn, send_btn]
        )
        kb_upload_btn.upload(
            handle_kb_file_upload,
            inputs=[kb_upload_btn],
            outputs=[kb_file_status, kb_display, source_dropdown, add_file, kb_upload_btn, send_btn]
        )

        # Settings change handlers
        speed_dropdown.change(
            handle_speed_change,
            inputs=[speed_dropdown],
            outputs=[status_display, add_file, kb_upload_btn, delete_btn, send_btn]
        )
        mode_dropdown.change(
            handle_mode_change,
            inputs=[mode_dropdown],
            outputs=[status_display, add_file, kb_upload_btn, delete_btn, send_btn]
        )

        # Knowledge base handlers
        refresh_kb_btn.click(handle_refresh_kb, outputs=[kb_display, source_dropdown, delete_btn])
        delete_btn.click(handle_delete_source, [source_dropdown], [kb_display, source_dropdown, delete_btn])
        source_dropdown.change(handle_source_dropdown_change, [source_dropdown], [delete_btn])

        # Auto-refresh knowledge base on page load
        demo.load(handle_refresh_kb, outputs=[kb_display, source_dropdown, delete_btn])

        # Custom JavaScript for Enter to submit, hide empty containers, and manage chatbot buttons
        demo.load(None, None, None, js="""
        function() {
            setTimeout(function() {
                // Enter to submit (only if send button is enabled)
                const textarea = document.querySelector('#msg_input textarea');
                if (textarea) {
                    textarea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            const submitBtn = document.querySelector('#send_btn');
                            // Only click if button exists and is not disabled
                            if (submitBtn && !submitBtn.disabled && !submitBtn.classList.contains('disabled')) {
                                submitBtn.click();
                            }
                        }
                    });
                }

                // Hide empty HTML containers
                function hideEmptyContainers() {
                    const htmlContainers = document.querySelectorAll('.html-container');
                    htmlContainers.forEach(function(container) {
                        const prose = container.querySelector('.prose');
                        if (prose && prose.innerHTML.trim() === '') {
                            container.style.display = 'none';
                        } else if (prose && prose.innerHTML.trim() !== '') {
                            container.style.display = '';
                        }
                    });
                }

                // Run immediately and observe changes
                hideEmptyContainers();

                const observer = new MutationObserver(function() {
                    hideEmptyContainers();
                });

                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });

                // Monitor chatbot for "Thinking..." message and disable/enable clear button
                const chatbotObserver = new MutationObserver(function() {
                    const chatbot = document.querySelector('#chatbot');
                    if (!chatbot) return;

                    // Find all buttons in chatbot
                    const chatbotButtons = chatbot.querySelectorAll('button');

                    // Check if last message contains "Thinking..."
                    const messages = chatbot.querySelectorAll('.message');
                    const lastMessage = messages[messages.length - 1];
                    const isThinking = lastMessage && lastMessage.textContent.includes('💭 Thinking...');

                    // Disable/enable only clear button, keep copy buttons enabled
                    chatbotButtons.forEach(function(btn) {
                        // Check if this is a copy button (has copy icon or title)
                        const isCopyButton = btn.querySelector('svg') &&
                            (btn.title && btn.title.toLowerCase().includes('copy')) ||
                            btn.getAttribute('aria-label') && btn.getAttribute('aria-label').toLowerCase().includes('copy') ||
                            btn.classList.contains('copy-button');

                        // Skip copy buttons - keep them enabled
                        if (isCopyButton) {
                            return;
                        }

                        // Disable/enable other buttons (like clear)
                        if (isThinking) {
                            btn.disabled = true;
                            btn.style.pointerEvents = 'none';
                            btn.style.opacity = '0.5';
                            btn.blur(); // Remove focus to eliminate blue outline
                        } else {
                            btn.disabled = false;
                            btn.style.pointerEvents = '';
                            btn.style.opacity = '';
                        }
                    });
                });

                chatbotObserver.observe(document.body, {
                    childList: true,
                    subtree: true
                });

                // Remove focus and blue outlines from disabled elements
                const disabledObserver = new MutationObserver(function() {
                    // Target all potentially disabled input elements
                    const disabledElements = document.querySelectorAll('input:disabled, textarea:disabled, select:disabled, button:disabled, .disabled');
                    disabledElements.forEach(function(el) {
                        el.blur(); // Remove focus
                        el.style.outline = 'none';
                        el.style.boxShadow = 'none';
                        el.style.borderColor = 'transparent';

                        // Also handle parent containers (Gradio wraps inputs in .block divs)
                        let parent = el.closest('.block');
                        if (parent) {
                            parent.style.borderColor = 'transparent';
                            parent.style.outline = 'none';
                            parent.style.boxShadow = 'none';
                        }
                    });

                    // Specifically target generating state containers
                    const generatingElements = document.querySelectorAll('.generating');
                    generatingElements.forEach(function(el) {
                        el.style.borderColor = 'transparent';
                    });
                });

                disabledObserver.observe(document.body, {
                    attributes: true,
                    attributeFilter: ['disabled', 'class'],
                    childList: true,
                    subtree: true
                });
            }, 1000);
        }
        """)


    return demo

# =============================================================================
# Main entry point
# =============================================================================

def main():
    """Start Gradio web server."""
    print("="*60)
    print("RAG CVE Validation System - Web UI")
    print("="*60)

    # Initialize system
    initialize_system(speed_level=DEFAULT_SPEED)

    # Create and launch interface
    demo = create_interface()

    print(f"\nLaunching web UI...")
    print(f"  Server: {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT}")
    print(f"  Share: {GRADIO_SHARE}")

    # Print database status
    print(f"\nKnowledge Base Status:")
    try:
        stats = chroma_manager.get_stats()
        print(f"  Total documents: {stats['total_docs']}")
        print(f"  Sources: {len(stats['sources'])}")
        for name, info in list(stats['sources'].items())[:5]:
            print(f"    - {name}: {info['count']} chunks")
    except Exception as e:
        print(f"  [WARNING] Error loading stats: {e}")

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        inbrowser=True  # Auto-open browser
    )

if __name__ == "__main__":
    main()
