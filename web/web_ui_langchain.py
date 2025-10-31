"""
Gradio Web UI (Phase 2: LangChain)
Two-column interface using LangChain chains and memory.

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

# Import LangChain RAG
from rag.langchain_impl import LangChainRAG
from core.session_manager import SessionManager
import re
import numpy as np
import uuid

# Import configuration
from config import (
    GRADIO_SERVER_PORT_LANGCHAIN,
    GRADIO_SHARE,
    GRADIO_SERVER_NAME,
    DEFAULT_SPEED,
    DEFAULT_MODE,
    DEFAULT_SCHEMA,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_PRECISION,
    ENABLE_SESSION_AUTO_EMBED,
    RETRIEVAL_TOP_K
)

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
session_manager = None  # SessionManager for multi-file context
session_id = str(uuid.uuid4())  # Unique session ID

# Track current settings
current_speed = DEFAULT_SPEED
current_mode = DEFAULT_MODE

# Track uploaded files
chat_uploaded_file = None  # Left side: for chat validation (current file display)
chat_file_uploading = False  # Left side upload status

# Upload directories
TEMP_UPLOAD_DIR = Path("temp_uploads")  # For chat files (temporary, no embeddings)
KB_UPLOAD_DIR = Path("kb_uploads")  # For KB files (backup, generate embeddings)

# Ensure upload directories exist and clean up old temp files
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Clean up leftover temp files from previous sessions (avoid disk space waste)
if TEMP_UPLOAD_DIR.exists():
    import glob
    temp_files = list(TEMP_UPLOAD_DIR.glob("*"))
    if temp_files:
        print(f"[INFO] Cleaning up {len(temp_files)} leftover temp file(s) from previous session...")
        for temp_file in temp_files:
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except Exception as e:
                print(f"[WARNING] Could not delete {temp_file.name}: {e}")

def initialize_system(speed_level: str = DEFAULT_SPEED, force_reload: bool = False):
    """Initialize LangChain RAG system based on speed level."""
    global rag_system, current_speed, session_manager

    if rag_system is not None and not force_reload:
        return f"⚠️ System already initialized with speed={current_speed}"

    # Clean up existing model if reloading
    if force_reload and rag_system is not None:
        print(f"Reloading model with new speed: {speed_level}")
        rag_system.cleanup()
        rag_system = None

    print("Initializing LangChain RAG system for web UI...")

    # Speed level configuration
    use_fp16 = speed_level in ['fast', 'fastest']
    use_sdpa = speed_level == 'fastest'

    # Initialize SessionManager for multi-file context (only if enabled)
    if session_manager is None and ENABLE_SESSION_AUTO_EMBED:
        session_manager = SessionManager(session_id=session_id)
        print(f"[OK] SessionManager initialized (session_id={session_id[:8]}...)")
    elif not ENABLE_SESSION_AUTO_EMBED:
        print("[INFO] SessionManager disabled (ENABLE_SESSION_AUTO_EMBED=False)")

    # Initialize LangChain RAG with session_manager
    rag_system = LangChainRAG(session_manager=session_manager)
    rag_system.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)

    # Update current speed
    current_speed = speed_level

    print(f"[OK] LangChain RAG system ready (speed={current_speed})")
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
    from config import LLAMA_MODEL_NAME

    model_name = 'Not loaded'
    if rag_system and rag_system._initialized:
        model_name = LLAMA_MODEL_NAME

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
        'summary_zh', 'summary_zh', 'summary_zh', 'summary_zh', 'summary_zh', 'content_zh', 'whatisit_zh', 'whatisit_zh',
        'maincontent_zh', 'maincontent_zh', 'keypoint_zh', 'keypoint_zh', 'gist_zh',
        # English
        'summarize', 'summary', 'summarise', 'what is this', 'what does',
        'content', 'about', 'main point', 'key point', 'overview'
    ]

    # Validate intent keywords (Chinese + English)
    validate_keywords = [
        # Chinese
        'validate_zh', 'validate_zh', 'check_zh', 'check_zh', 'verify_zh', 'verify_zh', 'confirm_zh', 'confirm_zh',
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
    Handle chat messages with LangChain's automatic memory management.

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

    # Check if file is still uploading
    if chat_file_uploading:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⏳ Please wait, file is still uploading..."})
        yield "", gr.update(), history, "", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # Build user message with file attachment if present
    user_content = message
    if chat_uploaded_file:
        file_name = Path(chat_uploaded_file).name
        user_content = f"{message}\n\n📎 **Attached:** {file_name}"

    # Immediately show user message with "Thinking..." placeholder and disable all UI elements (including input box)
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": "💭 Thinking..."})
    yield (
        "",  # msg_input (clear)
        gr.update(interactive=False),  # msg_input (disable - prevent typing during processing)
        history,  # history (with thinking message)
        "",  # chat_file_status (clear)
        gr.update(interactive=False),  # remove_file_btn (disable)
        gr.update(interactive=False),  # send_btn (disable)
        gr.update(interactive=False),  # upload_file_btn (disable)
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
        intent = detect_user_intent(message, has_file=bool(chat_uploaded_file))

        # Handle file-specific actions based on detected intent
        if chat_uploaded_file:
            if intent == 'summarize':
                response = process_uploaded_report(chat_uploaded_file, action='summarize', mode=current_mode)
            elif intent == 'validate':
                response = process_uploaded_report(chat_uploaded_file, action='validate', schema=DEFAULT_SCHEMA, mode=current_mode)
            else:
                # Default to Q&A on file content (Plan A approach)
                response = process_uploaded_report(chat_uploaded_file, action='qa', question=message, mode=current_mode)
        else:
            # Normal RAG query (no file attached)
            response = rag_system.query(question=message)

        # Delete uploaded file from disk if exists
        if chat_uploaded_file and os.path.exists(chat_uploaded_file):
            try:
                os.remove(chat_uploaded_file)
            except Exception as e:
                print(f"Warning: Could not delete file {chat_uploaded_file}: {e}")

        # Clear uploaded file reference after sending
        chat_uploaded_file = None

        # Update history with actual response and re-enable all UI elements
        history[-1]["content"] = response
        yield (
            "",  # msg_input (clear)
            gr.update(interactive=True),   # msg_input (enable - allow typing again)
            history,  # history (with response)
            "",  # chat_file_status (clear)
            gr.update(interactive=False),  # remove_file_btn (disable - no file)
            gr.update(interactive=False),  # send_btn (disable - input empty)
            gr.update(interactive=True),   # upload_file_btn (enable)
            gr.update(interactive=True),   # speed_dropdown (enable)
            gr.update(interactive=True),   # mode_dropdown (enable)
            gr.update(interactive=False),  # delete_btn (disable - safe default)
            gr.update(interactive=True),   # refresh_kb_btn (enable)
            gr.update(interactive=True),   # source_dropdown (enable)
            gr.update(interactive=True),   # add_file (enable)
            gr.update(interactive=True)    # kb_upload_btn (enable)
        )

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1]["content"] = error_msg

        # Delete uploaded file from disk if exists (even on error)
        import os
        if chat_uploaded_file and os.path.exists(chat_uploaded_file):
            try:
                os.remove(chat_uploaded_file)
            except Exception as e:
                print(f"Warning: Could not delete file {chat_uploaded_file}: {e}")

        # Clear uploaded file reference even on error
        chat_uploaded_file = None

        yield (
            "",  # msg_input (clear)
            gr.update(interactive=True),   # msg_input (enable - allow typing again)
            history,  # history (with error)
            "",  # chat_file_status (clear)
            gr.update(interactive=False),  # remove_file_btn (disable - no file)
            gr.update(interactive=False),  # send_btn (disable - input empty)
            gr.update(interactive=True),   # upload_file_btn (enable)
            gr.update(interactive=True),   # speed_dropdown (enable)
            gr.update(interactive=True),   # mode_dropdown (enable)
            gr.update(interactive=False),  # delete_btn (disable - safe default)
            gr.update(interactive=True),   # refresh_kb_btn (enable)
            gr.update(interactive=True),   # source_dropdown (enable)
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
        file_name = source_path.name

        # Show uploading status
        uploading_html = f"""
        <div style='padding: 10px; background-color: #fff3cd; border-radius: 5px; border-left: 4px solid #ffc107;'>
            <p style='margin: 0;'><b>📄 {file_name}</b></p>
            <p style='margin: 5px 0 0 0; color: #856404;'>🔄 Uploading...</p>
        </div>
        """

        # Simulate upload delay (for demo purposes, can remove in production)
        time.sleep(0.5)

        # Copy file to temp_upload directory
        dest_path = TEMP_UPLOAD_DIR / file_name
        shutil.copy2(source_path, dest_path)

        # Update global state
        chat_uploaded_file = str(dest_path)

        # Add file to SessionManager for multi-file context (if enabled)
        if session_manager and ENABLE_SESSION_AUTO_EMBED:
            try:
                file_info = session_manager.add_file(str(dest_path))
                print(f"[OK] File added to session: {file_name} ({file_info['chunks']} chunks)")
            except Exception as e:
                print(f"[WARNING] Failed to add file to session: {e}")

        chat_file_uploading = False

        # Show success status with file count
        file_count = len(session_manager.files) if (session_manager and ENABLE_SESSION_AUTO_EMBED) else 1
        success_html = f"""
        <div style='padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;'>
            <p style='margin: 0;'><b>📄 {file_name}</b></p>
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

    Returns:
        tuple: (Empty HTML to clear status display, button update to disable)
    """
    global chat_uploaded_file, chat_file_uploading
    import os

    # Delete file if exists
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
            from core.pdf_processor import PDFProcessor
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
                validation = rag_system.validate_cve_usage(report_text, cve_descriptions, max_tokens=validation_tokens)
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

            from core.pdf_processor import PDFProcessor
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

    Args:
        file: Gradio File object (path string)

    Returns:
        tuple: (status_html, updated_kb_display, updated_dropdown_choices, cleared_file_input)
    """
    import shutil

    if file is None:
        return "", format_kb_display(), gr.update(choices=get_source_names()), None

    try:
        # Get source file path
        source_path = Path(file)
        file_name = source_path.name

        # Copy file to kb_uploads directory
        dest_path = KB_UPLOAD_DIR / file_name
        shutil.copy2(source_path, dest_path)

        # Immediately process the file
        print(f"Adding {file_name} to knowledge base...")

        # Extract text
        from core.pdf_processor import PDFProcessor
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

        # Prepare metadata
        metadatas = [
            {
                'source_type': 'pdf',
                'source_name': file_name,
                'added_date': datetime.now().isoformat(),
                'chunk_index': i,
                'precision': EMBEDDING_PRECISION
            }
            for i in range(len(chunks))
        ]

        # Add to knowledge base (LangChain handles embedding generation)
        rag_system.add_document_to_kb(texts=chunks, metadatas=metadatas)

        print(f"[OK] Added {len(chunks)} chunks from {file_name} to knowledge base")

        # Return empty status (hide immediately), updated KB display, dropdown, and clear file input
        return "", format_kb_display(), gr.update(choices=get_source_names()), None

    except Exception as e:
        print(f"[ERROR] Error adding {file.name if file else 'file'} to knowledge base: {str(e)}")

        # Return empty status even on error (hide immediately), and clear file input
        return "", format_kb_display(), gr.update(choices=get_source_names()), None

def add_pdf_to_kb(file, source_name: str = None) -> str:
    """
    Add PDF to knowledge base using LangChain.

    Args:
        file: Gradio File object or file path string
        source_name: Optional source name

    Returns:
        str: Result message
    """
    if file is None:
        return "[WARNING] No file selected"

    try:
        from core.pdf_processor import PDFProcessor

        # Handle both string paths and Gradio File objects
        if isinstance(file, str):
            file_path = Path(file)
        else:
            file_path = Path(file.name)

        source_name = source_name or file_path.name

        print(f"Adding {source_name} to knowledge base...")

        # Extract text
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text(file_path)

        # Split into sentences (supports both English and Chinese)
        sentences = split_sentences(text)

        # Create chunks
        chunks = []
        for i in range(0, len(sentences), CHUNK_SIZE):
            chunk = "".join(sentences[i:i+CHUNK_SIZE]).strip()
            if chunk:
                chunks.append(chunk)

        # Prepare metadata
        metadatas = [
            {
                'source_type': 'pdf',
                'source_name': source_name,
                'added_date': datetime.now().isoformat(),
                'chunk_index': i,
                'precision': EMBEDDING_PRECISION
            }
            for i in range(len(chunks))
        ]

        # Add to knowledge base (LangChain handles embedding generation)
        rag_system.add_document_to_kb(texts=chunks, metadatas=metadatas)

        return f"[OK] Added {len(chunks)} chunks from '{source_name}' to knowledge base"

    except Exception as e:
        return f"[ERROR] Error adding to KB: {str(e)}"

def format_kb_display() -> str:
    """
    Format knowledge base sources for display.

    Returns:
        str: Formatted HTML string
    """
    try:
        stats = rag_system.get_kb_stats()

        if not stats:
            return "<p>[WARNING] No statistics available</p>"

        sources = stats.get('sources', {})

        # Build HTML with collapsible sources list
        html = f"""
        <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
            <h4>📊 Statistics</h4>
            <p><b>Total documents:</b> {stats.get('total_docs', 0)}</p>
            <p><b>By type:</b> {dict(stats.get('by_source_type', {}))}</p>
            <br>
            <details>
                <summary style='cursor: pointer; font-weight: bold; font-size: 1.1em;'>
                    📚 Sources ({len(sources)})
                </summary>
                <ul style='list-style: none; padding-left: 0; margin-top: 10px;'>
        """

        for source_name, info in sources.items():
            icon = "📄" if info['type'] == 'pdf' else "🔖"
            date = info['added_date'][:10]
            html += f"<li>{icon} <b>{source_name}</b> ({info['count']} chunks, added {date})</li>"

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
        stats = rag_system.get_kb_stats()
        if not stats:
            return []
        sources = stats.get('sources', {})
        return list(sources.keys())
    except Exception as e:
        print(f"Error getting source names: {e}")
        return []

def delete_source(source_name: str) -> tuple:
    """
    Delete a single source from knowledge base.

    Args:
        source_name: Name of source to delete

    Returns:
        tuple: (status_message, updated_kb_display, updated_dropdown_choices)
    """
    if not source_name:
        return "⚠️ No source selected", format_kb_display(), get_source_names()

    try:
        # Delete the source using chroma_manager
        n_deleted = rag_system.chroma_manager.delete_by_source(source_name)

        if n_deleted > 0:
            status = f"✅ Deleted {n_deleted} chunks from '{source_name}'"
        else:
            status = f"⚠️ Source '{source_name}' not found"

        # Get updated displays
        updated_display = format_kb_display()
        updated_choices = get_source_names()

        return status, updated_display, updated_choices

    except Exception as e:
        return f"❌ Error deleting source: {str(e)}", format_kb_display(), get_source_names()

# =============================================================================
# Gradio interface
# =============================================================================

def create_interface():
    """Create and configure Gradio interface."""

    with gr.Blocks(title="RAG CVE Validation System (LangChain)", theme=gr.themes.Soft(), css="""
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
        gr.Markdown("# 🛡️ RAG CVE Validation System (LangChain)")
        gr.Markdown("Conversational AI with LangChain automatic memory management")

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
            reload_model(new_speed)
            updated_status = get_current_status()
            return updated_status

        def handle_mode_change(new_mode):
            """Handle mode dropdown change - no reload needed."""
            update_mode(new_mode)
            updated_status = get_current_status()
            return updated_status

        def handle_msg_input_change(msg):
            """Handle message input change - enable/disable send button based on input."""
            # Enable send button only if message has content
            if msg and msg.strip():
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)

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
        msg_input.change(handle_msg_input_change, [msg_input], [send_btn])

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

        # Settings change handlers
        speed_dropdown.change(
            handle_speed_change,
            inputs=[speed_dropdown],
            outputs=[status_display]
        )
        mode_dropdown.change(
            handle_mode_change,
            inputs=[mode_dropdown],
            outputs=[status_display]
        )

        # File upload handlers - right side (KB)
        # Both drag-and-drop (add_file) and button click (kb_upload_btn) trigger the same handler
        add_file.upload(
            handle_kb_file_upload,
            inputs=[add_file],
            outputs=[kb_file_status, kb_display, source_dropdown, add_file]
        )
        kb_upload_btn.upload(
            handle_kb_file_upload,
            inputs=[kb_upload_btn],
            outputs=[kb_file_status, kb_display, source_dropdown, add_file]
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
    print("RAG CVE Validation System - Web UI (LangChain)")
    print("="*60)

    # Initialize system
    initialize_system(speed_level=DEFAULT_SPEED)

    # Create and launch interface
    demo = create_interface()

    print(f"\nLaunching web UI (LangChain version)...")
    print(f"  Server: {GRADIO_SERVER_NAME}:{GRADIO_SERVER_PORT_LANGCHAIN}")
    print(f"  Share: {GRADIO_SHARE}")

    # Print database status
    print(f"\nKnowledge Base Status:")
    try:
        stats = rag_system.get_kb_stats()
        print(f"  Total documents: {stats.get('total_docs', 0)}")
        print(f"  Sources: {len(stats.get('sources', {}))}")
        for name, info in list(stats.get('sources', {}).items())[:5]:
            print(f"    - {name}: {info['count']} chunks")
    except Exception as e:
        print(f"  [WARNING] Error loading stats: {e}")

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT_LANGCHAIN,
        share=GRADIO_SHARE,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
