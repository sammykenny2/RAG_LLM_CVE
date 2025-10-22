"""
Gradio Web UI (Phase 1: Pure Python)
Claude Projects-style interface for RAG CVE validation system.

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
    TEMP_UPLOAD_DIR,
    CVE_V5_PATH,
    CVE_V4_PATH,
    ENABLE_SESSION_AUTO_EMBED
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
kb_uploaded_file = None  # Right side: for knowledge base
kb_file_uploading = False  # Right side upload status

# Upload directories
TEMP_UPLOAD_DIR = Path("temp_uploads")  # For chat files (temporary, no embeddings)
KB_UPLOAD_DIR = Path("kb_uploads")  # For KB files (backup, generate embeddings)

# Ensure upload directories exist
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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

    # Initialize SessionManager for multi-file context
    if session_manager is None:
        session_manager = SessionManager(session_id=session_id)
        print(f"✅ SessionManager initialized (session_id={session_id[:8]}...)")

    # Initialize RAG with session_manager
    rag_system = PureRAG(session_manager=session_manager)
    rag_system.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)

    # Get references for knowledge base management
    chroma_manager = rag_system.chroma
    embedding_model = rag_system.embedder

    # Update current speed
    current_speed = speed_level

    print(f"✅ RAG system ready (speed={current_speed})")
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
    - summarize: 總結, 摘要, 概括, 整理, 內容, 講什麼, summary, summarize
    - validate: 驗證, 檢查, 核實, validate, verify, check

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
        '總結', '摘要', '概括', '概要', '整理', '內容', '講什麼', '说什么',
        '主要內容', '主要内容', '重點', '重点', '大意',
        # English
        'summarize', 'summary', 'summarise', 'what is this', 'what does',
        'content', 'about', 'main point', 'key point', 'overview'
    ]

    # Validate intent keywords (Chinese + English)
    validate_keywords = [
        # Chinese
        '驗證', '验证', '檢查', '检查', '核實', '核实', '確認', '确认',
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
        tuple: (empty string for input clear, updated history, empty string to clear file status, button update to disable)
    """
    global chat_uploaded_file, chat_file_uploading

    if not message.strip():
        yield "", history, "", gr.update()
        return

    # Check if file is still uploading
    if chat_file_uploading:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⏳ Please wait, file is still uploading..."})
        yield "", history, "", gr.update()
        return

    # Build user message with file attachment if present
    user_content = message
    if chat_uploaded_file:
        file_name = Path(chat_uploaded_file).name
        user_content = f"{message}\n\n📎 **Attached:** {file_name}"

    # Immediately show user message with "Thinking..." placeholder
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": "💭 Thinking..."})
    yield "", history, "", gr.update()

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
            response = rag_system.query(
                question=message,
                include_history=True,
                max_tokens=1000
            )

        # Delete uploaded file from disk if exists
        if chat_uploaded_file and os.path.exists(chat_uploaded_file):
            try:
                os.remove(chat_uploaded_file)
            except Exception as e:
                print(f"Warning: Could not delete file {chat_uploaded_file}: {e}")

        # Clear uploaded file reference after sending
        chat_uploaded_file = None

        # Update history with actual response, clear file status, and disable remove button
        history[-1]["content"] = response
        yield "", history, "", gr.update(interactive=False)

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

        yield "", history, "", gr.update(interactive=False)

def handle_chat_file_upload(file):
    """
    Handle file upload for chat (left side).

    Args:
        file: Gradio File object (path string)

    Returns:
        tuple: (HTML status display, button update to enable)
    """
    global chat_uploaded_file, chat_file_uploading
    import shutil
    import time

    if file is None:
        return "", gr.update(interactive=False)

    try:
        # Set uploading status
        chat_file_uploading = True

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
                print(f"✅ File added to session: {file_name} ({file_info['chunks']} chunks)")
            except Exception as e:
                print(f"⚠️ Failed to add file to session: {e}")

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
        top_k = 3
        validation_tokens = 256
    else:  # full mode
        max_pages = None
        top_k = 5
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
            summary = rag_system.summarize_report(text)  # Let RAG class use .env config
            return f"📝 Summary:\n\n{summary}"

        elif action == 'validate':
            # Process report and validate CVE usage
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

        elif action == 'qa':
            # Answer question about report (uses .env QA_* configuration)
            if not question:
                return "⚠️ No question provided for Q&A"

            pdf_processor = PDFProcessor()
            text = pdf_processor.extract_text(file_path, max_pages=max_pages)
            answer = rag_system.answer_question_about_report(text, question)  # Let RAG class use .env config
            return f"💬 Answer:\n\n{answer}"

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
        tuple: (status_html, updated_kb_display, updated_dropdown_choices)
    """
    global kb_uploaded_file, kb_file_uploading
    import shutil

    if file is None:
        return "", format_kb_display(), get_source_names()

    try:
        # Set uploading status
        kb_file_uploading = True

        # Get source file path
        source_path = Path(file)
        file_name = source_path.name

        # Copy file to kb_uploads directory
        dest_path = KB_UPLOAD_DIR / file_name
        shutil.copy2(source_path, dest_path)

        # Update global state
        kb_uploaded_file = str(dest_path)

        # Immediately process the file
        print(f"Adding {file_name} to knowledge base...")

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
                'source_name': file_name,
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

        # Clear uploaded file after processing
        kb_uploaded_file = None
        kb_file_uploading = False

        print(f"✅ Added {n_added} chunks from {file_name} to knowledge base")

        # Return empty status (hide immediately), updated KB display, dropdown, and clear file input
        return "", format_kb_display(), gr.update(choices=get_source_names()), None

    except Exception as e:
        kb_file_uploading = False
        print(f"❌ Error adding {file.name if file else 'file'} to knowledge base: {str(e)}")

        # Return empty status even on error (hide immediately), and clear file input
        return "", format_kb_display(), gr.update(choices=get_source_names()), None

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

    Args:
        source_name: Name of source to delete

    Returns:
        tuple: (status_message, updated_kb_display, updated_dropdown_choices)
    """
    if not source_name:
        return "⚠️ No source selected", format_kb_display(), get_source_names()

    try:
        # Delete the source
        n_deleted = chroma_manager.delete_by_source(source_name)

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

    with gr.Blocks(title="RAG CVE Validation System", theme=gr.themes.Soft()) as demo:
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
                    type='messages'  # Use OpenAI-style message format
                )

                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about CVEs, security reports, or upload a file for validation...",
                    lines=2,
                    show_label=False,
                    elem_id="msg_input"
                )

                # Action buttons row (Claude Projects style)
                with gr.Row():
                    upload_file_btn = gr.UploadButton(
                        "➕ Add File",
                        file_types=[".pdf"],
                        file_count="single",
                        size="sm",
                        scale=1
                    )
                    remove_file_btn = gr.Button("🗑️", size="sm", scale=0, min_width=40, interactive=False)
                    send_btn = gr.Button("Send →", size="sm", scale=1, variant="primary", elem_id="send_btn")
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
                        interactive=True
                    )
                    delete_btn = gr.Button("🗑️ Delete Selected Source", size="sm", variant="stop")

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
            return updated_display, gr.update(choices=updated_sources)

        def handle_delete_source(source_name):
            """Handle delete source button click."""
            status, updated_display, updated_sources = delete_source(source_name)
            print(status)  # Print to console instead of UI
            return updated_display, gr.update(choices=updated_sources, value=None)

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

        # Connect events - chat
        msg_input.submit(chat_respond, [msg_input, chatbot], [msg_input, chatbot, chat_file_status, remove_file_btn])
        send_btn.click(chat_respond, [msg_input, chatbot], [msg_input, chatbot, chat_file_status, remove_file_btn])

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
            outputs=[kb_file_status, kb_display, source_dropdown, add_file]
        )
        kb_upload_btn.upload(
            handle_kb_file_upload,
            inputs=[kb_upload_btn],
            outputs=[kb_file_status, kb_display, source_dropdown, add_file]
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

        # Knowledge base handlers
        refresh_kb_btn.click(handle_refresh_kb, outputs=[kb_display, source_dropdown])
        delete_btn.click(handle_delete_source, [source_dropdown], [kb_display, source_dropdown])

        # Auto-refresh knowledge base on page load
        demo.load(handle_refresh_kb, outputs=[kb_display, source_dropdown])

        # Custom JavaScript for Enter to submit and hide empty containers
        demo.load(None, None, None, js="""
        function() {
            setTimeout(function() {
                // Enter to submit
                const textarea = document.querySelector('#msg_input textarea');
                if (textarea) {
                    textarea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            const submitBtn = document.querySelector('#send_btn');
                            if (submitBtn) submitBtn.click();
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
        print(f"  ⚠️ Error loading stats: {e}")

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        inbrowser=True  # Auto-open browser
    )

if __name__ == "__main__":
    main()
