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
import re
import numpy as np

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
    TEMP_UPLOAD_DIR
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
chroma_manager = None
embedding_model = None

# Track current settings
current_speed = DEFAULT_SPEED
current_mode = DEFAULT_MODE

def initialize_system(speed_level: str = DEFAULT_SPEED, force_reload: bool = False):
    """Initialize RAG system based on speed level."""
    global rag_system, chroma_manager, embedding_model, current_speed

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

    # Initialize RAG
    rag_system = PureRAG()
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

def chat_respond(message: str, history: list):
    """
    Handle chat messages with RAG context retrieval.

    Generator function that yields intermediate states for better UX.

    Args:
        message: User message
        history: Gradio chat history (messages format: list of dicts with 'role' and 'content')

    Yields:
        tuple: (empty string for input clear, updated history)
    """
    if not message.strip():
        yield "", history
        return

    # Immediately show user message with "Thinking..." placeholder
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "💭 Thinking..."})
    yield "", history

    try:
        # Query RAG system
        response = rag_system.query(
            question=message,
            include_history=True,
            max_tokens=512
        )

        # Update history with actual response
        history[-1]["content"] = response
        yield "", history

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history[-1]["content"] = error_msg
        yield "", history

def upload_for_validation(file) -> str:
    """
    Handle file upload for validation.
    Returns instructions for user to select action.

    Args:
        file: Gradio File object

    Returns:
        str: Instructions message
    """
    if file is None:
        return "⚠️ No file selected"

    try:
        file_path = Path(file.name)
        return (
            f"📄 Uploaded: {file_path.name}\n\n"
            "Please select what you want to do:\n"
            "- Type 'summarize' to get a summary\n"
            "- Type 'validate' to validate CVE usage\n"
            "- Type 'add to kb' to add to knowledge base\n"
            "- Or ask any question about the report"
        )
    except Exception as e:
        return f"❌ Upload error: {str(e)}"

def process_uploaded_report(
    file,
    action: str,
    schema: str = DEFAULT_SCHEMA,
    mode: str = None
) -> str:
    """
    Process uploaded report based on user action.

    Args:
        file: Gradio File object
        action: 'summarize', 'validate', or 'add'
        schema: CVE schema to use
        mode: Processing mode (demo/full), uses global current_mode if None

    Returns:
        str: Processing result
    """
    if file is None:
        return "⚠️ No file uploaded"

    # Use global mode if not specified
    mode = mode or current_mode

    # Mode-specific settings
    if mode == 'demo':
        max_tokens = 256
        max_pages = 10
        top_k = 3
    else:  # full mode
        max_tokens = 700
        max_pages = None
        top_k = 5

    try:
        file_path = Path(file.name)

        if action == 'summarize':
            # Extract text and summarize
            pdf_processor = PDFProcessor()
            text = pdf_processor.extract_text(file_path, max_pages=max_pages)
            summary = rag_system.summarize_report(text, max_tokens=max_tokens)
            mode_info = f" (mode={mode}, max_tokens={max_tokens})"
            return f"📝 Summary{mode_info}:\n\n{summary}"

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
                max_tokens=max_tokens
            )
            mode_info = f" (mode={mode}, max_tokens={max_tokens})"
            return f"✅ Validation Result{mode_info}:\n\n{validation}\n\n📋 Found CVEs: {', '.join(cves)}"

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

def add_pdf_to_kb(file, source_name: str = None) -> str:
    """
    Add PDF to knowledge base permanently.

    Args:
        file: Gradio File object
        source_name: Optional source name (defaults to filename)

    Returns:
        str: Result message
    """
    if file is None:
        return "⚠️ No file selected"

    try:
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
                'source_name': source_name,
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

        return f"✅ Added {n_added} chunks from '{source_name}' to knowledge base"

    except Exception as e:
        return f"❌ Error adding to KB: {str(e)}"

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

        # Build HTML
        html = f"<div style='padding: 10px;'>"
        html += f"<h4>📊 Statistics</h4>"
        html += f"<p><b>Total documents:</b> {stats['total_docs']}</p>"
        html += f"<p><b>By type:</b> {dict(stats['by_source_type'])}</p>"
        html += f"<br>"
        html += f"<h4>📚 Sources ({len(sources)})</h4>"
        html += f"<ul style='list-style: none; padding-left: 0;'>"

        for source in sources:
            icon = "📄" if source['type'] == 'pdf' else "🔖"
            date = source['added_date'][:10]  # Just date, not time
            html += f"<li>{icon} <b>{source['name']}</b> ({source['count']} chunks, added {date})</li>"

        html += f"</ul></div>"

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
                    upload_modal_btn = gr.Button("➕ Add File", size="sm", scale=1)
                    send_btn = gr.Button("Send →", size="sm", scale=1, variant="primary", elem_id="send_btn")
                    with gr.Column(scale=8):
                        pass  # Spacer

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
                        label="Speed Level",
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
                    gr.Markdown("**Remove Source:**")
                    source_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select source to remove",
                        interactive=True
                    )
                    delete_btn = gr.Button("🗑️ Delete Selected Source", size="sm", variant="stop")

                    # Add section
                    gr.Markdown("**Add New Source:**")
                    add_kb_file = gr.File(
                        label="Select PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    add_kb_btn = gr.Button("➕ Add to Knowledge Base", variant="primary")

        # Modal for file upload (Claude Projects style)
        with gr.Row(visible=False) as upload_modal:
            with gr.Column(scale=1):
                pass  # Left spacer
            with gr.Column(scale=10):
                with gr.Group():
                    gr.Markdown("### 📄 Upload File")

                    upload_file = gr.File(
                        label="Select PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                    gr.Markdown("**What would you like to do?**")

                    with gr.Row():
                        summarize_btn = gr.Button("📝 Summarize", variant="secondary")
                        validate_btn = gr.Button("✅ Validate CVEs", variant="secondary")
                        add_to_kb_btn_modal = gr.Button("📚 Add to Knowledge Base", variant="primary")

                    upload_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Row():
                        close_modal_btn = gr.Button("Close", size="sm")
            with gr.Column(scale=1):
                pass  # Right spacer

        # Event handlers
        def show_upload_modal():
            """Show the upload modal."""
            return gr.update(visible=True)

        def hide_upload_modal():
            """Hide the upload modal."""
            return gr.update(visible=False)

        def handle_summarize(file):
            """Handle summarize action from modal."""
            if file is None:
                return "⚠️ Please select a file first", gr.update(visible=True)
            result = process_uploaded_report(file, action='summarize', mode=current_mode)
            return result, gr.update(visible=True)

        def handle_validate(file):
            """Handle validate action from modal."""
            if file is None:
                return "⚠️ Please select a file first", gr.update(visible=True)
            result = process_uploaded_report(file, action='validate', schema=DEFAULT_SCHEMA, mode=current_mode)
            return result, gr.update(visible=True)

        def handle_add_to_kb_modal(file):
            """Handle add to KB action from modal."""
            if file is None:
                return "⚠️ Please select a file first", gr.update(visible=True), format_kb_display(), gr.update()
            result = add_pdf_to_kb(file)
            updated_display = format_kb_display()
            updated_sources = get_source_names()
            return result, gr.update(visible=False), updated_display, gr.update(choices=updated_sources)

        def handle_add_to_kb(file):
            """Handle add to KB from right panel."""
            result = add_pdf_to_kb(file)
            print(result)  # Print to console instead of UI
            updated_display = format_kb_display()
            updated_sources = get_source_names()
            return updated_display, gr.update(choices=updated_sources)

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
        msg_input.submit(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(chat_respond, [msg_input, chatbot], [msg_input, chatbot])

        # Modal handlers
        upload_modal_btn.click(show_upload_modal, outputs=[upload_modal])
        close_modal_btn.click(hide_upload_modal, outputs=[upload_modal])

        summarize_btn.click(handle_summarize, inputs=[upload_file], outputs=[upload_status, upload_modal])
        validate_btn.click(handle_validate, inputs=[upload_file], outputs=[upload_status, upload_modal])
        add_to_kb_btn_modal.click(handle_add_to_kb_modal, inputs=[upload_file], outputs=[upload_status, upload_modal, kb_display, source_dropdown])

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
        add_kb_btn.click(handle_add_to_kb, [add_kb_file], [kb_display, source_dropdown])
        refresh_kb_btn.click(handle_refresh_kb, outputs=[kb_display, source_dropdown])
        delete_btn.click(handle_delete_source, [source_dropdown], [kb_display, source_dropdown])

        # Auto-refresh knowledge base on page load
        demo.load(handle_refresh_kb, outputs=[kb_display, source_dropdown])

        # Custom JavaScript for Enter to submit
        demo.load(None, None, None, js="""
        function() {
            setTimeout(function() {
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
