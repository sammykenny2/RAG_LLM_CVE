"""
Gradio Web UI (Phase 2: LangChain)
Claude Projects-style interface using LangChain chains and memory.

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
import re
import numpy as np

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
    EMBEDDING_PRECISION
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

# Track current settings
current_speed = DEFAULT_SPEED
current_mode = DEFAULT_MODE

def initialize_system(speed_level: str = DEFAULT_SPEED, force_reload: bool = False):
    """Initialize LangChain RAG system based on speed level."""
    global rag_system, current_speed

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

    # Initialize LangChain RAG
    rag_system = LangChainRAG()
    rag_system.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)

    # Update current speed
    current_speed = speed_level

    print(f"✅ LangChain RAG system ready (speed={current_speed})")
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

def chat_respond(message: str, history: list):
    """
    Handle chat messages with LangChain's automatic memory management.

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
        # Query LangChain RAG (memory managed automatically)
        response = rag_system.query(question=message)

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
            from core.pdf_processor import PDFProcessor
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
            validation = rag_system.validate_cve_usage(report_text, cve_descriptions, max_tokens=max_tokens)
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
    Add PDF to knowledge base using LangChain.

    Args:
        file: Gradio File object
        source_name: Optional source name

    Returns:
        str: Result message
    """
    if file is None:
        return "⚠️ No file selected"

    try:
        from core.pdf_processor import PDFProcessor

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

        return f"✅ Added {len(chunks)} chunks from '{source_name}' to knowledge base"

    except Exception as e:
        return f"❌ Error adding to KB: {str(e)}"

def format_kb_display() -> str:
    """
    Format knowledge base sources for display.

    Returns:
        str: Formatted HTML string
    """
    try:
        stats = rag_system.get_kb_stats()

        if not stats:
            return "<p>⚠️ No statistics available</p>"

        sources = stats.get('sources', {})

        # Build HTML
        html = f"<div style='padding: 10px;'>"
        html += f"<h4>📊 Statistics</h4>"
        html += f"<p><b>Total documents:</b> {stats.get('total_docs', 0)}</p>"
        html += f"<p><b>By type:</b> {dict(stats.get('by_source_type', {}))}</p>"
        html += f"<br>"
        html += f"<h4>📚 Sources ({len(sources)})</h4>"
        html += f"<ul style='list-style: none; padding-left: 0;'>"

        for source_name, info in sources.items():
            icon = "📄" if info['type'] == 'pdf' else "🔖"
            date = info['added_date'][:10]
            html += f"<li>{icon} <b>{source_name}</b> ({info['count']} chunks, added {date})</li>"

        html += f"</ul></div>"

        return html

    except Exception as e:
        return f"<p>❌ Error loading sources: {str(e)}</p>"

# =============================================================================
# Gradio interface
# =============================================================================

def create_interface():
    """Create and configure Gradio interface."""

    with gr.Blocks(title="RAG CVE Validation System (LangChain)", theme=gr.themes.Soft()) as demo:
        # Title
        gr.Markdown("# 🛡️ RAG CVE Validation System (LangChain)")
        gr.Markdown("Conversational AI with LangChain automatic memory management")

        with gr.Row():
            # Left column: Chat interface (7/12 width)
            with gr.Column(scale=7):
                gr.Markdown("### 💬 Conversation (LangChain)")

                chatbot = gr.Chatbot(
                    label="Chat History (Auto-managed by LangChain Memory)",
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

                    refresh_kb_btn = gr.Button("🔄 Refresh", size="sm")

                    add_kb_file = gr.File(
                        label="Add Files to Knowledge Base",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    add_kb_btn = gr.Button("➕ Add to Knowledge Base", variant="primary")

                    kb_message = gr.Textbox(label="Status", interactive=False)

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
                return "⚠️ Please select a file first", gr.update(visible=True), format_kb_display()
            result = add_pdf_to_kb(file)
            updated_display = format_kb_display()
            return result, gr.update(visible=False), updated_display

        def handle_add_to_kb(file):
            """Handle add to KB from right panel."""
            result = add_pdf_to_kb(file)
            updated_display = format_kb_display()
            return result, updated_display

        def handle_refresh_kb():
            """Refresh knowledge base display."""
            return format_kb_display()

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
        add_to_kb_btn_modal.click(handle_add_to_kb_modal, inputs=[upload_file], outputs=[upload_status, upload_modal, kb_display])

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
        add_kb_btn.click(handle_add_to_kb, [add_kb_file], [kb_message, kb_display])
        refresh_kb_btn.click(handle_refresh_kb, outputs=[kb_display])

        # Auto-refresh knowledge base on page load
        demo.load(handle_refresh_kb, outputs=[kb_display])

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
        print(f"  ⚠️ Error loading stats: {e}")

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT_LANGCHAIN,
        share=GRADIO_SHARE,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
