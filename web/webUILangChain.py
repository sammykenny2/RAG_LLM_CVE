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

def initialize_system(speed_level: str = DEFAULT_SPEED):
    """Initialize LangChain RAG system based on speed level."""
    global rag_system

    if rag_system is not None:
        return  # Already initialized

    print("Initializing LangChain RAG system for web UI...")

    # Speed level configuration
    use_fp16 = speed_level in ['fast', 'fastest']
    use_sdpa = speed_level == 'fastest'

    # Initialize LangChain RAG
    rag_system = LangChainRAG()
    rag_system.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)

    print("✅ LangChain RAG system ready")

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
    schema: str = DEFAULT_SCHEMA
) -> str:
    """
    Process uploaded report based on user action.

    Args:
        file: Gradio File object
        action: 'summarize', 'validate', or 'add'
        schema: CVE schema to use

    Returns:
        str: Processing result
    """
    if file is None:
        return "⚠️ No file uploaded"

    try:
        file_path = Path(file.name)

        if action == 'summarize':
            # Extract text and summarize
            from core.pdf_processor import PDFProcessor
            pdf_processor = PDFProcessor()
            text = pdf_processor.extract_text(file_path)
            summary = rag_system.summarize_report(text)
            return f"📝 Summary:\n\n{summary}"

        elif action == 'validate':
            # Process report and validate CVE usage
            report_text, cves, cve_descriptions = rag_system.process_report_for_cve_validation(
                str(file_path),
                schema=schema
            )
            validation = rag_system.validate_cve_usage(report_text, cve_descriptions)
            return f"✅ Validation Result:\n\n{validation}\n\n📋 Found CVEs: {', '.join(cves)}"

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

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Ask about CVEs, security reports, or upload a file for validation...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                with gr.Row():
                    upload_file = gr.File(
                        label="Upload Report for Validation",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                gr.Markdown("**After uploading:** Type 'summarize', 'validate', or 'add to kb'")

            # Right column: Settings and Knowledge Base (5/12 width)
            with gr.Column(scale=5):
                # Instructions panel
                gr.Markdown("### ⚙️ Analysis Settings")
                with gr.Group():
                    speed_dropdown = gr.Dropdown(
                        choices=['normal', 'fast', 'fastest'],
                        value=DEFAULT_SPEED,
                        label="Speed Level",
                        info="normal: FP32 | fast: FP16 (recommended) | fastest: FP16+SDPA"
                    )
                    mode_dropdown = gr.Dropdown(
                        choices=['demo', 'full'],
                        value=DEFAULT_MODE,
                        label="Mode",
                        info="demo: Memory-optimized | full: Complete features"
                    )
                    schema_dropdown = gr.Dropdown(
                        choices=['v5', 'v4', 'all'],
                        value=DEFAULT_SCHEMA,
                        label="CVE Schema",
                        info="v5: CVE 5.0 | v4: CVE 4.0 | all: Fallback"
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

        # Event handlers
        def handle_upload(file):
            instruction_msg = upload_for_validation(file)
            return instruction_msg

        def handle_add_to_kb(file):
            result = add_pdf_to_kb(file)
            updated_display = format_kb_display()
            return result, updated_display

        def handle_refresh_kb():
            return format_kb_display()

        # Connect events - directly bind chat_respond (it's a generator)
        msg_input.submit(chat_respond, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(chat_respond, [msg_input, chatbot], [msg_input, chatbot])

        upload_file.change(handle_upload, [upload_file], [msg_input])

        add_kb_btn.click(handle_add_to_kb, [add_kb_file], [kb_message, kb_display])
        refresh_kb_btn.click(handle_refresh_kb, outputs=[kb_display])

        # Welcome message
        gr.Markdown("""
        ---
        **Phase 2 Features:**
        - ✅ LangChain ConversationalRetrievalChain
        - ✅ Automatic memory management (last 10 rounds)
        - ✅ Standardized RAG workflow
        - ✅ Same interface as Phase 1

        **Quick Start:**
        1. Ask questions about CVEs in the chat
        2. Upload a PDF report for validation (left side)
        3. Add files to knowledge base (right side)

        *🤖 Powered by Llama 3.2-1B-Instruct + LangChain + RAG*
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

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT_LANGCHAIN,
        share=GRADIO_SHARE,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
