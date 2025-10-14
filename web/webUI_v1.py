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

def initialize_system(speed_level: str = DEFAULT_SPEED):
    """Initialize RAG system based on speed level."""
    global rag_system, chroma_manager, embedding_model

    if rag_system is not None:
        return  # Already initialized

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

    print("✅ RAG system ready")

# =============================================================================
# Chat interface handlers
# =============================================================================

def chat_respond(message: str, history: list) -> tuple:
    """
    Handle chat messages with RAG context retrieval.

    Args:
        message: User message
        history: Gradio chat history format [[user, bot], ...]

    Returns:
        tuple: (empty string for input clear, updated history)
    """
    if not message.strip():
        return "", history

    try:
        # Query RAG system
        response = rag_system.query(
            question=message,
            include_history=True,
            max_tokens=512
        )

        # Update history
        history.append([message, response])

        return "", history

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append([message, error_msg])
        return "", history

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
            pdf_processor = PDFProcessor()
            text = pdf_processor.extract_text(file_path)
            summary = rag_system.summarize_report(text, max_tokens=700)
            return f"📝 Summary:\n\n{summary}"

        elif action == 'validate':
            # Process report and validate CVE usage
            report_text, cves, cve_descriptions = rag_system.process_report_for_cve_validation(
                str(file_path),
                schema=schema
            )
            validation = rag_system.validate_cve_usage(
                report_text,
                cve_descriptions,
                max_tokens=700
            )
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

def delete_selected_sources(selected_sources: list) -> str:
    """
    Delete selected sources from knowledge base.

    Args:
        selected_sources: List of source names

    Returns:
        str: Result message
    """
    if not selected_sources:
        return "⚠️ No sources selected"

    try:
        deleted_count = 0
        for source_name in selected_sources:
            n_deleted = chroma_manager.delete_by_source(source_name)
            deleted_count += n_deleted

        return f"✅ Deleted {deleted_count} documents from {len(selected_sources)} source(s)"

    except Exception as e:
        return f"❌ Error deleting sources: {str(e)}"

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
                    show_copy_button=True
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
                        info="normal: FP32 baseline | fast: FP16 (recommended) | fastest: FP16+SDPA"
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
                        info="v5: CVE 5.0 only | v4: CVE 4.0 only | all: Fallback"
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
        def handle_send(message, history):
            return chat_respond(message, history)

        def handle_upload(file):
            # Show instructions when file is uploaded
            instruction_msg = upload_for_validation(file)
            return instruction_msg

        def handle_add_to_kb(file):
            result = add_pdf_to_kb(file)
            updated_display = format_kb_display()
            return result, updated_display

        def handle_refresh_kb():
            return format_kb_display()

        # Connect events
        msg_input.submit(handle_send, [msg_input, chatbot], [msg_input, chatbot])
        send_btn.click(handle_send, [msg_input, chatbot], [msg_input, chatbot])

        upload_file.change(handle_upload, [upload_file], [msg_input])

        add_kb_btn.click(handle_add_to_kb, [add_kb_file], [kb_message, kb_display])
        refresh_kb_btn.click(handle_refresh_kb, outputs=[kb_display])

        # Welcome message
        gr.Markdown("""
        ---
        **Quick Start:**
        1. Ask questions about CVEs in the chat
        2. Upload a PDF report for validation (left side)
        3. Add files to knowledge base for future queries (right side)

        *🤖 Powered by Llama 3.2-1B-Instruct + RAG*
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

    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=GRADIO_SHARE,
        inbrowser=True  # Auto-open browser
    )

if __name__ == "__main__":
    main()
