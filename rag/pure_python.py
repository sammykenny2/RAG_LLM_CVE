"""
Pure Python RAG implementation with conversation history.
Provides conversational AI interface without LangChain dependencies.
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
import re

# Import core modules
from core.models import LlamaModel
from core.embeddings import EmbeddingModel
from core.chroma_manager import ChromaManager
from core.cve_lookup import lookup_cve, extract_cves_regex, format_cve_description, batch_lookup_cves
from core.pdf_processor import PDFProcessor, extract_cve_context

# Import configuration
from config import (
    CONVERSATION_HISTORY_LENGTH,
    RETRIEVAL_TOP_K,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    VERBOSE_LOGGING
)


class ConversationHistory:
    """
    Manage conversation history with a fixed-size sliding window.

    Stores last N rounds of conversation (user + assistant messages).
    """

    def __init__(self, max_rounds: int = CONVERSATION_HISTORY_LENGTH):
        """
        Initialize conversation history.

        Args:
            max_rounds: Maximum number of conversation rounds to keep
        """
        self.max_rounds = max_rounds
        self.history = deque(maxlen=max_rounds * 2)  # *2 for user + assistant pairs

    def add_user_message(self, message: str):
        """Add user message to history."""
        self.history.append({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        """Add assistant message to history."""
        self.history.append({"role": "assistant", "content": message})

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in history.

        Returns:
            list: List of message dicts with 'role' and 'content'
        """
        return list(self.history)

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    def __len__(self):
        """Return number of messages in history."""
        return len(self.history)


class PureRAG:
    """
    Pure Python RAG system with conversation history.

    Provides query, summarization, validation, and Q&A capabilities
    without LangChain dependencies.
    """

    def __init__(
        self,
        llama_model: LlamaModel = None,
        embedding_model: EmbeddingModel = None,
        chroma_manager: ChromaManager = None,
        conversation_history: ConversationHistory = None
    ):
        """
        Initialize RAG system.

        Args:
            llama_model: LlamaModel instance (creates new if None)
            embedding_model: EmbeddingModel instance (creates new if None)
            chroma_manager: ChromaManager instance (creates new if None)
            conversation_history: ConversationHistory instance (creates new if None)
        """
        self.llama = llama_model or LlamaModel()
        self.embedder = embedding_model or EmbeddingModel()
        self.chroma = chroma_manager or ChromaManager()
        self.history = conversation_history or ConversationHistory()

        self._initialized = False

    def initialize(self, use_fp16: bool = True, use_sdpa: bool = False):
        """
        Initialize all components.

        Args:
            use_fp16: Use FP16 precision for Llama
            use_sdpa: Use SDPA attention for Llama
        """
        if self._initialized:
            if VERBOSE_LOGGING:
                print("⚠️ RAG system already initialized")
            return

        print("Initializing RAG system...")

        # Initialize Llama
        self.llama.use_fp16 = use_fp16
        self.llama.use_sdpa = use_sdpa
        self.llama.initialize()

        # Initialize embedder
        self.embedder.initialize()

        # Initialize Chroma
        self.chroma.initialize(create_if_not_exists=True)

        self._initialized = True

        if VERBOSE_LOGGING:
            print("✅ RAG system initialized")

    def _hybrid_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[str]:
        """
        Hybrid search: exact CVE ID match + semantic search.

        If query contains CVE ID(s), use metadata filtering for exact match.
        Otherwise, use semantic search.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            list: Retrieved document chunks
        """
        # Check for CVE IDs in query
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cve_ids = re.findall(cve_pattern, query, re.IGNORECASE)

        if cve_ids:
            # Try exact match first
            if VERBOSE_LOGGING:
                print(f"[Hybrid Search] Detected CVE IDs: {cve_ids}")

            for cve_id in cve_ids:
                results = self.chroma.query_by_metadata(
                    where={"cve_id": cve_id.upper()},
                    limit=top_k
                )

                if results and results.get('documents') and len(results['documents']) > 0:
                    if VERBOSE_LOGGING:
                        print(f"[Hybrid Search] Exact match found for {cve_id}: {len(results['documents'])} result(s)")
                    # Return exact matches (query_by_metadata returns flat list)
                    return results['documents']

            if VERBOSE_LOGGING:
                print(f"[Hybrid Search] No exact match, falling back to semantic search")

        # Fallback to semantic search
        if VERBOSE_LOGGING:
            print(f"[Hybrid Search] Using semantic search")

        query_embedding = self.embedder.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        results = self.chroma.query(
            query_embedding=query_embedding.tolist(),
            top_k=top_k
        )

        return results['documents'][0]

    def query(
        self,
        question: str,
        include_history: bool = True,
        top_k: int = RETRIEVAL_TOP_K,
        temperature: float = None,
        max_tokens: int = 256
    ) -> str:
        """
        Query knowledge base with conversation history.
        Uses hybrid search: exact match (if CVE ID detected) + semantic search.

        Args:
            question: User question
            include_history: Whether to include conversation history in prompt
            top_k: Number of similar chunks to retrieve
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens to generate

        Returns:
            str: Generated answer
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Extract context using hybrid search
        context_items = self._hybrid_search(question, top_k)
        context_str = "\n- ".join(context_items)

        # Build messages
        system_prompt = (
            "You are a helpful AI assistant with access to a knowledge base about CVEs and security reports. "
            "Answer questions based on the provided context.\n\n"
            f"Context:\n- {context_str}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if requested
        if include_history:
            messages.extend(self.history.get_messages())

        # Add current question
        messages.append({"role": "user", "content": question})

        # Generate response
        response = self.llama.generate(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        # Update history
        self.history.add_user_message(question)
        self.history.add_assistant_message(response)

        return response

    def summarize_report(
        self,
        report_text: str,
        max_tokens: int = 700
    ) -> str:
        """
        Generate executive summary of a security report.

        Args:
            report_text: Full report text
            max_tokens: Maximum tokens to generate

        Returns:
            str: Summary
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        system_prompt = (
            "You are a ChatBot that summarizes threat intelligence reports. "
            "Your task is to summarize the report given to you by the user."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize the following Threat Intelligence Report: {report_text}"}
        ]

        response = self.llama.generate(
            messages=messages,
            max_new_tokens=max_tokens
        )

        return response

    def validate_cve_usage(
        self,
        report_text: str,
        cve_descriptions: str,
        max_tokens: int = 700
    ) -> str:
        """
        Validate CVE usage in a report against official descriptions.

        Args:
            report_text: Security report text
            cve_descriptions: Official CVE descriptions
            max_tokens: Maximum tokens to generate

        Returns:
            str: Validation result
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        system_prompt = (
            "You are a chatbot that verifies the correct use of CVEs (Common Vulnerabilities and Exposures) mentioned in a "
            "Threat Intelligence Report. A CVE is used correctly when it closely matches the provided Correct CVE Description. "
            "Incorrect usage includes citing non-existent CVEs, misrepresenting the Correct CVE Description, or inaccurately applying the CVE.\n"
            f"Correct CVE Descriptions:\n{cve_descriptions}\n"
            "Instructions:\n"
            "1. Verify each CVE mentioned in the user-provided report.\n"
            "2. Indicate whether each CVE is used correctly or not.\n"
            "3. Provide a detailed explanation with direct quotes from both the report and the Correct CVE Description.\n"
            "A CVE in the report is incorrect if it describes a different vulnerability."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please verify if the following CVEs have been used correctly in this Threat Intelligence Report:\n{report_text}"}
        ]

        response = self.llama.generate(
            messages=messages,
            max_new_tokens=max_tokens
        )

        return response

    def answer_question_about_report(
        self,
        report_text: str,
        question: str,
        max_tokens: int = 700
    ) -> str:
        """
        Answer a specific question about a report.

        Args:
            report_text: Security report text
            question: User question
            max_tokens: Maximum tokens to generate

        Returns:
            str: Answer
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        system_prompt = "You are a chatbot that answers questions based on the text provided to you."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\nReport: {report_text}"}
        ]

        response = self.llama.generate(
            messages=messages,
            max_new_tokens=max_tokens
        )

        return response

    def process_report_for_cve_validation(
        self,
        pdf_path: str,
        schema: str = 'all',
        max_pages: Optional[int] = None
    ) -> Tuple[str, List[str], str]:
        """
        Process a PDF report and lookup CVE information for validation.

        Args:
            pdf_path: Path to PDF file
            schema: CVE schema to use ('v5', 'v4', or 'all')
            max_pages: Maximum pages to process (None for all)

        Returns:
            tuple: (report_text, cve_list, cve_descriptions_text)
        """
        # Extract text from PDF
        pdf_processor = PDFProcessor()
        report_text = pdf_processor.extract_text(pdf_path, max_pages=max_pages)

        # Extract CVEs using regex
        cves = extract_cves_regex(report_text)

        # Lookup CVE information
        cve_results = batch_lookup_cves(cves, schema=schema)

        # Format CVE descriptions
        cve_descriptions = []
        missing_cves = []

        for cve, info in cve_results.items():
            if info:
                cve_descriptions.append(format_cve_description(info))
            else:
                missing_cves.append(cve)

        cve_descriptions_text = "\n\n\n".join(cve_descriptions)

        # Handle missing CVEs (optional: could query LLM for suggestions)
        if missing_cves and VERBOSE_LOGGING:
            print(f"⚠️ Could not find {len(missing_cves)} CVEs: {', '.join(missing_cves)}")

        return report_text, cves, cve_descriptions_text

    def cleanup(self):
        """Clean up resources."""
        if self.llama:
            self.llama.cleanup()
        if self.embedder:
            self.embedder.cleanup()

        self._initialized = False

        if VERBOSE_LOGGING:
            print("✅ RAG system cleaned up")


# =============================================================================
# Utility functions (simplified API)
# =============================================================================

def create_rag_system(use_fp16: bool = True, use_sdpa: bool = False) -> PureRAG:
    """
    Create and initialize a RAG system.

    Args:
        use_fp16: Use FP16 precision
        use_sdpa: Use SDPA attention

    Returns:
        PureRAG: Initialized RAG system
    """
    rag = PureRAG()
    rag.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)
    return rag
