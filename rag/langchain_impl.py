"""
LangChain RAG implementation with conversation history.
Provides conversational AI interface using LangChain chains and memory.
"""

from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path
import re

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import core modules
from core.chroma_manager import ChromaManager
from core.cve_lookup import extract_cves_regex, batch_lookup_cves, format_cve_description
from core.pdf_processor import PDFProcessor

# Import configuration
from config import (
    LLAMA_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CONVERSATION_HISTORY_LENGTH,
    RETRIEVAL_TOP_K,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    VERBOSE_LOGGING,
    SUMMARY_CHUNK_TOKENS,
    SUMMARY_CHUNK_OVERLAP_TOKENS,
    SUMMARY_TOKENS_PER_CHUNK,
    SUMMARY_FINAL_TOKENS,
    SUMMARY_CHUNK_THRESHOLD_CHARS,
    SUMMARY_ENABLE_SECOND_STAGE
)


class LangChainRAG:
    """
    LangChain-based RAG system with automatic conversation management.

    Uses ConversationalRetrievalChain for RAG workflow and
    ConversationBufferWindowMemory for history management.
    """

    def __init__(self):
        """Initialize LangChain RAG system."""
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.qa_chain = None
        self.chroma_manager = None
        self._initialized = False

    def initialize(
        self,
        use_fp16: bool = True,
        use_sdpa: bool = False,
        memory_k: int = CONVERSATION_HISTORY_LENGTH
    ):
        """
        Initialize all LangChain components.

        Args:
            use_fp16: Use FP16 precision for Llama
            use_sdpa: Use SDPA attention (experimental)
            memory_k: Number of conversation rounds to keep
        """
        if self._initialized:
            if VERBOSE_LOGGING:
                print("⚠️ LangChain RAG already initialized")
            return

        print("Initializing LangChain RAG system...")

        # 1. Initialize Llama model as HuggingFacePipeline
        print("Loading Llama model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_NAME,
            trust_remote_code=True
        )

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if use_fp16:
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = "auto"

        if use_sdpa:
            try:
                model_kwargs["attn_implementation"] = "sdpa"
                if VERBOSE_LOGGING:
                    print("  └─ SDPA enabled")
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"  └─ SDPA not available: {e}")

        self.model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            **model_kwargs
        )

        # Create default text generation pipeline (for ConversationalRetrievalChain)
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False  # Only return generated text, not prompt
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        if VERBOSE_LOGGING:
            print("✅ Llama loaded as HuggingFacePipeline (tokenizer and model saved for dynamic pipelines)")

        # 2. Initialize embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        if VERBOSE_LOGGING:
            print("✅ Embeddings loaded")

        # 3. Initialize Chroma vectorstore
        print("Connecting to Chroma database...")
        self.chroma_manager = ChromaManager()
        self.chroma_manager.initialize(create_if_not_exists=True)

        # Convert to LangChain Chroma
        self.vectorstore = Chroma(
            client=self.chroma_manager.client,
            collection_name=self.chroma_manager.collection_name,
            embedding_function=self.embeddings
        )
        if VERBOSE_LOGGING:
            print("✅ Chroma connected")

        # 4. Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_k,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        if VERBOSE_LOGGING:
            print(f"✅ Memory initialized (k={memory_k})")

        # 5. Create ConversationalRetrievalChain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_TOP_K}
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=VERBOSE_LOGGING
        )
        if VERBOSE_LOGGING:
            print("✅ ConversationalRetrievalChain created")

        self._initialized = True
        print("✅ LangChain RAG system ready")

    def _create_pipeline(self, max_new_tokens: int = 512) -> HuggingFacePipeline:
        """
        Create a new HuggingFacePipeline with custom max_new_tokens.

        Args:
            max_new_tokens: Maximum tokens to generate

        Returns:
            HuggingFacePipeline: Pipeline with specified settings
        """
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        return HuggingFacePipeline(pipeline=pipe)

    def _hybrid_search_unified(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[str]:
        """
        Unified hybrid search: exact CVE ID match + semantic search fallback.

        This method mirrors the implementation in pure_python.py for consistency.

        If query contains CVE ID(s), tries metadata filtering for exact match.
        Falls back to semantic search if no exact match found.

        Args:
            query: User query
            top_k: Number of results to return

        Returns:
            list: Retrieved document chunks
        """
        # Check for CVE IDs in query
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cve_ids = re.findall(cve_pattern, query, re.IGNORECASE)

        if cve_ids and self.chroma_manager:
            # Try exact match first
            if VERBOSE_LOGGING:
                print(f"[Hybrid Search] Detected CVE IDs: {cve_ids}")

            for cve_id in cve_ids:
                results = self.chroma_manager.query_by_metadata(
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

        # Use vectorstore's retriever for semantic search
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        docs = retriever.get_relevant_documents(query)

        # Extract page_content from Document objects
        return [doc.page_content for doc in docs]

    def _format_messages_for_llama(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into Llama 3.2 chat template format.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            str: Formatted prompt for Llama
        """
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted

        # Fallback: simple formatting
        formatted_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted_parts.append(f"System: {content}\n")
            elif role == 'user':
                formatted_parts.append(f"User: {content}\n")
            elif role == 'assistant':
                formatted_parts.append(f"Assistant: {content}\n")

        formatted_parts.append("Assistant:")
        return "\n".join(formatted_parts)

    def query(self, question: str, return_sources: bool = False) -> str | Dict:
        """
        Query knowledge base with automatic conversation history.
        Uses unified hybrid search approach (consistent with pure_python.py).

        Args:
            question: User question
            return_sources: Whether to return source documents

        Returns:
            str or dict: Answer (or dict with answer and sources if return_sources=True)
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # 1. Unified hybrid search to get context
        context_items = self._hybrid_search_unified(question, top_k=RETRIEVAL_TOP_K)
        context_str = "\n\n".join(context_items)

        if VERBOSE_LOGGING:
            print(f"[Query] Retrieved {len(context_items)} context items")

        # 2. Build system prompt with context (consistent with pure_python.py)
        system_prompt = (
            "You are a helpful AI assistant with access to a knowledge base about CVEs and security reports. "
            "Answer questions based on the provided context.\n\n"
            f"Context:\n{context_str}"
        )

        # 3. Build messages list with conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history from memory
        if self.memory and self.memory.chat_memory:
            for msg in self.memory.chat_memory.messages:
                if hasattr(msg, 'type'):
                    # LangChain message format
                    role = "user" if msg.type == "human" else "assistant"
                    messages.append({"role": role, "content": msg.content})
                elif isinstance(msg, dict):
                    # Dict format
                    messages.append(msg)

        # Add current question
        messages.append({"role": "user", "content": question})

        if VERBOSE_LOGGING:
            print(f"[Query] Total messages in conversation: {len(messages)}")

        # 4. Format messages for Llama and generate response
        formatted_prompt = self._format_messages_for_llama(messages)

        # Use the default pipeline for generation
        response = self.llm(formatted_prompt)

        if VERBOSE_LOGGING:
            print(f"[Query] Generated response length: {len(response)} chars")

        # 5. Update memory with new interaction
        from langchain.schema import HumanMessage, AIMessage
        self.memory.chat_memory.add_message(HumanMessage(content=question))
        self.memory.chat_memory.add_message(AIMessage(content=response))

        # 6. Return response (with sources if requested)
        if return_sources:
            from langchain.docstore.document import Document
            source_docs = [Document(page_content=item) for item in context_items]
            return {
                "answer": response,
                "sources": source_docs
            }
        else:
            return response

    def clear_history(self):
        """Clear conversation history."""
        if self.memory:
            self.memory.clear()
            if VERBOSE_LOGGING:
                print("✅ Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """
        Get conversation history.

        Returns:
            list: List of message dicts
        """
        if self.memory:
            return self.memory.chat_memory.messages
        return []

    def _chunk_text_by_tokens(
        self,
        text: str,
        chunk_tokens: int = SUMMARY_CHUNK_TOKENS,
        overlap_tokens: int = SUMMARY_CHUNK_OVERLAP_TOKENS
    ) -> List[str]:
        """
        Split text into chunks based on token count (more accurate than character-based).

        Args:
            text: Text to chunk
            chunk_tokens: Target tokens per chunk
            overlap_tokens: Overlap between chunks

        Returns:
            list: List of text chunks
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")

        # Encode text to tokens
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) == 0:
            return []

        if len(token_ids) <= chunk_tokens:
            return [text]

        # Ensure overlap doesn't exceed half of chunk size
        overlap_tokens = min(overlap_tokens, chunk_tokens // 2)
        stride = max(chunk_tokens - overlap_tokens, 1)

        # Create chunks with overlap
        chunks = []
        for start in range(0, len(token_ids), stride):
            end = start + chunk_tokens
            chunk_token_ids = token_ids[start:end]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

            if end >= len(token_ids):
                break

        return chunks

    def summarize_report(self, report_text: str, max_tokens: int = None) -> str:
        """
        Generate executive summary (non-conversational, single-shot).

        Uses intelligent token-based chunking for long texts to avoid hallucination.
        - Short texts: Direct summarization
        - Long texts: Two-stage summarization (chunk-by-chunk + final condensing)

        Args:
            report_text: Full report text
            max_tokens: Maximum tokens for single-pass summary (default from config)

        Returns:
            str: Summary (concise if second-stage enabled, detailed if disabled)
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Use config defaults if not specified
        if max_tokens is None:
            max_tokens = SUMMARY_FINAL_TOKENS

        # Check if text is short enough for single-pass summarization
        if len(report_text) < SUMMARY_CHUNK_THRESHOLD_CHARS:
            # Short text: direct summarization
            return self._generate_single_summary(report_text, max_tokens)
        else:
            # Long text: two-stage summarization
            return self._generate_two_stage_summary(report_text)

    def _generate_single_summary(self, text: str, max_tokens: int = SUMMARY_FINAL_TOKENS) -> str:
        """
        Generate summary for a single text chunk (used for short documents).

        Args:
            text: Text to summarize
            max_tokens: Maximum tokens to generate

        Returns:
            str: Summary
        """
        llm = self._create_pipeline(max_new_tokens=max_tokens)
        prompt = "Summarize the following threat intelligence report. Provide a clear and concise executive summary.\n\n" + text
        response = llm(prompt)
        return response

    def _generate_two_stage_summary(self, text: str) -> str:
        """
        Generate summary using two-stage approach for long documents.

        Stage 1: Split into token-based chunks and summarize each
        Stage 2 (optional): Condense all chunk summaries into final executive summary

        Args:
            text: Long text to summarize

        Returns:
            str: Final summary (concise if second-stage enabled, detailed if disabled)
        """
        # Stage 1: Chunk and summarize
        chunks = self._chunk_text_by_tokens(
            text,
            chunk_tokens=SUMMARY_CHUNK_TOKENS,
            overlap_tokens=SUMMARY_CHUNK_OVERLAP_TOKENS
        )

        if not chunks:
            return "Error: Could not chunk text for summarization."

        if VERBOSE_LOGGING:
            print(f"📝 Summarizing {len(chunks)} chunks (Stage 1)...")

        # Create pipeline for first stage
        llm_stage1 = self._create_pipeline(max_new_tokens=SUMMARY_TOKENS_PER_CHUNK)

        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            if VERBOSE_LOGGING:
                print(f"   Processing chunk {i}/{len(chunks)}...")

            prompt = "Summarize this section of a threat intelligence report. Provide a concise summary of the key points.\n\n" + chunk
            summary = llm_stage1(prompt)
            chunk_summaries.append(summary)

        # Stage 2: Optional final condensing
        if SUMMARY_ENABLE_SECOND_STAGE:
            if VERBOSE_LOGGING:
                print(f"📝 Condensing summaries into executive summary (Stage 2)...")

            # Combine all chunk summaries
            combined_summaries = "\n\n".join([
                f"Part {i+1}: {summary}"
                for i, summary in enumerate(chunk_summaries)
            ])

            # Create pipeline for second stage
            llm_stage2 = self._create_pipeline(max_new_tokens=SUMMARY_FINAL_TOKENS)

            prompt = (
                "You are creating an executive summary. "
                "Condense the following section summaries into a single, coherent executive summary. "
                "Focus on the most important findings and eliminate redundancy.\n\n"
                + combined_summaries
            )

            final_summary = llm_stage2(prompt)
            return final_summary
        else:
            # Return all chunk summaries without condensing
            return "\n\n".join([
                f"Section {i+1}:\n{summary}"
                for i, summary in enumerate(chunk_summaries)
            ])

    def validate_cve_usage(
        self,
        report_text: str,
        cve_descriptions: str,
        max_tokens: int = 512
    ) -> str:
        """
        Validate CVE usage (non-conversational, single-shot).

        Args:
            report_text: Security report text
            cve_descriptions: Official CVE descriptions
            max_tokens: Maximum tokens to generate (dynamically applied)

        Returns:
            str: Validation result
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Create pipeline with custom max_tokens
        llm = self._create_pipeline(max_new_tokens=max_tokens)

        prompt = (
            f"Verify if the CVEs in the following report are used correctly.\n\n"
            f"Correct CVE Descriptions:\n{cve_descriptions}\n\n"
            f"Report:\n{report_text}\n\n"
            f"Provide detailed verification with quotes."
        )

        response = llm(prompt)
        return response

    def answer_question_about_report(
        self,
        report_text: str,
        question: str
    ) -> str:
        """
        Answer question about report (non-conversational, single-shot).

        Args:
            report_text: Security report text
            question: User question

        Returns:
            str: Answer
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        prompt = f"{question}\n\nReport:\n{report_text}"
        response = self.llm(prompt)

        return response

    def process_report_for_cve_validation(
        self,
        pdf_path: str,
        schema: str = 'all',
        max_pages: Optional[int] = None
    ) -> Tuple[str, List[str], str]:
        """
        Process PDF report and lookup CVE information.

        Args:
            pdf_path: Path to PDF file
            schema: CVE schema ('v5', 'v4', or 'all')
            max_pages: Maximum pages to process (None for all)

        Returns:
            tuple: (report_text, cve_list, cve_descriptions_text)
        """
        # Extract text
        pdf_processor = PDFProcessor()
        report_text = pdf_processor.extract_text(pdf_path, max_pages=max_pages)

        # Extract CVEs
        cves = extract_cves_regex(report_text)

        # Lookup CVE information
        cve_results = batch_lookup_cves(cves, schema=schema)

        # Format descriptions
        cve_descriptions = []
        missing_cves = []

        for cve, info in cve_results.items():
            if info:
                cve_descriptions.append(format_cve_description(info))
            else:
                missing_cves.append(cve)

        cve_descriptions_text = "\n\n\n".join(cve_descriptions)

        if missing_cves and VERBOSE_LOGGING:
            print(f"⚠️ Could not find {len(missing_cves)} CVEs: {', '.join(missing_cves)}")

        return report_text, cves, cve_descriptions_text

    def add_document_to_kb(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """
        Add documents to knowledge base.

        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each chunk
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Use vectorstore to add texts (embeddings are generated automatically)
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

        if VERBOSE_LOGGING:
            print(f"✅ Added {len(texts)} documents to knowledge base")

    def get_kb_stats(self) -> Dict:
        """
        Get knowledge base statistics.

        Returns:
            dict: Statistics
        """
        if self.chroma_manager:
            return self.chroma_manager.get_stats()
        return {}

    def delete_source(self, source_name: str) -> int:
        """
        Delete source from knowledge base.

        Args:
            source_name: Name of source to delete

        Returns:
            int: Number of documents deleted
        """
        if self.chroma_manager:
            return self.chroma_manager.delete_by_source(source_name)
        return 0

    def cleanup(self):
        """Clean up resources."""
        if self.llm:
            # LangChain handles cleanup internally
            self.llm = None

        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if self.memory:
            self.memory.clear()
            self.memory = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._initialized = False

        if VERBOSE_LOGGING:
            print("✅ LangChain RAG cleaned up")


# =============================================================================
# Utility functions (simplified API)
# =============================================================================

def create_langchain_rag(use_fp16: bool = True, use_sdpa: bool = False) -> LangChainRAG:
    """
    Create and initialize a LangChain RAG system.

    Args:
        use_fp16: Use FP16 precision
        use_sdpa: Use SDPA attention

    Returns:
        LangChainRAG: Initialized RAG system
    """
    rag = LangChainRAG()
    rag.initialize(use_fp16=use_fp16, use_sdpa=use_sdpa)
    return rag
