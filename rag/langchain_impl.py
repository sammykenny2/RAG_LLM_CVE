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
    VERBOSE_LOGGING
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

    def query(self, question: str, return_sources: bool = False) -> str | Dict:
        """
        Query knowledge base with automatic conversation history.
        Uses hybrid search: exact CVE ID match + semantic search.

        Args:
            question: User question
            return_sources: Whether to return source documents

        Returns:
            str or dict: Answer (or dict with answer and sources if return_sources=True)
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Check for CVE IDs and try exact match first
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        cve_ids = re.findall(cve_pattern, question, re.IGNORECASE)

        if cve_ids and self.chroma_manager:
            # Try exact CVE ID match
            if VERBOSE_LOGGING:
                print(f"[Hybrid Search] Detected CVE IDs: {cve_ids}")

            for cve_id in cve_ids:
                results = self.chroma_manager.query_by_metadata(
                    where={"cve_id": cve_id.upper()},
                    limit=RETRIEVAL_TOP_K
                )

                if results and results.get('documents') and len(results['documents']) > 0:
                    if VERBOSE_LOGGING:
                        print(f"[Hybrid Search] Exact match found for {cve_id}")

                    # Format direct answer from exact match
                    context = "\n\n".join(results['documents'][:3])  # Use top 3
                    prompt = f"Based on the following information, answer this question: {question}\n\nInformation:\n{context}"

                    response = self.llm(prompt)

                    if return_sources:
                        # Convert to source document format
                        from langchain.docstore.document import Document
                        source_docs = [
                            Document(page_content=doc, metadata=meta)
                            for doc, meta in zip(results['documents'], results.get('metadatas', []))
                        ]
                        return {
                            "answer": response,
                            "sources": source_docs
                        }
                    else:
                        return response

            if VERBOSE_LOGGING:
                print(f"[Hybrid Search] No exact match, using semantic search")

        # Fallback to normal chain (semantic search)
        result = self.qa_chain({"question": question})

        if return_sources:
            return {
                "answer": result["answer"],
                "sources": result.get("source_documents", [])
            }
        else:
            return result["answer"]

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

    def summarize_report(self, report_text: str, max_tokens: int = 512) -> str:
        """
        Generate executive summary (non-conversational, single-shot).

        Args:
            report_text: Full report text
            max_tokens: Maximum tokens to generate (dynamically applied)

        Returns:
            str: Summary
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Create pipeline with custom max_tokens
        llm = self._create_pipeline(max_new_tokens=max_tokens)

        prompt = f"Summarize the following threat intelligence report:\n\n{report_text}"
        response = llm(prompt)

        return response

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
