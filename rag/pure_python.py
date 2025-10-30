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
    VERBOSE_LOGGING,
    SUMMARY_CHUNK_TOKENS,
    SUMMARY_CHUNK_OVERLAP_TOKENS,
    SUMMARY_TOKENS_PER_CHUNK,
    SUMMARY_FINAL_TOKENS,
    SUMMARY_CHUNK_THRESHOLD_CHARS,
    SUMMARY_ENABLE_SECOND_STAGE,
    VALIDATION_CHUNK_TOKENS,
    VALIDATION_CHUNK_OVERLAP_TOKENS,
    VALIDATION_TOKENS_PER_CHUNK,
    VALIDATION_CHUNK_THRESHOLD_CHARS,
    VALIDATION_ENABLE_CVE_FILTERING,
    VALIDATION_FINAL_TOKENS,
    VALIDATION_ENABLE_SECOND_STAGE,
    QA_CHUNK_TOKENS,
    QA_CHUNK_OVERLAP_TOKENS,
    QA_TOKENS_PER_CHUNK,
    QA_CHUNK_THRESHOLD_CHARS,
    QA_FINAL_TOKENS,
    QA_ENABLE_SECOND_STAGE
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
        conversation_history: ConversationHistory = None,
        session_manager = None
    ):
        """
        Initialize RAG system.

        Args:
            llama_model: LlamaModel instance (creates new if None)
            embedding_model: EmbeddingModel instance (creates new if None)
            chroma_manager: ChromaManager instance (creates new if None)
            conversation_history: ConversationHistory instance (creates new if None)
            session_manager: SessionManager instance for multi-file context (optional)
        """
        self.llama = llama_model or LlamaModel()
        self.embedder = embedding_model or EmbeddingModel()
        self.chroma = chroma_manager or ChromaManager()
        self.history = conversation_history or ConversationHistory()
        self.session_manager = session_manager  # Optional: for multi-file conversations

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
                print("[WARNING] RAG system already initialized")
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
            print("[OK] RAG system initialized")

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

    def _merge_results(
        self,
        kb_results: List[str],
        session_results: List[Dict] = None
    ) -> List[Dict]:
        """
        Merge results from permanent KB and session files.

        Args:
            kb_results: Results from permanent knowledge base (list of strings)
            session_results: Results from session manager (list of dicts with 'text', 'source', 'score')

        Returns:
            list: Merged results sorted by score, with dict format:
                {'text': str, 'source': str, 'source_type': str, 'score': float}
        """
        merged = []

        # Add KB results (convert to dict format)
        for text in kb_results:
            merged.append({
                'text': text,
                'source': 'Knowledge Base',
                'source_type': 'permanent',
                'score': 1.0  # KB results from hybrid search don't have scores
            })

        # Add session results (already in dict format)
        if session_results:
            for result in session_results:
                merged.append({
                    'text': result.get('text', ''),
                    'source': result.get('source', 'Unknown'),
                    'source_type': result.get('source_type', 'session'),
                    'score': result.get('score', 0.0)
                })

        # Sort by score (descending)
        merged.sort(key=lambda x: x['score'], reverse=True)

        return merged

    def _build_prompt_with_sources(self, context_items: List[Dict]) -> str:
        """
        Build context string with source attribution.

        Args:
            context_items: List of dicts with 'text' and 'source' keys

        Returns:
            str: Formatted context string with source attribution
        """
        formatted_items = []

        for item in context_items:
            source = item.get('source', 'Unknown')
            text = item.get('text', '')
            formatted_items.append(f"From {source}: {text}")

        return "\n\n".join(formatted_items)

    def query(
        self,
        question: str,
        include_history: bool = True,
        top_k: int = RETRIEVAL_TOP_K,
        temperature: float = None,
        max_tokens: int = 1000
    ) -> str:
        """
        Query knowledge base with conversation history.
        Uses dual-source retrieval: session files (priority) + KB (supplement).

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

        # 1. Query session files first if session_manager exists (prioritize uploaded files)
        session_results = None
        if self.session_manager:
            try:
                session_results = self.session_manager.query(question, top_k=top_k)
                if VERBOSE_LOGGING and session_results:
                    print(f"[Dual-Source] Retrieved {len(session_results)} results from session files")
            except Exception as e:
                if VERBOSE_LOGGING:
                    print(f"[Dual-Source] Session query failed: {e}")

        # 2. If session has enough results, use them; otherwise supplement with KB
        if session_results and len(session_results) >= top_k:
            # Use only session results (user uploaded files are most relevant)
            top_results = self._merge_results([], session_results)[:top_k]
            if VERBOSE_LOGGING:
                print(f"[Dual-Source] Using {len(top_results)} results from session files only")
        else:
            # Supplement with KB results
            kb_count = top_k - (len(session_results) if session_results else 0)
            kb_results = self._hybrid_search(question, top_k=kb_count) if kb_count > 0 else []

            # 3. Merge and rank results
            merged_results = self._merge_results(kb_results, session_results)
            top_results = merged_results[:top_k]

            if VERBOSE_LOGGING:
                session_count = len(session_results) if session_results else 0
                kb_count_actual = len(kb_results) if kb_results else 0
                print(f"[Dual-Source] Using {session_count} session + {kb_count_actual} KB results")

        # 4. Build prompt with source attribution
        context_str = self._build_prompt_with_sources(top_results)

        # Build messages
        system_prompt = (
            "You are a professional cybersecurity analyst specializing in CVE (Common Vulnerabilities and Exposures) research and threat intelligence analysis. "
            "Your expertise includes vulnerability assessment, security advisories, attack patterns, and exploit analysis.\n\n"
            "Core Responsibilities:\n"
            "- Provide accurate, evidence-based answers about CVEs, vulnerabilities, and security threats\n"
            "- Cite specific information from the knowledge base when available\n"
            "- Clearly distinguish between confirmed facts and inferences\n"
            "- Use precise technical terminology (e.g., RCE, privilege escalation, information disclosure)\n"
            "- Respond in the same language as the user's question\n\n"
            "Answer Guidelines:\n"
            "1. For CVE-specific queries: Provide CVE ID, affected products, vulnerability type, severity, and remediation\n"
            "2. For technical questions: Include technical details, attack vectors, and mitigation strategies\n"
            "3. For contextual questions: Reference related CVEs or similar vulnerabilities when relevant\n"
            "4. If information is insufficient: State limitations clearly and avoid speculation\n"
            "5. For ambiguous queries: Ask for clarification or provide multiple interpretations\n"
            "6. For protocol vulnerability analysis: Address attack conditions, impact scope, exploitability, and protocol-specific mechanisms\n"
            "7. For remediation recommendations: Provide prioritized measures with expected outcomes and verification methods\n\n"
            f"Knowledge Base Context:\n{context_str}\n\n"
            "Provide professional, precise, and actionable responses based on the context above."
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
        if not self.llama.tokenizer:
            raise RuntimeError("Tokenizer not initialized")

        # Encode text to tokens
        token_ids = self.llama.tokenizer.encode(text, add_special_tokens=False)

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
            chunk_text = self.llama.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

            if end >= len(token_ids):
                break

        return chunks

    def summarize_report(
        self,
        report_text: str,
        max_tokens: int = None
    ) -> str:
        """
        Generate executive summary of a security report.

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
        system_prompt = (
            "You are a senior threat intelligence analyst specializing in cybersecurity incident analysis and vulnerability assessment. "
            "Your role is to synthesize complex security reports into clear, actionable executive summaries for SOC teams and security leadership.\n\n"
            "Summary Requirements:\n"
            "- Focus on critical threats, affected systems, and recommended actions\n"
            "- Highlight CVEs with severity levels and exploitation status\n"
            "- Identify threat actors, TTPs (Tactics, Techniques, Procedures), and attack vectors\n"
            "- Use clear structure: Overview → Key Findings → Impact Assessment → Recommendations\n"
            "- Maintain technical accuracy while ensuring accessibility for non-technical stakeholders\n"
            "- Preserve critical details (CVE IDs, affected versions, patch information)\n\n"
            "Provide a professional, concise, and actionable executive summary."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize the following Threat Intelligence Report:\n\n{text}"}
        ]

        response = self.llama.generate(
            messages=messages,
            max_new_tokens=max_tokens
        )

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
            print(f"[INFO] Summarizing {len(chunks)} chunks (Stage 1)...")

        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            if VERBOSE_LOGGING:
                print(f"[INFO] Processing chunk {i}/{len(chunks)}...")

            system_prompt = (
                "You are a senior threat intelligence analyst specializing in cybersecurity incident analysis. "
                "Your task is to extract and summarize the key security findings from this section of a threat intelligence report.\n\n"
                "Focus on:\n"
                "- CVEs mentioned (ID, severity, affected products)\n"
                "- Threat actors and attack campaigns\n"
                "- Vulnerabilities and exploitation techniques\n"
                "- Indicators of Compromise (IoCs)\n"
                "- Mitigation or remediation guidance\n\n"
                "Provide a concise, technically accurate summary of the key security points in this section."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this section:\n\n{chunk}"}
            ]

            summary = self.llama.generate(
                messages=messages,
                max_new_tokens=SUMMARY_TOKENS_PER_CHUNK
            )

            chunk_summaries.append(summary)

        # Stage 2: Optional final condensing
        if SUMMARY_ENABLE_SECOND_STAGE:
            if VERBOSE_LOGGING:
                print(f"[INFO] Condensing summaries into executive summary (Stage 2)...")

            # Combine all chunk summaries
            combined_summaries = "\n\n".join([
                f"Part {i+1}: {summary}"
                for i, summary in enumerate(chunk_summaries)
            ])

            system_prompt = (
                "You are a senior threat intelligence analyst creating an executive summary for security leadership. "
                "Your task is to synthesize multiple section summaries into a unified, coherent executive summary.\n\n"
                "Executive Summary Structure:\n"
                "1. Threat Overview: High-level summary of primary threats and incidents\n"
                "2. Key Vulnerabilities: Critical CVEs with severity, affected systems, and exploitation status\n"
                "3. Threat Actor Activity: Notable campaigns, TTPs, and attribution\n"
                "4. Impact Assessment: Potential or confirmed impact to systems and operations\n"
                "5. Recommendations: Prioritized actions for mitigation and remediation\n\n"
                "Guidelines:\n"
                "- Eliminate redundancy while preserving all critical CVE IDs and technical details\n"
                "- Prioritize high-severity findings and active threats\n"
                "- Use clear, professional language suitable for executive audiences\n"
                "- Maintain technical accuracy and actionable insights\n\n"
                "Create a comprehensive, well-structured executive summary."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create an executive summary from these sections:\n\n{combined_summaries}"}
            ]

            final_summary = self.llama.generate(
                messages=messages,
                max_new_tokens=SUMMARY_FINAL_TOKENS
            )

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
        max_tokens: int = None
    ) -> str:
        """
        Validate CVE usage in a report against official descriptions.

        Uses intelligent chunking and chunk-aware CVE filtering for 7-10x speedup.
        - Short texts: Direct validation with all CVE descriptions
        - Long texts: Chunked validation with filtered CVE descriptions (only relevant CVEs per chunk)

        Args:
            report_text: Security report text
            cve_descriptions: Official CVE descriptions (triple-newline separated blocks)
            max_tokens: Maximum tokens per chunk (default from config)

        Returns:
            str: Validation result
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        # Use config default if not specified
        if max_tokens is None:
            max_tokens = VALIDATION_TOKENS_PER_CHUNK

        # Check if text is short enough for single-pass validation
        if len(report_text) < VALIDATION_CHUNK_THRESHOLD_CHARS:
            # Short text: direct validation with all CVE descriptions
            return self._validate_single_pass(report_text, cve_descriptions, max_tokens)
        else:
            # Long text: chunked validation with optional CVE filtering
            return self._validate_chunked(report_text, cve_descriptions, max_tokens)

    def _validate_single_pass(
        self,
        report_text: str,
        cve_descriptions: str,
        max_tokens: int
    ) -> str:
        """
        Validate short report in single pass (no chunking).

        Args:
            report_text: Report text to validate
            cve_descriptions: All CVE descriptions
            max_tokens: Maximum tokens to generate

        Returns:
            str: Validation result
        """
        system_prompt = (
            "You are a CVE accuracy auditor with expertise in vulnerability analysis and security advisory verification. "
            "Your role is to rigorously validate whether CVEs cited in threat intelligence reports accurately reflect their official descriptions.\n\n"
            f"Official CVE Descriptions (Source of Truth):\n{cve_descriptions}\n\n"
            "Validation Criteria:\n"
            "A CVE is CORRECTLY used when:\n"
            "[OK] The vulnerability type matches (e.g., RCE, XSS, privilege escalation)\n"
            "[OK] The affected product/vendor matches\n"
            "[OK] The attack vector and impact align with official description\n"
            "[OK] The context of usage is consistent with the CVE's documented behavior\n\n"
            "A CVE is INCORRECTLY used when:\n"
            "[FAIL] Non-existent CVE ID (not found in official database)\n"
            "[FAIL] Mismatched vulnerability type (e.g., citing buffer overflow for an XSS flaw)\n"
            "[FAIL] Wrong affected product or vendor\n"
            "[FAIL] Misrepresenting severity, impact, or exploitability\n"
            "[FAIL] Confusing similar CVEs or incorrect attribution\n\n"
            "Validation Process:\n"
            "1. Extract each CVE mentioned in the report\n"
            "2. Match against official CVE descriptions\n"
            "3. Compare technical details: vulnerability type, affected product, attack vector\n"
            "4. Provide verdict: [OK] (Correct) or [FAIL] (Incorrect)\n"
            "5. Support verdict with direct quotes from both report and official description\n"
            "6. Use clear, evidence-based reasoning\n\n"
            "Output Format:\n"
            "CVE-YYYY-NNNNN: [[OK] Correct / [FAIL] Incorrect]\n"
            "Report states: \"[direct quote from report]\"\n"
            "Official description: \"[relevant quote from CVE database]\"\n"
            "Verdict: [Explanation with specific technical comparison]\n\n"
            "Be precise, objective, and evidence-based in your assessment."
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

    def _validate_chunked(
        self,
        report_text: str,
        cve_descriptions: str,
        max_tokens: int
    ) -> str:
        """
        Validate long report using chunking with optional CVE filtering.

        Optimization: If CVE filtering enabled, only sends relevant CVE descriptions
        to each chunk (reduces input tokens by ~50%, speeds up 7-10x).

        Args:
            report_text: Long report text to validate
            cve_descriptions: All CVE descriptions (triple-newline separated)
            max_tokens: Maximum tokens per chunk

        Returns:
            str: Combined validation results from all chunks
        """
        # Parse CVE descriptions into dictionary for fast lookup (if filtering enabled)
        cve_dict = {}
        if VALIDATION_ENABLE_CVE_FILTERING:
            for block in cve_descriptions.split('\n\n\n'):
                block = block.strip()
                if not block:
                    continue
                # Extract CVE ID from block (format: "-CVE Number: CVE-2025-1234")
                cve_match = re.search(r'CVE-\d{4}-\d{4,7}', block)
                if cve_match:
                    cve_id = cve_match.group(0)
                    cve_dict[cve_id] = block

        # Chunk the report text
        chunks = self._chunk_text_by_tokens(
            report_text,
            chunk_tokens=VALIDATION_CHUNK_TOKENS,
            overlap_tokens=VALIDATION_CHUNK_OVERLAP_TOKENS
        )

        if not chunks:
            return "Error: Could not chunk report for validation."

        if VERBOSE_LOGGING:
            print(f"[SEARCH] Validating {len(chunks)} chunks...")

        # Validate each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            if VERBOSE_LOGGING:
                print(f"[SEARCH] Processing chunk {i}/{len(chunks)}...")

            # Extract CVEs mentioned in this chunk
            chunk_cves = extract_cves_regex(chunk)

            # Filter CVE descriptions if enabled
            if VALIDATION_ENABLE_CVE_FILTERING and chunk_cves:
                # Only send relevant CVE descriptions
                filtered_cve_desc = "\n\n\n".join([
                    cve_dict[cve] for cve in chunk_cves if cve in cve_dict
                ])
                if not filtered_cve_desc:
                    filtered_cve_desc = "No matching CVE descriptions found for this chunk."
            elif VALIDATION_ENABLE_CVE_FILTERING:
                # No CVEs in this chunk
                filtered_cve_desc = "No CVEs mentioned in this chunk."
            else:
                # Filtering disabled: send all CVE descriptions
                filtered_cve_desc = cve_descriptions

            # Build system prompt with (filtered) CVE descriptions
            system_prompt = (
                "You are a CVE accuracy auditor with expertise in vulnerability analysis and security advisory verification. "
                "Your role is to rigorously validate whether CVEs cited in this section of the threat intelligence report accurately reflect their official descriptions.\n\n"
                f"Official CVE Descriptions (Source of Truth):\n{filtered_cve_desc}\n\n"
                "Validation Criteria:\n"
                "A CVE is CORRECTLY used when:\n"
                "[OK] Vulnerability type matches (e.g., RCE, XSS, privilege escalation)\n"
                "[OK] Affected product/vendor matches\n"
                "[OK] Attack vector and impact align with official description\n"
                "[OK] Usage context is consistent with CVE's documented behavior\n\n"
                "A CVE is INCORRECTLY used when:\n"
                "[FAIL] Non-existent CVE ID\n"
                "[FAIL] Mismatched vulnerability type\n"
                "[FAIL] Wrong affected product or vendor\n"
                "[FAIL] Misrepresenting severity or exploitability\n\n"
                "Validation Process:\n"
                "1. Extract each CVE mentioned in this section\n"
                "2. Match against official CVE descriptions\n"
                "3. Compare: vulnerability type, affected product, attack vector\n"
                "4. Provide verdict: [OK] (Correct) or [FAIL] (Incorrect)\n"
                "5. Support with direct quotes from both sources\n\n"
                "Output Format:\n"
                "CVE-YYYY-NNNNN: [[OK]/[FAIL]]\n"
                "Report: \"[quote]\"\n"
                "Official: \"[quote]\"\n"
                "Verdict: [evidence-based explanation]\n\n"
                "Be precise and objective."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please verify if the following CVEs have been used correctly in this Threat Intelligence Report:\n{chunk}"}
            ]

            result = self.llama.generate(
                messages=messages,
                max_new_tokens=max_tokens
            )

            chunk_results.append(result)

        # Stage 2: Optional final consolidation
        if VALIDATION_ENABLE_SECOND_STAGE:
            if VERBOSE_LOGGING:
                print(f"[SEARCH] Consolidating validation results into final report (Stage 2)...")

            # Combine all chunk results for consolidation
            combined_results = "\n\n".join([
                f"Chunk {i+1} validation:\n{result}"
                for i, result in enumerate(chunk_results)
            ])

            system_prompt = (
                "You are a senior CVE auditor creating a final consolidated validation report. "
                "Your task is to synthesize chunk-level validation results into a unified, authoritative assessment.\n\n"
                "Consolidation Requirements:\n"
                "1. Deduplicate CVEs: If the same CVE appears in multiple chunks, create ONE consolidated verdict\n"
                "2. Resolve conflicts: If verdicts differ across chunks, analyze evidence and provide final judgment\n"
                "3. Preserve evidence: Include the most compelling quotes from both report and official descriptions\n"
                "4. Maintain consistency: Use uniform format for all CVEs\n"
                "5. Sort by CVE ID: List in chronological order (by year and number)\n\n"
                "Final Report Format:\n"
                "=== CVE Validation Summary ===\n\n"
                "CVE-YYYY-NNNNN: [[OK] Correct / [FAIL] Incorrect]\n"
                "Report Context: \"[direct quote]\"\n"
                "Official Description: \"[relevant quote]\"\n"
                "Assessment: [Clear explanation of match/mismatch with specific technical details]\n\n"
                "[Repeat for each unique CVE]\n\n"
                "Summary Statistics:\n"
                "- Total CVEs validated: X\n"
                "- Correctly used: Y ([OK])\n"
                "- Incorrectly used: Z ([FAIL])\n\n"
                "Chunk-level validation results to consolidate:\n"
                f"{combined_results}\n\n"
                "Provide a professional, evidence-based consolidated validation report."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please consolidate these chunk-level validation results into a final validation report."}
            ]

            final_report = self.llama.generate(
                messages=messages,
                max_new_tokens=VALIDATION_FINAL_TOKENS
            )

            return final_report
        else:
            # Return all chunk results without consolidation
            return "\n\n".join([
                f"Chunk {i+1} validation:\n{result}"
                for i, result in enumerate(chunk_results)
            ])

    def answer_question_about_report(
        self,
        report_text: str,
        question: str,
        max_tokens: int = None
    ) -> str:
        """
        Answer a specific question about a report.

        For long documents, uses two-stage approach:
        - Stage 1: Answer question for each chunk (QA_TOKENS_PER_CHUNK tokens)
        - Stage 2: Consolidate all chunk answers into final response (QA_FINAL_TOKENS tokens)

        Args:
            report_text: Security report text
            question: User question
            max_tokens: Maximum tokens to generate (if None, uses QA_FINAL_TOKENS)

        Returns:
            str: Answer (consolidated if document is long, direct if short)
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")

        if max_tokens is None:
            max_tokens = QA_FINAL_TOKENS

        # Short document: use single-pass
        if len(report_text) <= QA_CHUNK_THRESHOLD_CHARS:
            if VERBOSE_LOGGING:
                print(f"[INFO] Q&A on short document ({len(report_text)} chars, single-pass)...")

            return self._answer_question_single_pass(report_text, question, max_tokens)

        # Long document: use two-stage chunked approach
        if VERBOSE_LOGGING:
            print(f"[INFO] Q&A on long document ({len(report_text)} chars, two-stage chunking)...")

        return self._answer_question_chunked(report_text, question, max_tokens)

    def _answer_question_single_pass(self, text: str, question: str, max_tokens: int) -> str:
        """Single-pass Q&A for short documents."""
        system_prompt = (
            "You are a technical security analyst specializing in threat intelligence report analysis. "
            "Your role is to answer questions accurately based on the provided security report content.\n\n"
            "Response Guidelines:\n"
            "- Answer based strictly on the provided text\n"
            "- Include direct quotes to support your answers\n"
            "- For CVE questions: Provide CVE ID, vulnerability type, affected systems, and impact\n"
            "- For technical questions: Include relevant technical details and security implications\n"
            "- If information is not in the text: Clearly state \"The report does not contain this information\"\n"
            "- Use precise technical terminology\n"
            "- Respond in the same language as the question\n\n"
            "Provide accurate, evidence-based answers with citations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{question}\n\nReport: {text}"}
        ]

        response = self.llama.generate(
            messages=messages,
            max_new_tokens=max_tokens
        )

        return response

    def _answer_question_chunked(self, report_text: str, question: str, max_tokens: int) -> str:
        """
        Two-stage Q&A for long documents.

        Stage 1: Answer question for each chunk
        Stage 2 (optional): Consolidate all chunk answers into final response

        Args:
            report_text: Long security report text
            question: User question
            max_tokens: Maximum tokens for final response

        Returns:
            str: Final answer (consolidated if second-stage enabled, concatenated if disabled)
        """
        # Stage 1: Chunk and answer question for each
        chunks = self._chunk_text_by_tokens(
            report_text,
            chunk_tokens=QA_CHUNK_TOKENS,
            overlap_tokens=QA_CHUNK_OVERLAP_TOKENS
        )

        if not chunks:
            return "Error: Could not chunk text for Q&A."

        if VERBOSE_LOGGING:
            print(f"[INFO] Answering question on {len(chunks)} chunks (Stage 1)...")

        chunk_answers = []
        for i, chunk in enumerate(chunks, 1):
            if VERBOSE_LOGGING:
                print(f"[INFO] Processing chunk {i}/{len(chunks)}...")

            system_prompt = (
                "You are a technical security analyst extracting information from a section of a threat intelligence report. "
                "Your task is to answer the question based strictly on this text section.\n\n"
                "Response Guidelines:\n"
                "- If the section contains relevant information: Provide a concise, accurate answer with direct quotes\n"
                "- If the section does NOT contain relevant information: State \"Not found in this section\"\n"
                "- Include CVE IDs, technical details, and security context when present\n"
                "- Be precise and avoid speculation beyond the text\n\n"
                "Answer based on this section only."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{question}\n\nText section: {chunk}"}
            ]

            answer = self.llama.generate(
                messages=messages,
                max_new_tokens=QA_TOKENS_PER_CHUNK
            )

            chunk_answers.append(f"Section {i}: {answer}")

        # Stage 2: Optional consolidation
        if QA_ENABLE_SECOND_STAGE:
            if VERBOSE_LOGGING:
                print(f"[INFO] Consolidating {len(chunk_answers)} answers (Stage 2)...")

            all_answers = "\n\n".join(chunk_answers)

            system_prompt = (
                "You are a senior security analyst synthesizing information from multiple sections of a threat intelligence report. "
                "Your task is to consolidate section-level answers into a unified, comprehensive response.\n\n"
                "Consolidation Guidelines:\n"
                "1. Synthesize information: Merge related points from different sections\n"
                "2. Remove redundancy: Eliminate duplicate information while preserving unique details\n"
                "3. Preserve citations: Keep direct quotes and specific references (CVE IDs, product names, etc.)\n"
                "4. Handle \"Not found\" responses: If all sections say \"Not found\", state \"The report does not contain this information\"\n"
                "5. Maintain structure: Organize the answer logically (e.g., Overview → Details → Implications)\n"
                "6. Use precise language: Employ technical terminology appropriate to the question\n\n"
                "Provide a clear, accurate, and comprehensive answer that fully addresses the question."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Original question: {question}\n\n"
                    f"Answers from different sections:\n{all_answers}\n\n"
                    f"Please consolidate these answers into one comprehensive response. "
                    f"If multiple sections say 'Not found', respond with 'Information not found in the document.'"
                )}
            ]

            final_answer = self.llama.generate(
                messages=messages,
                max_new_tokens=max_tokens
            )

            return final_answer
        else:
            # Return all chunk answers concatenated
            return "\n\n".join(chunk_answers)

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
            print(f"[WARNING] Could not find {len(missing_cves)} CVEs: {', '.join(missing_cves)}")

        return report_text, cves, cve_descriptions_text

    def cleanup(self):
        """Clean up resources."""
        if self.llama:
            self.llama.cleanup()
        if self.embedder:
            self.embedder.cleanup()

        self._initialized = False

        if VERBOSE_LOGGING:
            print("[OK] RAG system cleaned up")


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
