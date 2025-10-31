"""
Session manager module for multi-file conversation context.
Manages temporary file uploads with session-scoped Chroma collections.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import uuid

import chromadb

# Import core modules
from core.pdf_processor import PDFProcessor
from core.embeddings import EmbeddingModel

# Import configuration
from config import (
    TEMP_UPLOAD_DIR,
    SESSION_MAX_FILES,
    SESSION_MAX_FILE_SIZE_MB,
    CHUNK_SIZE,
    CHUNK_OVERLAP_RATIO,
    EMBEDDING_PRECISION,
    VERBOSE_LOGGING,
    RETRIEVAL_TOP_K
)


class SessionManager:
    """
    Manages session-scoped file uploads and embeddings for multi-file conversations.

    Architecture:
        - Each session has a unique ID
        - Files are stored in temp_uploads/session_{id}/
        - Embeddings stored in session-scoped Chroma collection
        - Automatic cleanup on session end

    Example usage:
        session = SessionManager(session_id="abc123")
        file_info = session.add_file("report.pdf")
        results = session.query("What are the CVEs?", top_k=5)
        session.remove_file("report.pdf")
        session.cleanup()
    """

    def __init__(
        self,
        session_id: str,
        db_base_path: Path = None
    ):
        """
        Initialize session manager.

        Args:
            session_id: Unique session identifier
            db_base_path: Base path for Chroma databases (default: TEMP_UPLOAD_DIR)
        """
        self.session_id = session_id
        self.collection_name = f"session_{session_id}"

        # Set up paths
        if db_base_path is None:
            db_base_path = TEMP_UPLOAD_DIR

        self.session_dir = Path(db_base_path) / f"session_{session_id}"
        self.db_path = self.session_dir / "chroma_db"

        # Create session directory
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            if VERBOSE_LOGGING:
                print(f"[OK] Loaded existing session collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": f"Session {session_id} temporary files"}
            )
            if VERBOSE_LOGGING:
                print(f"[OK] Created new session collection: {self.collection_name}")

        # Initialize embedding model (shared across files)
        self.embedder = EmbeddingModel()
        self.embedder.initialize()

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()

        # Track uploaded files
        self.files: Dict[str, Dict] = {}  # {filename: file_info}

        if VERBOSE_LOGGING:
            print(f"SessionManager initialized:")
            print(f"  └─ Session ID: {self.session_id}")
            print(f"  └─ Collection: {self.collection_name}")
            print(f"  └─ Session directory: {self.session_dir}")

    def add_file(self, file_path: str) -> Dict:
        """
        Add a file to the session.

        Args:
            file_path: Path to file to add

        Returns:
            dict: File info object with keys:
                - name: filename
                - path: full path
                - status: 'ready', 'processing', 'error'
                - chunks: number of chunks
                - added_date: ISO timestamp
                - error: error message (if status='error')

        Raises:
            ValueError: If file size exceeds limit or max files reached
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = file_path.name

        # Check max files limit
        if len(self.files) >= SESSION_MAX_FILES:
            raise ValueError(
                f"Maximum {SESSION_MAX_FILES} files per session. "
                f"Remove a file before adding another."
            )

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > SESSION_MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File size ({file_size_mb:.1f} MB) exceeds limit "
                f"({SESSION_MAX_FILE_SIZE_MB} MB)"
            )

        # Check if already exists
        if filename in self.files:
            raise ValueError(f"File already exists in session: {filename}")

        # Initialize file info
        file_info = {
            "name": filename,
            "path": str(file_path),
            "status": "processing",
            "chunks": 0,
            "added_date": datetime.now().isoformat(),
            "error": None
        }

        self.files[filename] = file_info

        try:
            if VERBOSE_LOGGING:
                print(f"Processing file: {filename}")

            # Extract text from PDF
            text = self.pdf_processor.extract_text(file_path)

            if VERBOSE_LOGGING:
                print(f"  └─ Extracted {len(text)} characters")

            # Split into chunks with overlap
            chunks = self._split_text_with_overlap(text)

            if VERBOSE_LOGGING:
                print(f"  └─ Split into {len(chunks)} chunks")

            # Generate embeddings
            embeddings = self.embedder.encode(
                chunks,
                show_progress_bar=VERBOSE_LOGGING,
                precision=EMBEDDING_PRECISION
            )

            if VERBOSE_LOGGING:
                print(f"  └─ Generated {len(embeddings)} embeddings")

            # Convert to list for Chroma
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()

            # Generate IDs and metadata
            current_count = self.collection.count()
            ids = [f"doc_{current_count + i}" for i in range(len(chunks))]

            metadatas = [
                {
                    "source_type": "session",
                    "source_name": filename,
                    "added_date": file_info["added_date"],
                    "chunk_index": i,
                    "session_id": self.session_id
                }
                for i in range(len(chunks))
            ]

            # Add to Chroma (batch if needed)
            batch_size = 5000
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=chunks[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

            # Update file info
            file_info["status"] = "ready"
            file_info["chunks"] = len(chunks)

            if VERBOSE_LOGGING:
                print(f"[OK] File added successfully: {filename} ({len(chunks)} chunks)")

            # Clear GPU cache after embedding to free VRAM for subsequent LLM operations
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                if VERBOSE_LOGGING:
                    print(f"[OK] GPU cache cleared after embedding")

            return file_info

        except Exception as e:
            # Update file info with error
            file_info["status"] = "error"
            file_info["error"] = str(e)

            if VERBOSE_LOGGING:
                print(f"[ERROR] Error processing file: {filename}")
                print(f"  └─ {e}")

            return file_info

    def remove_file(self, filename: str) -> bool:
        """
        Remove a file from the session.

        Args:
            filename: Name of file to remove

        Returns:
            bool: True if removed, False if not found
        """
        if filename not in self.files:
            if VERBOSE_LOGGING:
                print(f"[WARNING] File not found in session: {filename}")
            return False

        try:
            # Get documents matching this filename
            results = self.collection.get(
                where={"source_name": filename}
            )

            ids_to_delete = results.get('ids', [])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                if VERBOSE_LOGGING:
                    print(f"[OK] Deleted {len(ids_to_delete)} chunks from: {filename}")

            # Remove from tracking
            del self.files[filename]

            return True

        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"[ERROR] Error removing file: {filename}")
                print(f"  └─ {e}")
            return False

    def query(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        """
        Query session files for relevant context.

        Args:
            question: Question to search for
            top_k: Number of results to return

        Returns:
            list: List of result dicts with keys:
                - text: chunk text
                - source: filename
                - source_type: 'session'
                - score: similarity score (1 - distance)
                - metadata: full metadata dict
        """
        if self.collection.count() == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedder.encode(
            [question],
            convert_to_numpy=False
        )

        # Convert to list for Chroma
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()[0]

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        # Format results
        formatted_results = []

        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            formatted_results.append({
                "text": doc,
                "source": meta.get("source_name", "unknown"),
                "source_type": "session",
                "score": 1 - distance,  # Convert distance to similarity
                "metadata": meta
            })

        return formatted_results

    def list_files(self) -> List[Dict]:
        """
        List all files in the session.

        Returns:
            list: List of file info dicts
        """
        return list(self.files.values())

    def cleanup(self):
        """
        Clean up session: delete collection and temporary files.
        """
        try:
            # Delete Chroma collection
            self.client.delete_collection(self.collection_name)
            if VERBOSE_LOGGING:
                print(f"[OK] Deleted collection: {self.collection_name}")
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"[WARNING] Error deleting collection: {e}")

        try:
            # Delete session directory
            if self.session_dir.exists():
                shutil.rmtree(self.session_dir)
                if VERBOSE_LOGGING:
                    print(f"[OK] Deleted session directory: {self.session_dir}")
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"[WARNING] Error deleting session directory: {e}")

        # Clean up embedding model
        try:
            self.embedder.cleanup()
        except Exception as e:
            if VERBOSE_LOGGING:
                print(f"[WARNING] Error cleaning up embedder: {e}")

        if VERBOSE_LOGGING:
            print(f"[OK] Session cleaned up: {self.session_id}")

    def _split_text_with_overlap(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks (sentences).

        Args:
            text: Full text to split

        Returns:
            list: List of text chunks
        """
        # Split into sentences (simple approach using '. ')
        sentences = [s.strip() + '.' for s in text.split('. ') if s.strip()]

        if len(sentences) == 0:
            return [text]  # Return as single chunk if no sentences

        # Calculate overlap
        chunk_size = CHUNK_SIZE
        overlap_ratio = CHUNK_OVERLAP_RATIO
        step_size = max(1, int(chunk_size * (1 - overlap_ratio)))

        # Create overlapping chunks
        chunks = []
        for i in range(0, len(sentences), step_size):
            chunk_sentences = sentences[i:i + chunk_size]
            if chunk_sentences:  # Only add non-empty chunks
                chunk_text = ' '.join(chunk_sentences)
                chunks.append(chunk_text)

        return chunks
