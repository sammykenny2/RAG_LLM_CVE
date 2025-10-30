"""
Chroma vector database management module.
Provides CRUD operations for managing CVE and PDF embeddings.
"""

import chromadb
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union

# Import configuration
from config import (
    EMBEDDING_PATH,
    RETRIEVAL_TOP_K,
    VERBOSE_LOGGING
)


class ChromaManager:
    """
    Manager for Chroma vector database operations.

    Metadata schema:
        source_type: 'pdf' or 'cve'
        source_name: filename or CVE list name
        added_date: ISO format timestamp
        chunk_index: int (for chunked documents)
        precision: 'float32' or 'float16'

    Example usage:
        manager = ChromaManager()
        manager.add_documents(
            texts=["chunk1", "chunk2"],
            embeddings=[emb1, emb2],
            metadata=[{"source_type": "pdf", "source_name": "report.pdf"}, ...]
        )
        results = manager.query("vulnerability description", top_k=5)
        manager.delete_by_source("report.pdf")
    """

    def __init__(self, db_path: Union[str, Path] = None, collection_name: str = "cve_embeddings"):
        """
        Initialize Chroma manager.

        Args:
            db_path: Path to Chroma database directory (default from config)
            collection_name: Name of collection to use
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Use EMBEDDING_PATH + .chroma suffix for consistency
            # EMBEDDING_PATH is base path (e.g., ./CVEEmbeddings)
            # Chroma database should be ./CVEEmbeddings.chroma/
            base_path = Path(EMBEDDING_PATH)
            self.db_path = Path(f"{base_path}.chroma")

        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialized = False

    def initialize(self, create_if_not_exists: bool = True):
        """
        Initialize Chroma client and collection.

        Args:
            create_if_not_exists: Create collection if it doesn't exist
        """
        if self._initialized:
            if VERBOSE_LOGGING:
                print("[WARNING] Chroma manager already initialized, skipping...")
            return

        if VERBOSE_LOGGING:
            print(f"Initializing Chroma database: {self.db_path}")

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            if VERBOSE_LOGGING:
                print(f"[OK] Loaded existing collection: {self.collection_name}")
        except Exception:
            if create_if_not_exists:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "CVE and PDF embeddings for RAG system"}
                )
                if VERBOSE_LOGGING:
                    print(f"[OK] Created new collection: {self.collection_name}")
            else:
                raise RuntimeError(f"Collection '{self.collection_name}' not found")

        self._initialized = True

    def get_collection(self):
        """
        Get Chroma collection (initializes if not already loaded).

        Returns:
            Collection: Chroma collection object
        """
        if not self._initialized:
            self.initialize()
        return self.collection

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> int:
        """
        Add documents to Chroma collection.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata dicts (one per document)
            ids: List of unique IDs (auto-generated if None)

        Returns:
            int: Number of documents added
        """
        if not self._initialized:
            self.initialize()

        n_docs = len(texts)

        # Generate IDs if not provided
        if ids is None:
            # Get next ID by counting existing documents
            current_count = self.collection.count()
            ids = [f"doc_{current_count + i}" for i in range(n_docs)]

        # Generate metadata if not provided
        if metadata is None:
            metadata = [
                {
                    "source_type": "unknown",
                    "source_name": "unknown",
                    "added_date": datetime.now().isoformat(),
                    "chunk_index": i,
                    "precision": "float32"
                }
                for i in range(n_docs)
            ]
        else:
            # Fill in missing fields
            for i, meta in enumerate(metadata):
                if "added_date" not in meta:
                    meta["added_date"] = datetime.now().isoformat()
                if "chunk_index" not in meta:
                    meta["chunk_index"] = i

        # Add to collection in batches (Chroma has 5000 batch limit)
        batch_size = 5000
        for i in range(0, n_docs, batch_size):
            batch_end = min(i + batch_size, n_docs)
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadata[i:batch_end]
            )

        if VERBOSE_LOGGING:
            print(f"[OK] Added {n_docs} documents to collection")

        return n_docs

    def query(
        self,
        query_embedding: List[float],
        top_k: int = None,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query similar documents from collection.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (default from config)
            where: Metadata filter dict (e.g., {"source_type": "pdf"})

        Returns:
            dict: Query results with keys: ids, documents, metadatas, distances
        """
        if not self._initialized:
            self.initialize()

        top_k = top_k if top_k is not None else RETRIEVAL_TOP_K

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )

        return results

    def query_by_metadata(
        self,
        where: Dict,
        limit: int = None
    ) -> Dict:
        """
        Query documents by metadata filter only (no embedding search).
        Useful for exact CVE ID lookup.

        Args:
            where: Metadata filter dict (e.g., {"cve_id": "CVE-2024-1234"})
            limit: Maximum number of results (default: all matches)

        Returns:
            dict: Results with keys: ids, documents, metadatas
        """
        if not self._initialized:
            self.initialize()

        # Use get() for metadata-only queries (faster than query())
        results = self.collection.get(
            where=where,
            limit=limit,
            include=['documents', 'metadatas']
        )

        return results

    def delete_by_source(self, source_name: str, source_type: str = None) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source_name: Name of source to delete (e.g., "report.pdf")
            source_type: Optional type filter ('pdf' or 'cve')

        Returns:
            int: Number of documents deleted
        """
        if not self._initialized:
            self.initialize()

        # Build where filter
        where_filter = {"source_name": source_name}
        if source_type:
            where_filter["source_type"] = source_type

        # Get IDs matching filter
        results = self.collection.get(where=where_filter)
        ids_to_delete = results['ids']

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            if VERBOSE_LOGGING:
                print(f"[OK] Deleted {len(ids_to_delete)} documents from source: {source_name}")
            return len(ids_to_delete)
        else:
            if VERBOSE_LOGGING:
                print(f"[WARNING] No documents found for source: {source_name}")
            return 0

    def delete_by_year(self, year: int, schema: str = None) -> int:
        """
        Delete all CVE documents from a specific year.

        Args:
            year: Year to delete (e.g., 2025)
            schema: Optional schema filter ('v5', 'v4', or 'all' for both)

        Returns:
            int: Number of documents deleted
        """
        if not self._initialized:
            self.initialize()

        # Determine which source_names to delete based on schema
        source_names_to_delete = []
        if schema == 'v5':
            source_names_to_delete = [f"CVE_{year}_v5"]
        elif schema == 'v4':
            source_names_to_delete = [f"CVE_{year}_v4"]
        else:  # 'all' or None - delete both
            source_names_to_delete = [f"CVE_{year}_v5", f"CVE_{year}_v4"]

        total_deleted = 0

        for source_name in source_names_to_delete:
            # Get IDs matching this source_name
            results = self.collection.get(
                where={
                    "source_type": "cve",
                    "source_name": source_name
                }
            )
            ids_to_delete = results['ids']

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                print(f"  [OK] Deleted {len(ids_to_delete)} documents from {source_name}")
                total_deleted += len(ids_to_delete)

        if total_deleted > 0:
            print(f"[OK] Total deleted for year {year}: {total_deleted} documents")
        else:
            print(f"[WARNING] No documents found for year {year}")

        return total_deleted

    def get_stats(self) -> Dict:
        """
        Get statistics about the collection.

        Returns:
            dict: Statistics including:
                - total_docs: Total document count
                - by_source_type: Count by source type
                - sources: List of unique source names with counts
        """
        if not self._initialized:
            self.initialize()

        total_docs = self.collection.count()

        # Get all metadata
        all_data = self.collection.get()
        metadatas = all_data.get('metadatas', [])

        # Count by source type
        by_source_type = {}
        sources = {}

        for meta in metadatas:
            # Handle None metadata (documents added without metadata)
            if meta is None:
                meta = {}

            source_type = meta.get('source_type', 'unknown')
            source_name = meta.get('source_name', 'unknown')

            # Count by type
            by_source_type[source_type] = by_source_type.get(source_type, 0) + 1

            # Count by source name
            if source_name not in sources:
                sources[source_name] = {
                    'type': source_type,
                    'count': 0,
                    'added_date': meta.get('added_date', 'unknown')
                }
            sources[source_name]['count'] += 1

        return {
            'total_docs': total_docs,
            'by_source_type': by_source_type,
            'sources': sources
        }

    def list_sources(self) -> List[Dict]:
        """
        List all unique sources in the collection.

        Returns:
            list: List of source dicts with keys: name, type, count, added_date
        """
        stats = self.get_stats()
        sources = stats['sources']

        # Convert to list format
        source_list = [
            {
                'name': name,
                'type': info['type'],
                'count': info['count'],
                'added_date': info['added_date']
            }
            for name, info in sources.items()
        ]

        # Sort by added_date (most recent first)
        source_list.sort(key=lambda x: x['added_date'], reverse=True)

        return source_list

    def reset_collection(self):
        """Delete and recreate collection (USE WITH CAUTION)."""
        if not self._initialized:
            self.initialize(create_if_not_exists=False)

        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "CVE and PDF embeddings for RAG system"}
        )

        if VERBOSE_LOGGING:
            print(f"[WARNING] Collection '{self.collection_name}' has been reset")


# =============================================================================
# Utility functions (backward compatible)
# =============================================================================

def get_chroma_collection(db_path: Union[str, Path] = None, collection_name: str = "cve_embeddings"):
    """
    Get Chroma collection (backward compatible function).

    Args:
        db_path: Path to database
        collection_name: Name of collection

    Returns:
        Collection: Chroma collection object
    """
    manager = ChromaManager(db_path=db_path, collection_name=collection_name)
    return manager.get_collection()
